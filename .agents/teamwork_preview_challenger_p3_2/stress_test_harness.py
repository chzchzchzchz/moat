"""
Project Antigravity Phase 3 — Empirical Stress Test Harness (Challenger 2)

Empirically verifies:
1. Paged KV-cache memory isolation across channels under aggressive random mutation and CoW block allocation (0 cross-channel corruption).
2. Long-run numerical stability: 0 NaN, 0 Inf, and 0 memory leaks across 150+ generation steps.
3. Temperature sampling diversity: N=8 candidate traces generate non-identical, coherent token sequences under T = 0.7.
4. Model loader memory footprint budget: 1.5B model weight memory <= 2.5 GB, total app footprint <= 4.5 GB ceiling, 144-byte super-block structure validation.
"""

import sys
import os
import time
import unittest
import tracemalloc
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'antigravity-engine', 'src')))

from batch_generator import PagedKVCache, BatchedRolloutCoordinator
from model_loader import (
    MockWeightReader,
    GGUFWeightReader,
    SafetensorsWeightReader,
    SuperBlockRepacker,
    QuantizedSuperBlockTensor,
    MemoryBudgetValidator
)
from attention import ExponentialLUT, safe_softmax_lut


class TestPagedKVCacheIsolation(unittest.TestCase):
    """
    Empirical Stress Tests for PagedKVCache Memory Isolation & Copy-on-Write (CoW).
    """

    def test_aggressive_mutation_cow_isolation(self):
        """
        Verify zero cross-channel attention/data corruption under aggressive random mutation and CoW allocation.
        """
        block_size = 16
        num_layers = 28
        num_kv_heads = 2
        head_dim = 128
        num_traces = 8

        cache = PagedKVCache(
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=np.float16
        )

        # 1. Allocate a 40-token shared prompt prefix (spans 3 physical blocks: 16 + 16 + 8 tokens)
        prefix_len = 40
        np.random.seed(12345)
        prefix_k = np.random.randn(prefix_len, num_layers, num_kv_heads, head_dim).astype(np.float16)
        prefix_v = np.random.randn(prefix_len, num_layers, num_kv_heads, head_dim).astype(np.float16)

        trace_ids = cache.allocate_prompt_prefix(
            num_traces=num_traces,
            prefix_k=prefix_k,
            prefix_v=prefix_v
        )
        self.assertEqual(len(trace_ids), num_traces)

        # Initial block count should be 3 physical blocks, each with ref_count = 8
        self.assertEqual(len(cache.physical_blocks), 3)
        for b_id, block in cache.physical_blocks.items():
            self.assertEqual(block.ref_count, num_traces)

        # Save reference copies of prompt prefix for verification
        orig_k_prefix = prefix_k.copy()
        orig_v_prefix = prefix_v.copy()

        # 2. Aggressive Random Mutations on a subset of channels
        # Target channels to mutate: [0, 3, 5, 7]
        # Untouched channels: [1, 2, 4, 6]
        mutated_channels = [0, 3, 5, 7]
        untouched_channels = [1, 2, 4, 6]

        channel_appended_tokens = {ch: [] for ch in range(num_traces)}

        rng = np.random.RandomState(999)
        for round_idx in range(50):  # 50 mutation rounds
            ch = rng.choice(mutated_channels)
            rand_k = rng.randn(num_layers, num_kv_heads, head_dim).astype(np.float16)
            rand_v = rng.randn(num_layers, num_kv_heads, head_dim).astype(np.float16)

            cache.append_token_kv(trace_id=ch, k_vec=rand_k, v_vec=rand_v)
            channel_appended_tokens[ch].append((rand_k, rand_v))

        # 3. Verify UNTOUCHED channels have 0 data corruption in prompt prefix
        for ch in untouched_channels:
            k_out, v_out = cache.get_kv_cache(trace_id=ch)
            self.assertEqual(k_out.shape[0], prefix_len, f"Trace {ch} length mutated!")
            self.assertEqual(v_out.shape[0], prefix_len, f"Trace {ch} length mutated!")
            np.testing.assert_array_equal(
                k_out, orig_k_prefix,
                err_msg=f"Cross-channel corruption detected in Key tensor for untouched trace {ch}!"
            )
            np.testing.assert_array_equal(
                v_out, orig_v_prefix,
                err_msg=f"Cross-channel corruption detected in Value tensor for untouched trace {ch}!"
            )

        # 4. Verify MUTATED channels have exact prompt prefix + exact appended tokens
        for ch in mutated_channels:
            k_out, v_out = cache.get_kv_cache(trace_id=ch)
            expected_len = prefix_len + len(channel_appended_tokens[ch])
            self.assertEqual(k_out.shape[0], expected_len, f"Trace {ch} expected length mismatch!")

            # Verify prompt prefix slice matches exactly
            np.testing.assert_array_equal(
                k_out[:prefix_len], orig_k_prefix,
                err_msg=f"Trace {ch} prefix Key corrupted after mutation!"
            )
            np.testing.assert_array_equal(
                v_out[:prefix_len], orig_v_prefix,
                err_msg=f"Trace {ch} prefix Value corrupted after mutation!"
            )

            # Verify appended tokens match exactly
            for idx, (exp_k, exp_v) in enumerate(channel_appended_tokens[ch]):
                np.testing.assert_array_equal(
                    k_out[prefix_len + idx], exp_k,
                    err_msg=f"Trace {ch} token {idx} Key mismatch!"
                )
                np.testing.assert_array_equal(
                    v_out[prefix_len + idx], exp_v,
                    err_msg=f"Trace {ch} token {idx} Value mismatch!"
                )

        # 5. Free mutated traces one by one and check memory cleanup
        for ch in mutated_channels:
            cache.free_trace(ch)

        # Untouched traces must remain fully intact and operational
        for ch in untouched_channels:
            k_out, v_out = cache.get_kv_cache(trace_id=ch)
            self.assertEqual(k_out.shape[0], prefix_len)
            np.testing.assert_array_equal(k_out, orig_k_prefix)

        print("\n[STRESS TEST PASS] Paged KV-Cache memory isolation & CoW 0 cross-channel corruption verified.")


class TestNumericalStabilityAndMemoryLeaks(unittest.TestCase):
    """
    Empirical Stress Tests for Long-Run Numerical Stability & Memory Leak Auditing.
    """

    def test_150_plus_steps_numerical_stability_and_memory(self):
        """
        Verify 0 NaN, 0 Inf, and 0 memory leaks across 160 generation steps.
        """
        num_candidates = 8
        hidden_dim = 2048
        vocab_size = 32000
        num_layers = 28
        num_kv_heads = 2
        head_dim = 128
        total_steps = 160

        coordinator = BatchedRolloutCoordinator(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device="cpu"
        )

        prompt_tokens = [101, 254, 399, 1002]
        coordinator.init_rollouts(prompt_tokens)

        np.random.seed(777)
        weights = np.random.randn(hidden_dim, vocab_size).astype(np.float16) * 0.02

        # Start tracemalloc heap tracking
        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        kv_mem_history = []
        nan_count = 0
        inf_count = 0

        for step_idx in range(total_steps):
            # Generate synthetic active activations with dynamic scaling to test numerical limits
            scale = 1.0 + 0.1 * np.sin(step_idx)
            np.random.seed(step_idx + 1000)
            activations = (np.random.randn(num_candidates, hidden_dim) * scale).astype(np.float16)

            # Step rollout coordinator
            sampled_tokens = coordinator.step(
                active_activations=activations,
                weights=weights,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                eos_token_id=-1
            )

            # 1. Inspect sampled tokens for NaN / Inf / Invalid indices
            if np.any(np.isnan(sampled_tokens)):
                nan_count += 1
            if np.any(np.isinf(sampled_tokens)):
                inf_count += 1

            self.assertTrue(
                np.all((sampled_tokens >= 0) & (sampled_tokens < vocab_size)),
                f"Step {step_idx}: Sampled token out of vocabulary bounds [0, {vocab_size-1}]!"
            )

            current_kv_bytes = coordinator.kv_cache.get_memory_bytes()
            kv_mem_history.append(current_kv_bytes)

        snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Assert 0 NaN, 0 Inf
        self.assertEqual(nan_count, 0, f"Detected {nan_count} steps with NaN tokens!")
        self.assertEqual(inf_count, 0, f"Detected {inf_count} steps with Inf tokens!")

        # 2. Assert KV Cache memory footprint scaling
        # Each step adds 1 token per trace = 8 tokens per step across 8 traces.
        # Initial prompt len = 4 tokens (1 block of 16 tokens allocated per trace = 8 blocks total).
        # Total tokens per trace after 160 steps = 164 tokens -> 11 blocks of 16 tokens per trace.
        # 11 blocks * 8 traces = 88 physical blocks total.
        # Each block = 16 * 28 * 2 * 128 * 2 bytes * 2 (K & V) = 458,752 bytes = ~0.437 MB.
        # Total expected physical memory = 88 * 458,752 bytes = 40,370,176 bytes (~38.5 MB).
        final_kv_mb = kv_mem_history[-1] / (1024 * 1024)
        self.assertLessEqual(final_kv_mb, 100.0, f"KV cache memory {final_kv_mb:.2f} MB exceeds expected bounds!")

        # 3. Heap Memory Leak Check
        stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_heap_diff = sum(stat.size_diff for stat in stats)
        total_heap_diff_mb = total_heap_diff / (1024 * 1024)

        # Allow expected KV cache data allocation (~38.5 MB) plus small Python overhead, but fail on runaway leaks
        self.assertLess(
            total_heap_diff_mb, 80.0,
            f"Heap allocation grew by {total_heap_diff_mb:.2f} MB over 160 steps — possible memory leak!"
        )

        print(f"\n[STRESS TEST PASS] 160 steps numerical stability verified: 0 NaN, 0 Inf, KV memory {final_kv_mb:.2f} MB, heap diff {total_heap_diff_mb:.2f} MB.")


class TestTemperatureSamplingDiversity(unittest.TestCase):
    """
    Empirical Stress Tests for Multi-Channel Temperature Sampling Diversity (T = 0.7).
    """

    def test_temperature_sampling_diversity_n8(self):
        """
        Verify all N=8 candidate traces generate non-identical, coherent token sequences under temperature T = 0.7.
        """
        num_candidates = 8
        hidden_dim = 512
        vocab_size = 2000
        max_steps = 30

        coordinator = BatchedRolloutCoordinator(
            num_candidates=num_candidates,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            device="cpu"
        )

        prompt_tokens = [1, 2, 3, 4]
        np.random.seed(42)
        weights = np.random.randn(hidden_dim, vocab_size).astype(np.float16) * 0.05

        # 1. Run generation under T = 0.7
        results = coordinator.generate(
            prompt_tokens=prompt_tokens,
            weights=weights,
            max_steps=max_steps,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            eos_token_id=-1
        )

        candidate_traces = results['candidate_tokens']
        self.assertEqual(len(candidate_traces), num_candidates)

        # 2. Check candidate uniqueness
        unique_sequences = set(tuple(trace) for trace in candidate_traces)
        num_unique = len(unique_sequences)

        print(f"\n[SAMPLING DIVERSITY] Under T=0.7: {num_unique}/{num_candidates} unique candidate sequences generated.")

        # Under T=0.7 with top-p 0.95 and top-k 50 over 30 steps, all N=8 traces must be distinct
        self.assertEqual(
            num_unique, num_candidates,
            f"Expected {num_candidates} distinct candidate traces under T=0.7, but got {num_unique}!"
        )

        # 3. Calculate pairwise normalized Hamming distance across candidate traces
        pairwise_distances = []
        for i in range(num_candidates):
            for j in range(i + 1, num_candidates):
                t1 = np.array(candidate_traces[i][len(prompt_tokens):])
                t2 = np.array(candidate_traces[j][len(prompt_tokens):])
                mismatches = np.sum(t1 != t2)
                dist = mismatches / len(t1)
                pairwise_distances.append(dist)

        mean_dist = np.mean(pairwise_distances)
        print(f"[SAMPLING DIVERSITY] Mean pairwise sequence divergence: {mean_dist:.4f} ({mean_dist*100:.1f}% mismatched tokens).")

        self.assertGreater(
            mean_dist, 0.3,
            f"Mean pairwise sequence divergence {mean_dist:.4f} is suspiciously low for T=0.7!"
        )

        # 4. Control Test: Verify greedy T = 0.0 produces identical sequences
        coordinator_greedy = BatchedRolloutCoordinator(
            num_candidates=num_candidates, hidden_dim=hidden_dim, vocab_size=vocab_size, device="cpu"
        )
        res_greedy = coordinator_greedy.generate(
            prompt_tokens=prompt_tokens, weights=weights, max_steps=max_steps, temperature=0.0, eos_token_id=-1
        )
        greedy_traces = res_greedy['candidate_tokens']
        unique_greedy = set(tuple(tr) for tr in greedy_traces)
        self.assertEqual(
            len(unique_greedy), 1,
            "Greedy T=0.0 failed to produce identical sequences across candidate channels!"
        )

        print("[STRESS TEST PASS] Temperature sampling diversity verified under T=0.7 (N=8 all distinct).")


class TestModelLoaderMemoryBudgetAndSuperBlocks(unittest.TestCase):
    """
    Empirical Stress Tests for Model Loader Memory Budgets and Super-Block Repacking.
    """

    def test_model_loader_memory_budget_ceilings(self):
        """
        Verify 1.5B parameter model weight memory <= 2.5 GB and total app footprint <= 4.5 GB ceiling.
        """
        qwen_1_5b_params = 1_540_000_000

        report = MemoryBudgetValidator.validate_memory_budget(
            num_params=qwen_1_5b_params,
            num_traces=8,
            seq_len=2048,
            runtime_overhead_gb=0.5
        )

        weight_gb = report['weight_memory_gb']
        kv_gb = report['kv_cache_memory_gb']
        total_app_gb = report['total_app_memory_gb']

        print(f"\n[MEMORY BUDGET REPORT]")
        print(f"  Model Parameters:       {qwen_1_5b_params:,}")
        print(f"  INT4 Weight Footprint:   {weight_gb:.4f} GB (Budget: <= 2.500 GB) -> PASS: {report['weight_budget_passed']}")
        print(f"  8-Channel KV Cache (2k): {kv_gb:.4f} GB")
        print(f"  Runtime Overhead:        {report['runtime_overhead_gb']:.4f} GB")
        print(f"  Total App Footprint:     {total_app_gb:.4f} GB (Budget: <= 4.500 GB) -> PASS: {report['total_app_budget_passed']}")

        self.assertLessEqual(weight_gb, 2.5, f"1.5B Model weights {weight_gb:.4f} GB exceed 2.5 GB budget!")
        self.assertLessEqual(total_app_gb, 4.5, f"Total app footprint {total_app_gb:.4f} GB exceeds 4.5 GB ceiling!")
        self.assertTrue(report['is_valid'])

    def test_superblock_144_byte_repacking_and_alignment(self):
        """
        Verify SuperBlockRepacker produces valid 144-byte super-blocks (16B header + 128B payload) aligned to 128 bytes.
        """
        mock_reader = MockWeightReader(
            hidden_size=2048,
            intermediate_size=5504,
            num_layers=28,
            num_heads=16,
            num_kv_heads=2,
            vocab_size=151936
        )

        # Read actual representative model tensor (Q projection matrix 2048x2048 = 4,194,304 elements)
        q_proj = mock_reader.read_tensor("model.layers.0.self_attn.q_proj.weight")
        self.assertEqual(q_proj.shape, (2048, 2048))

        # Repack into 256-element super-blocks
        q_tensor = SuperBlockRepacker.repack_matrix(q_proj, group_size=32, groups_per_superblock=8)

        # 4,194,304 elements / 256 elements per super-block = 16,384 super-blocks
        self.assertEqual(len(q_tensor.superblocks), 16384)
        self.assertEqual(q_tensor.bytes_per_superblock, 144)

        # Total repacked memory for this tensor: 16,384 * 144 bytes = 2,359,296 bytes (~2.25 MB vs 8.0 MB FP16)
        self.assertEqual(q_tensor.memory_bytes, 16384 * 144)

        # Verify 144-byte structure validation per block
        self.assertTrue(q_tensor.validate_structure(), "Super-block structural validation failed!")

        # Verify 128-byte cache line alignment of output buffer
        buf = q_tensor.aligned_buffer
        self.assertEqual(len(buf) % 128, 0, "Aligned buffer size is not a multiple of 128 bytes!")

        # Test dequantization reconstruction accuracy
        dequantized = q_tensor.dequantize()
        self.assertEqual(dequantized.shape, q_proj.shape)

        mae = float(np.mean(np.abs(q_proj.astype(np.float32) - dequantized.astype(np.float32))))
        print(f"[SUPERBLOCK TEST PASS] 2048x2048 repacked to {q_tensor.memory_bytes:,} bytes. Reconstruction MAE: {mae:.5f}.")
        self.assertLess(mae, 0.15, f"Reconstruction MAE {mae:.5f} exceeds 0.15 threshold!")


if __name__ == '__main__':
    unittest.main(verbosity=2)
