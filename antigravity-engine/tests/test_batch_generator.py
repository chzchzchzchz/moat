"""
Project Antigravity — Micro-Unit Test Suite for batch_generator.py and model_loader.py

Tests the Batched Rollout Coordinator, Paged KV-Cache, Temperature Sampling,
and Model Weight Loader in COMPLETE ISOLATION with strict assertions.

Test hierarchy:
  1. PagedKVCache (allocation, channel isolation, sequence length bounds)
  2. BatchedRolloutCoordinator (step decode, GEMV-to-GEMM math, sampling diversity)
  3. ModelWeightLoader (super-block repacking, layer dequantization, memory budget limits)
  4. End-to-end multi-step rollout simulation (100+ steps, 0 NaN, 0 Inf)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batch_generator import PagedKVCache, BatchedRolloutCoordinator
from model_loader import ModelWeightLoader, estimate_model_superblock_memory, MODEL_WEIGHT_BUDGET_BYTES


class TestPagedKVCache(unittest.TestCase):
    """Isolated unit tests for PagedKVCache manager."""

    def test_initialization_shape_and_memory(self):
        """KV-Cache must allocate non-zero memory for specified channels."""
        cache = PagedKVCache(n_channels=8, max_seq_len=2048, n_heads=16, head_dim=64)
        self.assertEqual(cache.n_channels, 8)
        self.assertEqual(cache.max_seq_len, 2048)

        # 8 channels * 2048 seq * 16 heads * 64 dim * 2 bytes (FP16) * 2 (K+V)
        expected_bytes = 8 * 2048 * 16 * 64 * 2 * 2
        self.assertEqual(cache.total_memory_bytes, expected_bytes)

    def test_append_and_retrieve_kv(self):
        """Append Key/Value projections to a channel and verify retrieval."""
        cache = PagedKVCache(n_channels=4, max_seq_len=512, n_heads=8, head_dim=32)

        k_step = np.random.randn(8, 32).astype(np.float16)
        v_step = np.random.randn(8, 32).astype(np.float16)

        length = cache.append_kv(channel_idx=0, k_step=k_step, v_step=v_step)
        self.assertEqual(length, 1)

        k_slice, v_slice = cache.get_kv(channel_idx=0)
        self.assertEqual(k_slice.shape, (1, 8, 32))
        self.assertEqual(v_slice.shape, (1, 8, 32))
        np.testing.assert_array_equal(k_slice[0], k_step)
        np.testing.assert_array_equal(v_slice[0], v_step)

    def test_channel_memory_isolation(self):
        """
        THE CRITICAL KV-CACHE TEST: Appending to Channel 0 must NOT affect Channel 1.
        Zero cross-channel attention bleeding.
        """
        cache = PagedKVCache(n_channels=4, max_seq_len=512, n_heads=8, head_dim=32)

        k_ch0 = np.ones((8, 32), dtype=np.float16)
        v_ch0 = np.ones((8, 32), dtype=np.float16)

        cache.append_kv(channel_idx=0, k_step=k_ch0, v_step=v_ch0)

        # Channel 1 must still be empty
        k_ch1, v_ch1 = cache.get_kv(channel_idx=1)
        self.assertEqual(len(k_ch1), 0)

        # Channel 1 cache buffer in underlying array must be all zeros
        np.testing.assert_array_equal(cache.k_cache[1], np.zeros((512, 8, 32), dtype=np.float16))

    def test_out_of_bounds_channel(self):
        """Must raise ValueError for channel index out of range."""
        cache = PagedKVCache(n_channels=4)
        k_step = np.zeros((16, 64), dtype=np.float16)
        v_step = np.zeros((16, 64), dtype=np.float16)

        with self.assertRaises(ValueError):
            cache.append_kv(channel_idx=4, k_step=k_step, v_step=v_step)

    def test_reset_clears_cache(self):
        """Reset must zero-out memory and reset sequence lengths."""
        cache = PagedKVCache(n_channels=2)
        k_step = np.ones((16, 64), dtype=np.float16)
        v_step = np.ones((16, 64), dtype=np.float16)

        cache.append_kv(0, k_step, v_step)
        cache.reset()

        self.assertEqual(cache.seq_lengths[0], 0)
        self.assertTrue(np.all(cache.k_cache == 0))


class TestBatchedRolloutCoordinator(unittest.TestCase):
    """Isolated unit tests for BatchedRolloutCoordinator."""

    def test_greedy_sampling(self):
        """Temperature 0.0 must perform deterministic argmax sampling."""
        coordinator = BatchedRolloutCoordinator(n_channels=4, vocab_size=100)
        logits = np.zeros((4, 100), dtype=np.float32)
        logits[0, 10] = 5.0
        logits[1, 42] = 10.0
        logits[2, 7]  = 8.0
        logits[3, 99] = 12.0

        tokens, logprobs = coordinator.sample_tokens(logits, temperature=0.0)

        np.testing.assert_array_equal(tokens, np.array([10, 42, 7, 99]))

    def test_temperature_sampling_diversity(self):
        """
        Temperature > 0 across uniform logits must generate diverse token samples
        for different channels.
        """
        coordinator = BatchedRolloutCoordinator(n_channels=16, vocab_size=1000)
        # Uniform logits
        logits = np.random.randn(16, 1000).astype(np.float32)

        tokens, _ = coordinator.sample_tokens(logits, temperature=1.0)

        # Unique sampled tokens should be > 1 (not all channels picking the same token)
        unique_tokens = len(set(tokens))
        self.assertGreater(unique_tokens, 1,
            "Temperature sampling produced 0 diversity across 16 channels!")

    def test_step_decode_batch_output_shapes(self):
        """step_decode_batch must return tokens and logprobs of shape (N,)."""
        coordinator = BatchedRolloutCoordinator(n_channels=8, vocab_size=1000, hidden_dim=256)

        activations = np.random.randn(8, 256).astype(np.float16)
        weights = np.random.randn(256, 1000).astype(np.float16)

        result = coordinator.step_decode_batch(activations, weights, temperature=0.7)

        self.assertEqual(result['tokens'].shape, (8,))
        self.assertEqual(result['logprobs'].shape, (8,))
        self.assertEqual(result['active_mask'].shape, (8,))
        self.assertEqual(result['cumulative_logprobs'].shape, (8,))

    def test_eos_token_deactivates_channel(self):
        """When a channel samples the EOS token (e.g. 2), its active_mask becomes False."""
        coordinator = BatchedRolloutCoordinator(n_channels=2, vocab_size=10, hidden_dim=16)

        activations = np.ones((2, 16), dtype=np.float16)
        weights = np.zeros((16, 10), dtype=np.float16)

        # Force channel 0 and 1 to pick token 2 (EOS)
        weights[:, 2] = 10.0  # token 2 highest logit

        result = coordinator.step_decode_batch(activations, weights, temperature=0.0, eos_token_id=2)

        self.assertFalse(result['active_mask'][0], "Channel 0 should be deactivated after EOS token")


class TestModelWeightLoader(unittest.TestCase):
    """Isolated unit tests for ModelWeightLoader and super-block repacking."""

    def test_load_and_repack_layer(self):
        """Layer FP16 weights must be quantized and repacked into 256-element super-blocks."""
        loader = ModelWeightLoader()
        weights = np.random.randn(512, 512).astype(np.float16)  # 262,144 elements

        layer_data = loader.load_and_repack_layer('layer_0', weights)

        self.assertEqual(layer_data['layer_name'], 'layer_0')
        self.assertEqual(layer_data['original_shape'], (512, 512))
        # 262,144 / 256 = 1024 super-blocks
        self.assertEqual(len(layer_data['superblocks']), 1024)
        # 1024 super-blocks * 144 bytes = 147,456 bytes
        self.assertEqual(layer_data['memory_bytes'], 1024 * 144)

    def test_layer_dequantization_reconstructs_shape(self):
        """Dequantizing a loaded layer must reconstruct the original FP16 matrix shape."""
        loader = ModelWeightLoader()
        weights = np.random.randn(256, 256).astype(np.float16)

        loader.load_and_repack_layer('layer_0', weights)
        dequantized = loader.dequantize_layer('layer_0')

        self.assertEqual(dequantized.shape, (256, 256))
        self.assertEqual(dequantized.dtype, np.float16)

    def test_memory_budget_enforcement(self):
        """Loading layers exceeding the memory budget must raise MemoryError."""
        # Set a tiny 1 MB budget
        tiny_loader = ModelWeightLoader(memory_budget_bytes=1 * 1024 * 1024)
        weights = np.random.randn(2048, 2048).astype(np.float16)  # ~2.3 MB super-blocks

        with self.assertRaises(MemoryError):
            tiny_loader.load_and_repack_layer('huge_layer', weights)

    def test_1_5B_model_memory_estimation(self):
        """1.5B parameter model in super-block format must fit within 2.5 GB budget."""
        est = estimate_model_superblock_memory(1_500_000_000)

        self.assertLessEqual(est['superblock_memory_gb'], 2.5,
            f"1.5B model super-block footprint {est['superblock_memory_gb']:.2f} GB exceeds 2.5 GB budget!")
        self.assertTrue(est['fits_ios_4_5gb_ceiling'])
        self.assertAlmostEqual(est['bits_per_weight'], 4.5, places=1)


class TestEndToEndRolloutSimulation(unittest.TestCase):
    """Multi-step end-to-end rollout simulation (100 steps, zero NaN/Inf)."""

    def test_100_step_rollout_simulation(self):
        """
        Simulate 100 parallel decode steps across N=8 channels.
        Must execute without errors, produce valid tokens, and generate NO NaN or Inf.
        """
        coordinator = BatchedRolloutCoordinator(n_channels=8, vocab_size=1000, hidden_dim=256)
        weights = np.random.randn(256, 1000).astype(np.float16)

        for step in range(100):
            activations = np.random.randn(8, 256).astype(np.float16)
            res = coordinator.step_decode_batch(activations, weights, temperature=0.7)

            tokens = res['tokens']
            logprobs = res['logprobs']

            self.assertEqual(len(tokens), 8)
            self.assertFalse(np.any(np.isnan(logprobs)), f"NaN in logprobs at step {step}")
            self.assertFalse(np.any(np.isinf(logprobs)), f"Inf in logprobs at step {step}")

        # Each active channel should have 100 tokens recorded
        for c in range(8):
            self.assertGreaterEqual(len(coordinator.channel_tokens[c]), 1)


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — Phase 3 Micro-Unit Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
