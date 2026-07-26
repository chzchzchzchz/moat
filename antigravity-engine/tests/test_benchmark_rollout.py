"""
Project Antigravity — Standalone Benchmark Harness for BatchedRolloutCoordinator

Empirically benchmarks BatchedRolloutCoordinator performance, throughput scaling,
and per-token latency across batch sizes N ∈ [1, 2, 4, 8, 16] for 50+ generation steps.

Verifies:
  1. N=8 batched rollout coordinator completes 50 generation steps in <= 1.0s (target: ~0.25s).
  2. Per-token latency (ms/tok) and total throughput (tok/s) scaling for GEMM (N=8) vs GEMV (N=1).
  3. Batch scaling profile for N ∈ [1, 2, 4, 8, 16].
"""

import sys
import os
import time
import unittest
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from batch_generator import BatchedRolloutCoordinator

try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False


class TestBatchedRolloutBenchmark(unittest.TestCase):
    """
    Benchmark suite for BatchedRolloutCoordinator measuring execution time,
    throughput scaling, and per-token latency across N ∈ [1, 2, 4, 8, 16].
    """

    def setUp(self):
        self.hidden_dim = 2048
        self.vocab_size = 32000
        self.num_steps = 50
        self.prompt = [1, 10, 100, 1000]
        np.random.seed(42)
        self.weights = np.random.randn(self.hidden_dim, self.vocab_size).astype(np.float16)
        self.device = "mps" if HAS_MPS else "cpu"

    def test_n8_completes_50_steps_under_1s(self):
        """
        REQUIREMENT 2: Verify that N=8 batched rollout coordinator
        completes 50 generation steps in <= 1.0s (target: ~0.25s).
        """
        coordinator = BatchedRolloutCoordinator(
            n_channels=8,
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size
        )

        # Warm-up run
        _ = coordinator.generate(
            prompt_tokens=self.prompt,
            weights=self.weights,
            max_steps=5,
            temperature=0.7,
            eos_token_id=-1
        )

        # Measured benchmark run
        start_time = time.perf_counter()
        results = coordinator.generate(
            prompt_tokens=self.prompt,
            weights=self.weights,
            max_steps=self.num_steps,
            temperature=0.7,
            eos_token_id=-1
        )
        elapsed_s = time.perf_counter() - start_time

        total_tokens = 8 * self.num_steps
        tok_s = total_tokens / elapsed_s
        step_latency_ms = (elapsed_s / self.num_steps) * 1000.0
        per_tok_latency_ms = (elapsed_s / total_tokens) * 1000.0

        print(f"\n{'='*70}")
        print(f"  BENCHMARK REQUIREMENT 2: N=8 50-Step Latency Check ({self.device.upper()})")
        print(f"{'='*70}")
        print(f"  Total Wall Time:      {elapsed_s:.4f} s (Budget: <= 1.000s, Target: ~0.250s)")
        print(f"  Total Steps:          {results['total_steps']}")
        print(f"  Avg Step Latency:     {step_latency_ms:.3f} ms/step")
        print(f"  Per-Token Latency:    {per_tok_latency_ms:.3f} ms/token")
        print(f"  Total Throughput:     {tok_s:.2f} tok/s")
        print(f"{'='*70}")

        self.assertLessEqual(
            elapsed_s, 1.0,
            f"N=8 rollout coordinator 50 steps took {elapsed_s:.3f}s, exceeding 1.0s budget!"
        )

    def test_benchmark_scaling_n1_to_n16(self):
        """
        REQUIREMENTS 1 & 3:
        Benchmark harness across N ∈ [1, 2, 4, 8, 16] for 50 generation steps.
        Measures per-token latency and total throughput (tok/s) scaling for GEMM (N=8) vs GEMV (N=1).
        """
        channels = [1, 2, 4, 8, 16]
        results_summary = []

        print(f"\n{'='*75}")
        print(f"  THROUGHPUT & LATENCY BENCHMARK: N ∈ [1, 2, 4, 8, 16] ({self.device.upper()})")
        print(f"{'='*75}")
        print(f"  {'N':>3} | {'Wall (s)':>9} | {'Step (ms)':>10} | {'Per-Tok (ms)':>12} | {'Tok/s':>10} | {'Throughput x':>12}")
        print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*12}")

        baseline_tok_s = None

        for N in channels:
            coordinator = BatchedRolloutCoordinator(
                n_channels=N,
                hidden_dim=self.hidden_dim,
                vocab_size=self.vocab_size
            )

            # Warm-up
            _ = coordinator.generate(
                prompt_tokens=self.prompt,
                weights=self.weights,
                max_steps=5,
                temperature=0.7,
                eos_token_id=-1
            )

            # Benchmark 50 steps
            t0 = time.perf_counter()
            res = coordinator.generate(
                prompt_tokens=self.prompt,
                weights=self.weights,
                max_steps=self.num_steps,
                temperature=0.7,
                eos_token_id=-1
            )
            wall_time = time.perf_counter() - t0

            total_toks = N * self.num_steps
            step_ms = (wall_time / self.num_steps) * 1000.0
            per_tok_ms = (wall_time / total_toks) * 1000.0
            tok_s = total_toks / wall_time

            if N == 1:
                baseline_tok_s = tok_s
                tp_scaling = 1.0
            else:
                tp_scaling = tok_s / baseline_tok_s if baseline_tok_s else 1.0

            results_summary.append({
                'N': N,
                'wall_s': wall_time,
                'step_ms': step_ms,
                'per_tok_ms': per_tok_ms,
                'tok_s': tok_s,
                'tp_scaling': tp_scaling
            })

            print(f"  {N:>3} | {wall_time:>9.4f} | {step_ms:>10.3f} | {per_tok_ms:>12.3f} | {tok_s:>10.1f} | {tp_scaling:>11.2f}x")

        print(f"{'='*75}")

        # Extract N=1 (GEMV) vs N=8 (GEMM) metrics
        n1 = next(r for r in results_summary if r['N'] == 1)
        n8 = next(r for r in results_summary if r['N'] == 8)

        gemv_per_tok = n1['per_tok_ms']
        gemm_per_tok = n8['per_tok_ms']
        per_tok_speedup = gemv_per_tok / gemm_per_tok if gemm_per_tok > 0 else 0

        print(f"\n  [GEMV (N=1) vs GEMM (N=8) COMPARISON]")
        print(f"  GEMV (N=1)  Per-Token Latency: {gemv_per_tok:.3f} ms/tok, Throughput: {n1['tok_s']:.1f} tok/s")
        print(f"  GEMM (N=8)  Per-Token Latency: {gemm_per_tok:.3f} ms/tok, Throughput: {n8['tok_s']:.1f} tok/s")
        print(f"  Throughput Speedup Ratio (N=8 vs N=1): {n8['tp_scaling']:.2f}x")
        print(f"  Per-Token Latency Efficiency Ratio:   {per_tok_speedup:.2f}x")
        print(f"{'='*75}\n")


if __name__ == '__main__':
    print("=" * 75)
    print("Standalone Rollout Coordinator Benchmark Harness")
    print("=" * 75)
    unittest.main(verbosity=2)
