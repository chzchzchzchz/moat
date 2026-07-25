"""
Project Antigravity — GEMV vs GEMM Batched Decode Benchmark (CPU Correctness Proxy)

This benchmark validates that batched GEMM produces mathematically
identical results to sequential GEMV. It runs on CPU as a correctness
proxy — the actual speedup from matrix tile saturation occurs on GPU/ANE.

WHY CPU SHOWS NO SPEEDUP:
  Apple's Accelerate BLAS dynamically optimizes both GEMV and GEMM on CPU.
  There are no "idle matrix tiles" on a general-purpose CPU — the BLAS
  library adapts its strategy per operation. The entire speedup thesis
  targets GPU/ANE simdgroup_matrix hardware tiles that physically idle
  during GEMV but saturate during batched GEMM.

  See test_batched_speed_metal.py for the Metal GPU benchmark where
  the 3-8x speedup actually manifests.

Usage:
    python3 tests/test_batched_speed.py
"""

import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def simulate_gemv(weight_matrix: np.ndarray, activation_vector: np.ndarray) -> np.ndarray:
    """Single-token GEMV decode step. Shape: (1,K) @ (K,M) → (1,M)."""
    if activation_vector.ndim == 1:
        activation_vector = activation_vector.reshape(1, -1)
    return activation_vector @ weight_matrix


def simulate_gemm(weight_matrix: np.ndarray, activation_batch: np.ndarray) -> np.ndarray:
    """Batched GEMM decode step. Shape: (N,K) @ (K,M) → (N,M)."""
    return activation_batch @ weight_matrix


# =========================================================================
# MUST-PASS: Mathematical Correctness Tests
# =========================================================================

class TestBatchedCorrectness(unittest.TestCase):
    """
    These tests MUST pass unconditionally. If batched GEMM doesn't
    produce the same math as sequential GEMV, nothing else matters.
    """

    def test_gemm_output_matches_sequential_gemv(self):
        """Batched GEMM(N=8) output must equal 8 sequential GEMVs."""
        np.random.seed(42)
        K, M, N = 2048, 2048, 8
        W = np.random.randn(K, M).astype(np.float16)
        activations = np.random.randn(N, K).astype(np.float16)

        sequential = np.vstack([simulate_gemv(W, activations[i]) for i in range(N)])
        batched = simulate_gemm(W, activations)

        np.testing.assert_allclose(
            batched.astype(np.float32), sequential.astype(np.float32),
            rtol=1e-2, atol=1e-2,
            err_msg="CRITICAL: Batched GEMM ≠ sequential GEMV — math is broken"
        )

    def test_correctness_across_batch_sizes(self):
        """Verify math correctness for N=1,2,4,8,16."""
        np.random.seed(7)
        K, M = 2048, 2048
        W = np.random.randn(K, M).astype(np.float16)

        for N in [1, 2, 4, 8, 16]:
            activations = np.random.randn(N, K).astype(np.float16)
            sequential = np.vstack([simulate_gemv(W, activations[i]) for i in range(N)])
            batched = simulate_gemm(W, activations)

            np.testing.assert_allclose(
                batched.astype(np.float32), sequential.astype(np.float32),
                rtol=1e-2, atol=1e-2,
                err_msg=f"Math mismatch at N={N}"
            )

    def test_output_shapes_correct(self):
        """Output shape must be (N, M) for all batch sizes."""
        K, M = 2048, 2048
        W = np.random.randn(K, M).astype(np.float16)
        for N in [1, 2, 4, 8, 16]:
            x = np.random.randn(N, K).astype(np.float16)
            self.assertEqual(simulate_gemm(W, x).shape, (N, M))

    def test_no_nan_no_inf(self):
        """No NaN or Inf in any output."""
        np.random.seed(42)
        K, M = 2048, 2048
        W = np.random.randn(K, M).astype(np.float16)
        for N in [1, 4, 8, 16]:
            x = np.random.randn(N, K).astype(np.float16)
            result = simulate_gemm(W, x)
            self.assertFalse(np.any(np.isnan(result)), f"NaN at N={N}")
            self.assertFalse(np.any(np.isinf(result)), f"Inf at N={N}")


# =========================================================================
# INFORMATIONAL: CPU Performance Profiling (no pass/fail assertions)
# =========================================================================

class TestCPUPerformanceProfile(unittest.TestCase):
    """
    CPU performance profiling — results are INFORMATIONAL ONLY.
    CPU BLAS does not exhibit GPU-style matrix tile underutilization,
    so no speedup is expected. These numbers exist to:
      1. Establish a CPU baseline for comparison with Metal results
      2. Verify the benchmark harness works correctly
    """

    def test_profile_per_token_throughput(self):
        """Profile per-token cost at various batch sizes. Informational only."""
        np.random.seed(42)
        K, M = 2048, 2048
        batch_sizes = [1, 2, 4, 8, 16]
        n_iterations = 20

        W = np.random.randn(K, M).astype(np.float16)

        # Warm up
        for _ in range(3):
            _ = np.random.randn(16, K).astype(np.float16) @ W

        print(f"\n{'='*65}")
        print(f"  CPU Performance Profile (Informational — No Speedup Expected)")
        print(f"{'='*65}")
        print(f"  Backend: NumPy + Apple Accelerate BLAS")
        print(f"  Matrix:  ({K}, {M}) FP16")
        print(f"  ")
        print(f"  {'N':>4} | {'Total (ms)':>12} | {'Per-Tok (ms)':>12} | "
              f"{'Tok/s':>10} | {'vs N=1':>8}")
        print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

        per_token_costs = []
        for N in batch_sizes:
            times = []
            for _ in range(n_iterations):
                x = np.random.randn(N, K).astype(np.float16)
                start = time.perf_counter()
                _ = simulate_gemm(W, x)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            median_time = np.median(times)
            per_tok = median_time / N
            tok_s = N / median_time
            per_token_costs.append(per_tok)
            ratio = per_token_costs[0] / per_tok if per_tok > 0 else 0

            print(f"  {N:>4} | {median_time*1000:>12.2f} | {per_tok*1000:>12.2f} | "
                  f"{tok_s:>10.1f} | {ratio:>7.2f}x")

        print(f"  ")
        print(f"  NOTE: ~1.0x ratio is EXPECTED on CPU. The speedup thesis")
        print(f"  targets GPU matrix tiles, not CPU BLAS.")
        print(f"{'='*65}")

        # No assertion — this is purely informational
        # The assertion lives in the Metal GPU benchmark


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — GEMV vs GEMM Correctness Benchmark (CPU)")
    print("=" * 70)
    unittest.main(verbosity=2)
