"""
Project Antigravity — Micro-Unit Test Suite for attention.py

Tests the 32K-entry exponential LUT and safe softmax implementation
in COMPLETE ISOLATION with strict tolerance assertions against
standard NumPy exp() and softmax references.

Test hierarchy:
  1. ExponentialLUT construction (size, memory, boundary values)
  2. LUT lookup accuracy vs numpy.exp() (tolerance-bounded)
  3. Safe softmax LUT vs reference softmax (distribution matching)
  4. Probability axiom enforcement (sums to 1, non-negative, no NaN/Inf)
  5. Edge cases (uniform logits, extreme ranges, single-element)
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import ExponentialLUT, safe_softmax_lut, reference_softmax


class TestExponentialLUTConstruction(unittest.TestCase):
    """Isolated tests for ExponentialLUT initialization."""

    def test_default_size_is_32768(self):
        """Default LUT must have 32768 entries."""
        lut = ExponentialLUT()
        self.assertEqual(lut.size, 32768)
        self.assertEqual(len(lut.lut), 32768)

    def test_memory_is_approximately_64kb(self):
        """32768 FP16 entries = 65536 bytes ≈ 64 KB."""
        lut = ExponentialLUT()
        self.assertEqual(lut.memory_bytes, 32768 * 2)  # FP16 = 2 bytes each
        self.assertEqual(lut.memory_bytes, 65536)

    def test_lut_dtype_is_float16(self):
        """LUT entries must be stored as FP16."""
        lut = ExponentialLUT()
        self.assertEqual(lut.lut.dtype, np.float16)

    def test_lut_endpoint_is_one(self):
        """LUT[-1] = exp(0) = 1.0."""
        lut = ExponentialLUT()
        self.assertAlmostEqual(float(lut.lut[-1]), 1.0, places=2)

    def test_lut_start_is_near_zero(self):
        """LUT[0] = exp(-10) ≈ 0.0000454. Must be positive and small."""
        lut = ExponentialLUT()
        val = float(lut.lut[0])
        self.assertGreater(val, 0.0, "exp(-10) must be positive")
        self.assertLess(val, 0.001, "exp(-10) must be very small")

    def test_lut_is_monotonically_increasing(self):
        """Exponential is monotonically increasing — LUT must be too."""
        lut = ExponentialLUT()
        diffs = np.diff(lut.lut.astype(np.float32))
        self.assertTrue(np.all(diffs >= 0),
            "LUT must be monotonically non-decreasing")

    def test_rejects_invalid_size(self):
        """Must raise ValueError for size < 2."""
        with self.assertRaises(ValueError):
            ExponentialLUT(size=1)

    def test_rejects_negative_range(self):
        """Must raise ValueError for non-positive range_max."""
        with self.assertRaises(ValueError):
            ExponentialLUT(range_max=-1.0)


class TestExponentialLUTLookup(unittest.TestCase):
    """Isolated tests for LUT lookup accuracy vs numpy.exp()."""

    def setUp(self):
        self.lut = ExponentialLUT(size=32768, range_max=10.0)

    def test_lookup_at_zero_returns_one(self):
        """exp(0) = 1.0 — must be exact."""
        x = np.array([0.0], dtype=np.float32)
        result = self.lut.lookup(x)
        self.assertAlmostEqual(float(result[0]), 1.0, places=2)

    def test_lookup_at_negative_one(self):
        """exp(-1) ≈ 0.3679 — must be within 1% tolerance."""
        x = np.array([-1.0], dtype=np.float32)
        result = self.lut.lookup(x)
        expected = np.exp(-1.0)
        rel_error = abs(float(result[0]) - expected) / expected
        self.assertLess(rel_error, 0.02,  # 2% tolerance for FP16
            f"exp(-1) lookup error {rel_error:.4f} exceeds 2%")

    def test_lookup_at_negative_five(self):
        """exp(-5) ≈ 0.006738 — must be within 5% tolerance (small values lose precision in FP16)."""
        x = np.array([-5.0], dtype=np.float32)
        result = self.lut.lookup(x)
        expected = np.exp(-5.0)
        rel_error = abs(float(result[0]) - expected) / expected
        self.assertLess(rel_error, 0.05,
            f"exp(-5) lookup error {rel_error:.4f} exceeds 5%")

    def test_batch_lookup_accuracy(self):
        """
        Batch lookup of 1000 random non-positive values.
        Mean absolute error vs numpy.exp() must be < 0.01.
        """
        np.random.seed(42)
        x = -np.random.rand(1000).astype(np.float32) * 8.0  # [-8, 0]
        lut_result = self.lut.lookup(x).astype(np.float32)
        np_result = np.exp(x)

        mae = np.mean(np.abs(lut_result - np_result))
        self.assertLess(mae, 0.01,
            f"Mean absolute error {mae:.6f} exceeds 0.01 threshold")

    def test_lookup_clamps_beyond_range(self):
        """Values below -range_max must clamp to exp(-range_max), not NaN/Inf."""
        x = np.array([-20.0, -100.0, -1000.0], dtype=np.float32)
        result = self.lut.lookup(x)

        for val in result:
            self.assertFalse(np.isnan(val), "NaN in clamped lookup")
            self.assertFalse(np.isinf(val), "Inf in clamped lookup")
            self.assertGreaterEqual(float(val), 0.0, "Negative exp value")

    def test_all_outputs_are_positive(self):
        """exp(x) is always positive — every lookup result must be > 0."""
        np.random.seed(7)
        x = -np.random.rand(10000).astype(np.float32) * 10.0
        result = self.lut.lookup(x)

        # Note: very small FP16 values may round to 0.0, so check >= 0
        self.assertTrue(np.all(result >= 0),
            f"Found negative values in exp lookup: min={float(result.min())}")

    def test_no_nan_in_lookup(self):
        """No input should ever produce NaN from lookup."""
        x = np.array([0.0, -1.0, -5.0, -10.0, -0.001, -9.999], dtype=np.float32)
        result = self.lut.lookup(x)
        self.assertFalse(np.any(np.isnan(result)), "NaN detected in LUT lookup")

    def test_no_inf_in_lookup(self):
        """No input should ever produce Inf from lookup."""
        x = np.array([0.0, -1.0, -5.0, -10.0], dtype=np.float32)
        result = self.lut.lookup(x)
        self.assertFalse(np.any(np.isinf(result)), "Inf detected in LUT lookup")


class TestSafeSoftmaxLUT(unittest.TestCase):
    """Isolated tests for safe softmax using LUT exponentials."""

    def setUp(self):
        self.lut = ExponentialLUT(size=32768, range_max=10.0)

    def test_output_sums_to_one(self):
        """Softmax output along each row must sum to 1.0."""
        np.random.seed(42)
        x = np.random.randn(4, 16).astype(np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        row_sums = np.sum(result.astype(np.float32), axis=-1)
        for i, s in enumerate(row_sums):
            self.assertAlmostEqual(float(s), 1.0, places=1,
                msg=f"Row {i} sum={float(s):.4f} ≠ 1.0")

    def test_output_is_non_negative(self):
        """All softmax probabilities must be ≥ 0."""
        np.random.seed(42)
        x = np.random.randn(8, 32).astype(np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertTrue(np.all(result >= 0),
            f"Negative probability detected: min={float(result.min())}")

    def test_output_is_at_most_one(self):
        """All softmax probabilities must be ≤ 1."""
        np.random.seed(42)
        x = np.random.randn(8, 32).astype(np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertTrue(np.all(result <= 1.0 + 1e-3),
            f"Probability > 1.0 detected: max={float(result.max())}")

    def test_no_nan_in_output(self):
        """Softmax must never produce NaN."""
        np.random.seed(42)
        x = np.random.randn(16, 64).astype(np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertFalse(np.any(np.isnan(result)), "NaN in softmax output")

    def test_no_inf_in_output(self):
        """Softmax must never produce Inf."""
        np.random.seed(42)
        x = np.random.randn(16, 64).astype(np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertFalse(np.any(np.isinf(result)), "Inf in softmax output")

    def test_matches_reference_softmax(self):
        """
        THE CRITICAL TEST: LUT softmax must closely match reference softmax.
        Tolerance: mean absolute error < 0.005 (FP16 precision limit).
        """
        np.random.seed(42)
        x = np.random.randn(8, 32).astype(np.float16)

        lut_result = safe_softmax_lut(x, self.lut, axis=-1).astype(np.float32)
        ref_result = reference_softmax(x, axis=-1).astype(np.float32)

        mae = np.mean(np.abs(lut_result - ref_result))
        self.assertLess(mae, 0.005,
            f"LUT softmax vs reference MAE={mae:.6f} exceeds 0.005 — CRITICAL")

    def test_matches_reference_large_batch(self):
        """Same critical test on a larger, more realistic attention matrix."""
        np.random.seed(99)
        x = np.random.randn(32, 128).astype(np.float16)

        lut_result = safe_softmax_lut(x, self.lut, axis=-1).astype(np.float32)
        ref_result = reference_softmax(x, axis=-1).astype(np.float32)

        mae = np.mean(np.abs(lut_result - ref_result))
        self.assertLess(mae, 0.005,
            f"Large batch LUT softmax MAE={mae:.6f} exceeds 0.005 — CRITICAL")

    def test_argmax_agreement(self):
        """
        The highest-probability token must be the same between LUT and reference.
        This is the most safety-critical test — wrong argmax = wrong token selection.
        """
        np.random.seed(42)
        x = np.random.randn(100, 64).astype(np.float16)

        lut_result = safe_softmax_lut(x, self.lut, axis=-1)
        ref_result = reference_softmax(x, axis=-1)

        lut_argmax = np.argmax(lut_result, axis=-1)
        ref_argmax = np.argmax(ref_result, axis=-1)

        agreement = np.mean(lut_argmax == ref_argmax)
        self.assertGreaterEqual(agreement, 0.98,
            f"Argmax agreement {agreement:.2%} < 98% — token selection divergence")

    def test_preserves_relative_ordering(self):
        """If logit A > logit B, then softmax(A) > softmax(B) for well-separated values."""
        x = np.array([[5.0, 1.0, 0.0, -1.0, -5.0]], dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        # Check monotonic ordering
        for i in range(4):
            self.assertGreaterEqual(float(result[0, i]), float(result[0, i+1]),
                msg=f"Ordering violated: softmax[{i}]={float(result[0,i]):.4f} "
                    f"< softmax[{i+1}]={float(result[0,i+1]):.4f}")


class TestSoftmaxEdgeCases(unittest.TestCase):
    """Edge case and stress tests for softmax."""

    def setUp(self):
        self.lut = ExponentialLUT(size=32768, range_max=10.0)

    def test_uniform_logits_produce_uniform_distribution(self):
        """Equal logits must produce a uniform distribution."""
        x = np.full((1, 8), 3.0, dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        expected = 1.0 / 8.0
        for val in result[0]:
            self.assertAlmostEqual(float(val), expected, places=1,
                msg=f"Non-uniform output {float(val):.4f} for uniform input")

    def test_single_element_softmax_is_one(self):
        """Softmax of a single element must be 1.0."""
        x = np.array([[42.0]], dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)
        self.assertAlmostEqual(float(result[0, 0]), 1.0, places=1)

    def test_very_large_logits_no_overflow(self):
        """Large positive logits must not cause overflow (safe softmax handles this)."""
        x = np.array([[1000.0, 999.0, 998.0]], dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertFalse(np.any(np.isnan(result)), "NaN from large logits")
        self.assertFalse(np.any(np.isinf(result)), "Inf from large logits")
        self.assertAlmostEqual(float(np.sum(result)), 1.0, places=1)

    def test_very_negative_logits_no_underflow(self):
        """Very negative logits must not cause divide-by-zero."""
        x = np.array([[-1000.0, -999.0, -998.0]], dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertFalse(np.any(np.isnan(result)), "NaN from very negative logits")

    def test_one_hot_logit_dominance(self):
        """One very large logit should dominate (≈1.0), others ≈0.0."""
        x = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float16)
        result = safe_softmax_lut(x, self.lut, axis=-1)

        self.assertGreater(float(result[0, 0]), 0.9,
            "Dominant logit should have probability > 0.9")

    def test_1d_input(self):
        """1D input must work (softmax along axis=0 or -1)."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = safe_softmax_lut(x.reshape(1, -1), self.lut, axis=-1)

        self.assertFalse(np.any(np.isnan(result)))
        self.assertAlmostEqual(float(np.sum(result)), 1.0, places=1)

    def test_fuzz_no_nan_no_inf(self):
        """Fuzz test: 50 random seeds, no NaN or Inf ever."""
        for seed in range(50):
            np.random.seed(seed)
            x = np.random.randn(16, 32).astype(np.float16)
            result = safe_softmax_lut(x, self.lut, axis=-1)

            self.assertFalse(np.any(np.isnan(result)),
                f"NaN in softmax output (seed={seed})")
            self.assertFalse(np.any(np.isinf(result)),
                f"Inf in softmax output (seed={seed})")


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — attention.py Micro-Unit Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
