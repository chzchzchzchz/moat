"""
Project Antigravity — Micro-Unit Test Suite for dequant.py

Every component is tested in COMPLETE ISOLATION with strict tolerance
assertions against NumPy reference computations.

Test hierarchy:
  1. INT4 symmetric group quantization (range, clamping, zero groups)
  2. Super-block repacking (pack/unpack roundtrip, alignment)
  3. LUT dequantization (exact parity with arithmetic dequant)
  4. Quantize → Dequantize roundtrip (reconstruction error bounds)
  5. Edge cases (all-zero weights, extreme values, non-uniform distributions)
"""

import sys
import os
import unittest
import numpy as np

# Add parent src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dequant import (
    quantize_weights_int4,
    repack_to_superblocks,
    unpack_superblock,
    build_dequant_lut,
    lut_dequantize,
    arithmetic_dequantize,
)


class TestINT4Quantization(unittest.TestCase):
    """Isolated tests for INT4 symmetric group quantization."""

    def test_output_range_within_int4_bounds(self):
        """Every quantized value must be in [-8, 7] — no exceptions."""
        np.random.seed(42)
        weights = np.random.randn(1024).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        self.assertTrue(np.all(q >= -8), f"Min value {q.min()} < -8")
        self.assertTrue(np.all(q <= 7), f"Max value {q.max()} > 7")

    def test_output_dtype_is_int8(self):
        """Quantized weights must be stored as int8."""
        weights = np.random.randn(256).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float16)

    def test_output_shape_matches_input(self):
        """Output q_weights must have same length as input weights."""
        weights = np.random.randn(512).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        self.assertEqual(len(q), 512)
        self.assertEqual(len(scales), 512 // 32)  # 16 groups

    def test_scale_is_positive(self):
        """All scale factors must be positive (or 1.0 for zero-groups)."""
        np.random.seed(7)
        weights = np.random.randn(256).astype(np.float16)
        _, scales = quantize_weights_int4(weights, group_size=32)

        self.assertTrue(np.all(scales > 0), f"Found non-positive scale: {scales}")

    def test_zero_weight_group_produces_zero_quantized(self):
        """A group of all-zero weights must quantize to all zeros."""
        weights = np.zeros(32, dtype=np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        np.testing.assert_array_equal(q, np.zeros(32, dtype=np.int8))

    def test_symmetric_quantization_respects_magnitude(self):
        """Larger magnitude weights must map to larger absolute q values."""
        # Create a group where element 0 is largest, element 1 is smallest non-zero
        weights = np.zeros(32, dtype=np.float16)
        weights[0] = 1.0
        weights[1] = 0.1
        q, _ = quantize_weights_int4(weights, group_size=32)

        self.assertGreater(abs(int(q[0])), abs(int(q[1])))

    def test_max_value_maps_to_7(self):
        """The element with max absolute value in a group should map to +7 or -7."""
        weights = np.zeros(32, dtype=np.float16)
        weights[0] = 3.5  # positive max
        q, _ = quantize_weights_int4(weights, group_size=32)
        self.assertEqual(q[0], 7)

    def test_negative_max_maps_to_negative_7_or_8(self):
        """Negative max-magnitude element should map to -7 (symmetric) or -8."""
        weights = np.zeros(32, dtype=np.float16)
        weights[0] = -3.5  # negative max
        q, _ = quantize_weights_int4(weights, group_size=32)
        self.assertEqual(q[0], -7)

    def test_rejects_non_divisible_length(self):
        """Must raise ValueError if weight length isn't divisible by group_size."""
        weights = np.random.randn(33).astype(np.float16)
        with self.assertRaises(ValueError):
            quantize_weights_int4(weights, group_size=32)

    def test_rejects_non_1d_input(self):
        """Must raise ValueError for non-1D inputs."""
        weights = np.random.randn(4, 32).astype(np.float16)
        with self.assertRaises(ValueError):
            quantize_weights_int4(weights, group_size=32)

    def test_large_batch_quantization(self):
        """Quantize a realistic model-layer-sized weight tensor (2048x2048 = 4M elements)."""
        np.random.seed(99)
        weights = np.random.randn(2048 * 32).astype(np.float16)  # 65536 elements
        q, scales = quantize_weights_int4(weights, group_size=32)

        self.assertEqual(len(q), 65536)
        self.assertEqual(len(scales), 65536 // 32)
        self.assertTrue(np.all(q >= -8))
        self.assertTrue(np.all(q <= 7))


class TestSuperBlockRepacking(unittest.TestCase):
    """Isolated tests for 256-element super-block repacking."""

    def test_roundtrip_pack_unpack_identity(self):
        """Pack then unpack must return the exact original INT4 values."""
        np.random.seed(42)
        weights = np.random.randn(256).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        superblocks = repack_to_superblocks(q, scales)
        self.assertEqual(len(superblocks), 1)

        unpacked = unpack_superblock(superblocks[0])
        np.testing.assert_array_equal(
            unpacked, q,
            err_msg="Pack/unpack roundtrip failed — data corruption in nibble packing"
        )

    def test_packed_nibbles_size_is_128_bytes(self):
        """Each super-block's packed payload must be exactly 128 bytes (256 nibbles / 2)."""
        q = np.random.randint(-8, 8, size=256, dtype=np.int8)
        scales = np.ones(8, dtype=np.float16)

        superblocks = repack_to_superblocks(q, scales)
        self.assertEqual(len(superblocks[0]['packed_nibbles']), 128)
        self.assertEqual(superblocks[0]['packed_nibbles'].dtype, np.uint8)

    def test_scales_header_is_8_fp16(self):
        """Each super-block must carry exactly 8 FP16 scale factors."""
        q = np.random.randint(-8, 8, size=256, dtype=np.int8)
        scales = np.random.rand(8).astype(np.float16)

        superblocks = repack_to_superblocks(q, scales)
        self.assertEqual(len(superblocks[0]['scales']), 8)
        self.assertEqual(superblocks[0]['scales'].dtype, np.float16)

    def test_multiple_superblocks(self):
        """512 elements = 2 super-blocks."""
        q = np.random.randint(-8, 8, size=512, dtype=np.int8)
        scales = np.random.rand(16).astype(np.float16)

        superblocks = repack_to_superblocks(q, scales)
        self.assertEqual(len(superblocks), 2)

        # Roundtrip both
        for i, sb in enumerate(superblocks):
            unpacked = unpack_superblock(sb)
            np.testing.assert_array_equal(
                unpacked, q[i*256:(i+1)*256],
                err_msg=f"Super-block {i} roundtrip failed"
            )

    def test_boundary_int4_values_survive_packing(self):
        """Extreme INT4 values (-8 and 7) must survive nibble packing."""
        q = np.zeros(256, dtype=np.int8)
        q[0] = -8   # minimum
        q[1] = 7    # maximum
        q[2] = 0    # zero
        q[3] = -1   # negative
        scales = np.ones(8, dtype=np.float16)

        superblocks = repack_to_superblocks(q, scales)
        unpacked = unpack_superblock(superblocks[0])

        self.assertEqual(unpacked[0], -8)
        self.assertEqual(unpacked[1], 7)
        self.assertEqual(unpacked[2], 0)
        self.assertEqual(unpacked[3], -1)

    def test_rejects_non_256_aligned(self):
        """Must raise ValueError if weight count isn't a multiple of 256."""
        q = np.random.randint(-8, 8, size=200, dtype=np.int8)
        scales = np.ones(7, dtype=np.float16)

        with self.assertRaises(ValueError):
            repack_to_superblocks(q, scales)

    def test_total_superblock_memory_size(self):
        """Verify theoretical memory footprint: 144 bytes per super-block."""
        q = np.random.randint(-8, 8, size=256, dtype=np.int8)
        scales = np.ones(8, dtype=np.float16)

        superblocks = repack_to_superblocks(q, scales)
        sb = superblocks[0]

        header_bytes = sb['scales'].nbytes      # 8 * 2 = 16 bytes
        payload_bytes = sb['packed_nibbles'].nbytes  # 128 bytes
        total = header_bytes + payload_bytes

        self.assertEqual(header_bytes, 16, "Header must be 16 bytes (8 x FP16)")
        self.assertEqual(payload_bytes, 128, "Payload must be 128 bytes (256 nibbles packed)")
        self.assertEqual(total, 144, "Total super-block must be 144 bytes")


class TestLUTDequantization(unittest.TestCase):
    """Isolated tests for LUT-based FP16 dequantization."""

    def test_lut_has_16_entries(self):
        """Each dequant LUT must have exactly 16 entries (for INT4 range)."""
        lut = build_dequant_lut(np.float16(0.5))
        self.assertEqual(len(lut), 16)
        self.assertEqual(lut.dtype, np.float16)

    def test_lut_zero_entry_is_zero(self):
        """LUT[8] corresponds to quantized value 0 → must be 0.0."""
        lut = build_dequant_lut(np.float16(1.0))
        self.assertAlmostEqual(float(lut[8]), 0.0, places=5)

    def test_lut_symmetry(self):
        """LUT[-k] == -LUT[k] for symmetric quantization."""
        scale = np.float16(0.75)
        lut = build_dequant_lut(scale)

        # Index 8+k and 8-k should be negatives of each other
        for k in range(1, 8):
            pos_val = float(lut[8 + k])
            neg_val = float(lut[8 - k])
            self.assertAlmostEqual(pos_val, -neg_val, places=3,
                msg=f"LUT asymmetry at k={k}: +{pos_val} vs -{neg_val}")

    def test_lut_dequant_matches_arithmetic_dequant_exactly(self):
        """
        THE CRITICAL TEST: LUT dequantization must produce IDENTICAL results
        to standard arithmetic dequantization (q * scale) in FP16.
        """
        np.random.seed(42)
        weights = np.random.randn(1024).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        lut_result = lut_dequantize(q, scales, group_size=32)
        arith_result = arithmetic_dequantize(q, scales, group_size=32)

        np.testing.assert_array_equal(
            lut_result, arith_result,
            err_msg="LUT dequant does NOT match arithmetic dequant — CRITICAL FAILURE"
        )

    def test_lut_dequant_matches_arithmetic_large_tensor(self):
        """Same critical test on a larger, more realistic tensor."""
        np.random.seed(99)
        weights = np.random.randn(65536).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        lut_result = lut_dequantize(q, scales, group_size=32)
        arith_result = arithmetic_dequantize(q, scales, group_size=32)

        np.testing.assert_array_equal(
            lut_result, arith_result,
            err_msg="LUT vs arithmetic mismatch on large tensor — CRITICAL FAILURE"
        )

    def test_dequant_of_zero_quantized_is_zero(self):
        """Dequantizing all-zero quantized weights must give all zeros."""
        q = np.zeros(32, dtype=np.int8)
        scales = np.array([1.0], dtype=np.float16)

        result = lut_dequantize(q, scales, group_size=32)
        np.testing.assert_array_equal(result, np.zeros(32, dtype=np.float16))

    def test_dequant_max_value(self):
        """q=7 with scale S must dequantize to 7*S."""
        q = np.full(32, 7, dtype=np.int8)
        scale = np.float16(0.5)
        scales = np.array([scale], dtype=np.float16)

        result = lut_dequantize(q, scales, group_size=32)
        expected = np.float16(7.0 * 0.5)

        for val in result:
            self.assertAlmostEqual(float(val), float(expected), places=3)

    def test_dequant_min_value(self):
        """q=-8 with scale S must dequantize to -8*S."""
        q = np.full(32, -8, dtype=np.int8)
        scale = np.float16(0.5)
        scales = np.array([scale], dtype=np.float16)

        result = lut_dequantize(q, scales, group_size=32)
        expected = np.float16(-8.0 * 0.5)

        for val in result:
            self.assertAlmostEqual(float(val), float(expected), places=3)


class TestQuantizeDequantizeRoundtrip(unittest.TestCase):
    """End-to-end roundtrip: FP16 → INT4 → FP16. Measures reconstruction error."""

    def test_reconstruction_error_within_bounds(self):
        """
        Quantize then dequantize random FP16 weights.
        The maximum per-element error must be bounded by scale/2
        (half a quantization step).
        """
        np.random.seed(42)
        weights = np.random.randn(1024).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)
        reconstructed = lut_dequantize(q, scales, group_size=32)

        # Per-group max error should be <= scale / 2 (rounding error bound)
        n_groups = len(weights) // 32
        for g in range(n_groups):
            s, e = g * 32, (g + 1) * 32
            error = np.abs(weights[s:e].astype(np.float32) - reconstructed[s:e].astype(np.float32))
            max_error = float(np.max(error))
            scale_val = float(scales[g])
            bound = scale_val / 2.0 + 1e-3  # small epsilon for FP16 rounding

            self.assertLessEqual(max_error, bound,
                msg=f"Group {g}: max_error={max_error:.6f} exceeds bound={bound:.6f}")

    def test_reconstruction_preserves_sign(self):
        """Sign of every weight must be preserved through quantize/dequantize roundtrip.

        Weights within half a quantization step of zero legitimately quantize
        to q=0 and lose their sign — this is expected INT4 behavior, not a bug.
        We filter those out using the per-group scale as the dead-zone width.
        """
        np.random.seed(7)
        weights = np.random.randn(256).astype(np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)
        reconstructed = lut_dequantize(q, scales, group_size=32)

        # Build a mask: exclude weights that are within the quantization
        # dead zone (|w| < scale/2) where q rounds to 0
        mask = np.ones(256, dtype=bool)
        for g in range(256 // 32):
            s, e = g * 32, (g + 1) * 32
            dead_zone = float(scales[g]) / 2.0 + 1e-3
            mask[s:e] &= (np.abs(weights[s:e].astype(np.float32)) > dead_zone)

        original_signs = np.sign(weights[mask].astype(np.float32))
        recon_signs = np.sign(reconstructed[mask].astype(np.float32))

        mismatches = np.sum(original_signs != recon_signs)
        self.assertEqual(mismatches, 0,
            msg=f"{mismatches} sign mismatches in roundtrip (after dead-zone filter) — data corruption")

    def test_full_superblock_pipeline_roundtrip(self):
        """
        Full pipeline: FP16 → INT4 → SuperBlock Pack → Unpack → LUT Dequant → FP16
        Must match direct LUT dequantization (no data loss from packing).
        """
        np.random.seed(42)
        weights = np.random.randn(256).astype(np.float16)

        # Step 1: Quantize
        q, scales = quantize_weights_int4(weights, group_size=32)

        # Step 2: Direct dequant (reference)
        direct_result = lut_dequantize(q, scales, group_size=32)

        # Step 3: Pack into super-block
        superblocks = repack_to_superblocks(q, scales)

        # Step 4: Unpack from super-block
        unpacked_q = unpack_superblock(superblocks[0])

        # Step 5: Dequantize unpacked weights
        pipeline_result = lut_dequantize(unpacked_q, superblocks[0]['scales'], group_size=32)

        # Must be IDENTICAL — zero tolerance
        np.testing.assert_array_equal(
            pipeline_result, direct_result,
            err_msg="Full pipeline roundtrip introduced data corruption — CRITICAL"
        )


class TestEdgeCases(unittest.TestCase):
    """Stress tests for boundary conditions and pathological inputs."""

    def test_all_zero_weights(self):
        """Model layer with all-zero weights (rare but must not crash)."""
        weights = np.zeros(256, dtype=np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        np.testing.assert_array_equal(q, np.zeros(256, dtype=np.int8))

        result = lut_dequantize(q, scales, group_size=32)
        np.testing.assert_array_equal(result, np.zeros(256, dtype=np.float16))

    def test_uniform_positive_weights(self):
        """All weights are the same positive value."""
        weights = np.full(256, 1.0, dtype=np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)

        # All should quantize to the same value (7)
        self.assertTrue(np.all(q == q[0]))
        self.assertEqual(q[0], 7)

    def test_alternating_extreme_values(self):
        """Alternating +max and -max values."""
        weights = np.zeros(256, dtype=np.float16)
        weights[0::2] = 10.0
        weights[1::2] = -10.0

        q, scales = quantize_weights_int4(weights, group_size=32)

        self.assertTrue(np.all(q[0::2] == 7))
        self.assertTrue(np.all(q[1::2] == -7))

    def test_very_small_weights(self):
        """Tiny weights near machine epsilon must not produce NaN or Inf."""
        weights = np.full(256, 1e-4, dtype=np.float16)
        q, scales = quantize_weights_int4(weights, group_size=32)
        result = lut_dequantize(q, scales, group_size=32)

        self.assertFalse(np.any(np.isnan(result)), "NaN detected in dequantized output")
        self.assertFalse(np.any(np.isinf(result)), "Inf detected in dequantized output")

    def test_no_nan_in_any_output(self):
        """Fuzz test: random weights must NEVER produce NaN anywhere in pipeline."""
        for seed in range(10):
            np.random.seed(seed)
            weights = np.random.randn(1024).astype(np.float16)
            q, scales = quantize_weights_int4(weights, group_size=32)
            result = lut_dequantize(q, scales, group_size=32)

            self.assertFalse(np.any(np.isnan(q)), f"NaN in q_weights (seed={seed})")
            self.assertFalse(np.any(np.isnan(scales)), f"NaN in scales (seed={seed})")
            self.assertFalse(np.any(np.isnan(result)), f"NaN in dequant result (seed={seed})")

    def test_no_inf_in_any_output(self):
        """Fuzz test: random weights must NEVER produce Inf anywhere in pipeline."""
        for seed in range(10):
            np.random.seed(seed)
            weights = (np.random.randn(1024) * 100).astype(np.float16)
            # float16 max is 65504, so *100 may clip — that's fine, test the pipeline handles it
            q, scales = quantize_weights_int4(weights, group_size=32)
            result = lut_dequantize(q, scales, group_size=32)

            self.assertFalse(np.any(np.isinf(result)), f"Inf in dequant result (seed={seed})")


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — dequant.py Micro-Unit Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
