"""
Project Antigravity — Precomputed Softmax Exponential LUT & Safe Softmax

This module implements:
  1. 32,768-entry precomputed exponential lookup table (64 KB)
  2. Safe softmax with row-wise max subtraction (all inputs ≤ 0)
  3. Vector-gather index lookup for fast exp() replacement
  4. Standard reference softmax for validation

Every function is independently testable against NumPy/PyTorch references.
"""

import numpy as np
from typing import Optional


# =============================================================================
# 1. PRECOMPUTED EXPONENTIAL LUT (32K entries, ~64 KB)
# =============================================================================

class ExponentialLUT:
    """
    Precomputed exponential lookup table for fast softmax computation.

    Maps non-positive FP16 inputs x ∈ [-range_max, 0.0] to exp(x) ∈ (0, 1].
    Uses 32,768 entries occupying ~64 KB of memory (fits in L1 cache / SRAM).

    The "safe softmax" trick guarantees all inputs are non-positive
    by subtracting the row-wise maximum before computing exponentials.
    """

    def __init__(self, size: int = 32768, range_max: float = 10.0):
        """
        Build the precomputed exponential LUT.

        Args:
            size:      Number of table entries (default: 32768).
            range_max: Maximum absolute input magnitude (default: 10.0).
                       Inputs below -range_max are clamped to exp(-range_max) ≈ 0.
        """
        if size < 2:
            raise ValueError(f"LUT size must be >= 2, got {size}")
        if range_max <= 0:
            raise ValueError(f"range_max must be positive, got {range_max}")

        self.size = size
        self.range_max = range_max

        # Precompute: inputs from -range_max to 0.0 (non-positive domain)
        self.inputs = np.linspace(-range_max, 0.0, size, dtype=np.float32)
        self.lut = np.exp(self.inputs).astype(np.float16)

        # Step size for index mapping
        self.step = range_max / (size - 1)

        # Validate: LUT[0] ≈ exp(-range_max), LUT[-1] = exp(0) = 1.0
        assert abs(float(self.lut[-1]) - 1.0) < 0.01, \
            f"LUT endpoint should be ~1.0, got {self.lut[-1]}"

    def lookup(self, x_shifted: np.ndarray) -> np.ndarray:
        """
        Fast vector-gather exponential lookup.

        Given shifted inputs x_shifted ≤ 0 (after safe max subtraction),
        compute exp(x_shifted) via table lookup instead of dynamic exp().

        Index mapping:
            idx = clamp(|x_shifted| / step, 0, size-1)
            result = LUT[size - 1 - idx]   (reversed: 0 maps to LUT end = 1.0)

        Args:
            x_shifted: Array of non-positive floats (after row-max subtraction).

        Returns:
            Array of approximate exp(x_shifted) values in FP16.
        """
        # Map |x| to table indices
        abs_x = np.abs(x_shifted.astype(np.float32))
        indices = np.clip(abs_x / self.step, 0, self.size - 1).astype(np.intp)

        # Lookup: index 0 → exp(0) = 1.0 (largest), index max → exp(-range_max) ≈ 0
        # Our LUT is ordered from exp(-range_max) to exp(0), so reverse index
        result = self.lut[self.size - 1 - indices]

        return result

    @property
    def memory_bytes(self) -> int:
        """Total memory footprint of the LUT in bytes."""
        return self.lut.nbytes


# =============================================================================
# 2. SAFE SOFTMAX WITH LUT
# =============================================================================

def safe_softmax_lut(
    x: np.ndarray,
    exp_lut: ExponentialLUT,
    axis: int = -1
) -> np.ndarray:
    """
    Compute softmax using precomputed exponential LUT.

    Steps:
      1. Compute row-wise maximum m = max(x, axis=axis)
      2. Shift: x_shifted = x - m  (ensures all values ≤ 0)
      3. Lookup exp(x_shifted) via LUT vector-gather
      4. Normalize: softmax = exp_vals / sum(exp_vals)

    Args:
        x:       Input logits array (any shape, softmax along `axis`).
        exp_lut: Precomputed ExponentialLUT instance.
        axis:    Axis along which to compute softmax (default: -1).

    Returns:
        Softmax probabilities (same shape as x), dtype float16.
    """
    x_f32 = x.astype(np.float32)

    # Step 1: Safe offset (row-wise max subtraction)
    row_max = np.max(x_f32, axis=axis, keepdims=True)
    shifted = x_f32 - row_max  # All values ≤ 0

    # Step 2: LUT-based exponential lookup
    exp_vals = exp_lut.lookup(shifted).astype(np.float32)

    # Step 3: Normalize
    exp_sum = np.sum(exp_vals, axis=axis, keepdims=True)
    # Guard against division by zero (shouldn't happen with safe softmax)
    exp_sum = np.where(exp_sum > 0, exp_sum, np.float32(1.0))
    result = exp_vals / exp_sum

    return result.astype(np.float16)


# =============================================================================
# 3. REFERENCE STANDARD SOFTMAX (for validation only)
# =============================================================================

def reference_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Standard numerically-stable softmax using dynamic exp().
    Used ONLY for validation against LUT softmax.

    Args:
        x:    Input logits array.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax probabilities (same shape as x), dtype float16.
    """
    x_f32 = x.astype(np.float32)
    row_max = np.max(x_f32, axis=axis, keepdims=True)
    shifted = x_f32 - row_max
    exp_vals = np.exp(shifted)
    result = exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)
    return result.astype(np.float16)
