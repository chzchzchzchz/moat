"""
Project Antigravity — INT4 Quantization, Super-Block Repacking & LUT Dequantization

This module implements:
  1. INT4 symmetric fine-grained group quantization (group_size=32)
  2. 256-element super-block weight repacking (128-byte aligned)
  3. Fast table-lookup (LUT) FP16 dequantization
  4. 32,768-entry precomputed exponential LUT for safe softmax

Every function is designed to be independently testable with strict tolerance
assertions against PyTorch/NumPy reference implementations.
"""

import numpy as np
from typing import Tuple


# =============================================================================
# 1. INT4 SYMMETRIC GROUP QUANTIZATION
# =============================================================================

def quantize_weights_int4(
    weights_fp16: np.ndarray,
    group_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize FP16 weights to INT4 symmetric precision with fine-grained groups.

    For each group G of `group_size` elements:
      1. Compute alpha = max(|w_i|) for all w_i in G
      2. Compute scale S_G = alpha / 7  (symmetric around 0, range [-8, 7])
      3. Quantize: q_i = clamp(round(w_i / S_G), -8, 7)

    Args:
        weights_fp16: 1D float array of model weights.
        group_size:   Number of elements per quantization group (default: 32).

    Returns:
        q_weights: 1D int8 array of quantized values in [-8, 7].
        scales:    1D float16 array of per-group scale factors.

    Raises:
        ValueError: If weight array length is not a multiple of group_size.
    """
    if len(weights_fp16.shape) != 1:
        raise ValueError(f"Expected 1D array, got shape {weights_fp16.shape}")
    if len(weights_fp16) % group_size != 0:
        raise ValueError(
            f"Weight length {len(weights_fp16)} is not divisible by group_size={group_size}"
        )

    n_groups = len(weights_fp16) // group_size
    weights = weights_fp16.astype(np.float32)  # compute in float32 for precision
    groups = weights.reshape(n_groups, group_size)

    # Per-group max absolute value
    alphas = np.max(np.abs(groups), axis=1)  # shape: (n_groups,)

    # Scale factors: alpha / 7 (symmetric quantization)
    # Guard against zero-alpha groups (all-zero weights)
    scales = np.where(alphas > 0, alphas / 7.0, np.float32(1.0))

    # Quantize: round(w / scale), clamp to [-8, 7]
    scales_expanded = scales[:, np.newaxis]  # (n_groups, 1)
    q_float = np.round(groups / scales_expanded)
    q_clamped = np.clip(q_float, -8, 7).astype(np.int8)

    # Flatten back
    q_weights = q_clamped.reshape(-1)
    scales_fp16 = scales.astype(np.float16)

    return q_weights, scales_fp16


# =============================================================================
# 2. SUPER-BLOCK WEIGHT REPACKING (256-element, 128-byte aligned)
# =============================================================================

def repack_to_superblocks(
    q_weights: np.ndarray,
    scales: np.ndarray,
    group_size: int = 32,
    groups_per_superblock: int = 8
) -> list:
    """
    Coalesce 8 fine-grained INT4 quantization groups (32 elements each)
    into a single aligned super-block of 256 elements.

    Each super-block contains:
      - header: 8 FP16 scale factors (16 bytes)
      - payload: 256 INT4 values packed as 128 int8 nibble pairs (128 bytes)
    Total: 144 bytes per super-block.

    Args:
        q_weights: 1D int8 array of quantized weights (values in [-8, 7]).
        scales:    1D float16 array of per-group scales.
        group_size: Elements per group (default: 32).
        groups_per_superblock: Groups per super-block (default: 8).

    Returns:
        List of dicts, each with:
          'scales': np.ndarray of shape (8,) float16
          'packed_nibbles': np.ndarray of shape (128,) uint8 (pairs of INT4 nibbles)
          'raw_int4': np.ndarray of shape (256,) int8 (unpacked, for validation)
    """
    elements_per_sb = group_size * groups_per_superblock  # 256
    n_elements = len(q_weights)

    if n_elements % elements_per_sb != 0:
        raise ValueError(
            f"Weight length {n_elements} not divisible by super-block size {elements_per_sb}"
        )

    n_superblocks = n_elements // elements_per_sb
    superblocks = []

    for sb_idx in range(n_superblocks):
        start = sb_idx * elements_per_sb
        end = start + elements_per_sb

        # Extract 256 quantized values
        raw_int4 = q_weights[start:end].copy()

        # Extract 8 corresponding scale factors
        scale_start = sb_idx * groups_per_superblock
        scale_end = scale_start + groups_per_superblock
        sb_scales = scales[scale_start:scale_end].copy()

        # Pack pairs of INT4 values into uint8 nibbles
        # Each uint8 stores two INT4 values: low nibble = even index, high nibble = odd index
        # Shift INT4 from [-8,7] to [0,15] for unsigned packing
        unsigned = (raw_int4.astype(np.int16) + 8).astype(np.uint8)  # [0, 15]
        packed = np.zeros(elements_per_sb // 2, dtype=np.uint8)
        packed = (unsigned[0::2] & 0x0F) | ((unsigned[1::2] & 0x0F) << 4)

        superblocks.append({
            'scales': sb_scales,
            'packed_nibbles': packed,
            'raw_int4': raw_int4,
        })

    return superblocks


def unpack_superblock(superblock: dict, group_size: int = 32) -> np.ndarray:
    """
    Unpack a super-block's nibble-packed uint8 array back to INT4 values.

    Args:
        superblock: Dict with 'packed_nibbles' (128 uint8s).

    Returns:
        1D int8 array of shape (256,) with values in [-8, 7].
    """
    packed = superblock['packed_nibbles']
    n_elements = len(packed) * 2
    unpacked = np.zeros(n_elements, dtype=np.int8)

    # Low nibbles (even indices)
    unpacked[0::2] = (packed & 0x0F).astype(np.int8)
    # High nibbles (odd indices)
    unpacked[1::2] = ((packed >> 4) & 0x0F).astype(np.int8)

    # Shift back from [0,15] to [-8,7]
    unpacked = unpacked.astype(np.int16) - 8
    return unpacked.astype(np.int8)


# =============================================================================
# 3. FAST TABLE-LOOKUP (LUT) FP16 DEQUANTIZATION
# =============================================================================

def build_dequant_lut(scale: np.float16) -> np.ndarray:
    """
    Build a 16-element FP16 lookup table for a single quantization group.

    Maps INT4 values [-8, -7, ..., 6, 7] to their dequantized FP16 values:
      LUT[k + 8] = k * scale    for k in [-8, 7]

    The +8 offset converts signed INT4 to a 0-indexed table lookup.

    Args:
        scale: FP16 per-group scale factor.

    Returns:
        np.ndarray of shape (16,) dtype float16.
    """
    indices = np.arange(-8, 8, dtype=np.float32)
    lut = (indices * float(scale)).astype(np.float16)
    return lut


def lut_dequantize(
    q_weights: np.ndarray,
    scales: np.ndarray,
    group_size: int = 32
) -> np.ndarray:
    """
    Dequantize INT4 weights to FP16 using per-group lookup tables.

    For each group G with scale S_G:
      1. Build 16-element LUT: LUT[k+8] = k * S_G
      2. For each quantized value q_i in [-8,7]:
         dequantized_i = LUT[q_i + 8]

    This replaces expensive bit-masking/arithmetic with a single
    vector table-lookup operation per element.

    Args:
        q_weights: 1D int8 array of quantized values in [-8, 7].
        scales:    1D float16 array of per-group scale factors.
        group_size: Elements per group.

    Returns:
        1D float16 array of dequantized weights.
    """
    n_elements = len(q_weights)
    n_groups = n_elements // group_size
    result = np.zeros(n_elements, dtype=np.float16)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_q = q_weights[start:end]
        lut = build_dequant_lut(scales[g])
        # Table lookup: shift q from [-8,7] to [0,15] index
        indices = (group_q.astype(np.int16) + 8).astype(np.intp)
        result[start:end] = lut[indices]

    return result


def arithmetic_dequantize(
    q_weights: np.ndarray,
    scales: np.ndarray,
    group_size: int = 32
) -> np.ndarray:
    """
    Reference arithmetic dequantization (traditional method).
    Used ONLY for validation against LUT dequantization.

    dequantized_i = q_i * S_G

    Args:
        q_weights: 1D int8 array of quantized values in [-8, 7].
        scales:    1D float16 array of per-group scale factors.
        group_size: Elements per group.

    Returns:
        1D float16 array of dequantized weights.
    """
    n_elements = len(q_weights)
    n_groups = n_elements // group_size
    result = np.zeros(n_elements, dtype=np.float16)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_q = q_weights[start:end].astype(np.float32)
        result[start:end] = (group_q * float(scales[g])).astype(np.float16)

    return result
