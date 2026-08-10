"""
Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine.
"""

import numpy as np
from typing import Tuple, Dict, Union


def ensure_aligned_array(arr: np.ndarray, alignment: int = 128) -> np.ndarray:
    """
    Ensures C-contiguous memory layout and alignment of array data pointer
    to the specified byte boundary (default 128 bytes for SIMD registers).
    """
    arr_contig = np.ascontiguousarray(arr)
    if arr_contig.ctypes.data % alignment == 0:
        return arr_contig
    
    # Allocate aligned buffer and copy content
    buf = np.empty(arr_contig.nbytes + alignment, dtype=np.uint8)
    offset = (alignment - (buf.ctypes.data % alignment)) % alignment
    aligned_view = buf[offset:offset + arr_contig.nbytes].view(dtype=arr_contig.dtype).reshape(arr_contig.shape)
    np.copyto(aligned_view, arr_contig)
    return aligned_view


def quantize_fp16_to_int4(weights_fp16: np.ndarray, group_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes FP16 weights to INT4 signed values in range [-8, 7]
    with fine-grained group scales (group_size=32).

    Args:
        weights_fp16 (np.ndarray): Weight array in FP16 or FP32.
        group_size (int): Size of quantization group (default 32).

    Returns:
        q_weights (np.ndarray): INT8 array of quantized values in [-8, 7] with original shape.
        scales (np.ndarray): FP16 scale factors per group (1D array).
    """
    flat = weights_fp16.astype(np.float32).flatten()
    assert flat.size % group_size == 0, (
        f"Weight array length ({flat.size}) must be a multiple of group_size ({group_size})."
    )
    
    num_groups = flat.size // group_size
    reshaped = flat.reshape(num_groups, group_size)
    
    # Compute scale per group: alpha / 7.0 where alpha = max(|w|)
    max_abs = np.max(np.abs(reshaped), axis=1, keepdims=True)
    scales = np.where(max_abs == 0, 1.0, max_abs / 7.0)
    
    # Quantize to signed int4 range [-8, 7] and clamp
    q_weights = np.clip(np.round(reshaped / scales), -8, 7).astype(np.int8)
    scales_fp16 = scales.flatten().astype(np.float16)
    
    return q_weights.reshape(weights_fp16.shape), scales_fp16


def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Coalesces 8 fine-grained INT4 quantization groups of size 32 into a
    single aligned superblock of size 256 aligned to 128-byte boundaries
    for GPU/NPU SIMD registers.

    Args:
        weights_int4 (np.ndarray): INT4 packed or signed int8 weights.
        group_size (int): Group size (default 32).

    Returns:
        repacked (np.ndarray): Reshaped C-contiguous array with shape (num_superblocks, 8, group_size)
                               aligned to 128-byte boundaries (ctypes.data % 128 == 0).
    """
    flat = weights_int4.flatten()
    assert len(flat) % 256 == 0, "Weight array length must be a multiple of 256."
    num_superblocks = len(flat) // 256
    repacked = flat.reshape(num_superblocks, 8, group_size).astype(np.int8)
    return ensure_aligned_array(repacked, alignment=128)


def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Fast table-lookup dequantization mapping INT4 back to FP16 via vector gather
    operations over precomputed lookup table.
    """
    lut_fp16 = lut.astype(np.float16) if lut.dtype != np.float16 else lut
    max_idx = lut_fp16.shape[-1] - 1

    if q_weights.dtype == np.int8:
        indices = np.clip((q_weights.astype(np.int16) + 8), 0, max_idx).astype(np.intp)
    elif np.issubdtype(q_weights.dtype, np.signedinteger):
        if np.min(q_weights) < 0:
            indices = np.clip(q_weights.astype(np.int32) + 8, 0, max_idx).astype(np.intp)
        else:
            indices = np.clip(q_weights.astype(np.int32), 0, max_idx).astype(np.intp)
    else:
        indices = np.clip(q_weights.astype(np.int32), 0, max_idx).astype(np.intp)

    if lut_fp16.ndim == 1:
        return lut_fp16[indices]
    elif lut_fp16.ndim == 2:
        num_groups = lut_fp16.shape[0]
        flat_q = indices.reshape(num_groups, -1)
        group_idx = np.arange(num_groups)[:, None]
        return lut_fp16[group_idx, flat_q].reshape(q_weights.shape)
    elif lut_fp16.ndim == 3:
        num_superblocks, groups_per_sb, lut_size = lut_fp16.shape
        num_groups = num_superblocks * groups_per_sb
        flat_lut = lut_fp16.reshape(num_groups, lut_size)
        flat_q = indices.reshape(num_groups, -1)
        group_idx = np.arange(num_groups)[:, None]
        return flat_lut[group_idx, flat_q].reshape(q_weights.shape)
    else:
        raise ValueError(f"Unsupported LUT shape {lut.shape}. Must be 1D, 2D, or 3D.")



def unpack_superblock_weights(superblock_bytes: np.ndarray, num_superblocks: int) -> np.ndarray:
    """
    Unpacks raw 144-byte binary superblocks (16-byte FP16 scale header + 128-byte INT4 payload)
    back to FP16 weights array.

    Args:
        superblock_bytes (np.ndarray): Flattened byte array of raw superblock binary payload.
        num_superblocks (int): Number of superblocks in the byte stream.

    Returns:
        dequantized (np.ndarray): FP16 unpacked weights array of size num_superblocks * 256.
    """
    reshaped = superblock_bytes.reshape(num_superblocks, 144)
    scales_raw = reshaped[:, :16].view(np.float16).reshape(num_superblocks, 8)
    payload = reshaped[:, 16:]  # shape (num_superblocks, 128)
    
    # Unpack nibbles: low 4 bits, high 4 bits
    low_nibbles = (payload & 0x0F).astype(np.int8) - 8
    high_nibbles = ((payload >> 4) & 0x0F).astype(np.int8) - 8
    
    # Interleave low and high nibbles
    unpacked_int4 = np.empty((num_superblocks, 128, 2), dtype=np.int8)
    unpacked_int4[:, :, 0] = low_nibbles
    unpacked_int4[:, :, 1] = high_nibbles
    unpacked_int4 = unpacked_int4.reshape(num_superblocks, 8, 32)
    
    k_range = np.arange(-8, 8, dtype=np.float32)
    scales_flat = scales_raw.reshape(-1)
    lut_2d = scales_flat[:, None] * k_range[None, :]
    
    indices = unpacked_int4.reshape(num_superblocks * 8, 32)
    return lut_dequantize_fp16(indices, lut_2d).reshape(num_superblocks * 256)


def calculate_memory_footprint(num_params: int, group_size: int = 32, bits: int = 4) -> Dict[str, Union[int, float, bool]]:
    """
    Calculates exact memory footprint for model weights under fine-grained group quantization,
    verifying fit within the ~4.5GB RAM limit ceiling (e.g. on iPhone 15 Pro).

    Args:
        num_params (int): Total number of model parameters (e.g. 1,500,000,000 or 3,000,000,000).
        group_size (int): Quantization group size (default 32).
        bits (int): Bit precision per weight (default 4).

    Returns:
        dict containing memory metrics:
            num_params, weight_bytes, scale_bytes, total_bytes, total_mb, total_gb,
            ram_limit_gb, fits_in_ram, bytes_per_param_effective.
    """
    bytes_per_param = bits / 8.0  # 0.5 bytes for INT4
    weight_bytes = num_params * bytes_per_param
    num_groups = num_params // group_size
    scale_bytes = num_groups * 2.0  # FP16 scales (2 bytes per scale)
    
    total_bytes = weight_bytes + scale_bytes
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    
    ram_limit_gb = 4.5
    fits_in_ram = total_gb <= ram_limit_gb
    
    return {
        "num_params": num_params,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "total_gb": total_gb,
        "ram_limit_gb": ram_limit_gb,
        "fits_in_ram": fits_in_ram,
        "bytes_per_param_effective": total_bytes / num_params,
    }
