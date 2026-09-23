"""
Byte-layout and indexing parity between src/dequant.py and the Metal SuperBlock.

The INT4 kernels in src/shaders/batched_gemm.metal read weights through:

    struct SuperBlock { half scales[8]; uchar packed_nibbles[128]; };   // 144 bytes

and index them with shift/mask arithmetic:

    sb_idx = flat >> 8        // flat / 256 elements per super-block
    in_sb  = flat & 255       // flat % 256
    byte   = packed_nibbles[in_sb >> 1]
    nibble = (in_sb & 1) ? high(byte) : low(byte)      // values biased by +8
    scale  = scales[in_sb >> 5]                        // one scale per 32 elements

Those shifts must agree with repack_to_superblocks() in src/dequant.py, which is
what actually produces the bytes. A mismatch would not crash: the GPU would read
real memory at the wrong offsets and produce plausible-looking garbage weights.
Nothing else in the repository checks this, and the kernels cannot be run here,
so this test reproduces the kernel's arithmetic in Python over the real packed
bytes and compares against the reference dequantizer.
"""

import numpy as np
import pytest

from dequant import quantize_weights_int4, repack_to_superblocks, lut_dequantize

GROUP_SIZE = 32
ELEMENTS_PER_SUPERBLOCK = 256
SUPERBLOCK_BYTES = 144


def serialize(superblocks) -> bytes:
    """Lay super-blocks out exactly as the Metal struct expects them."""
    buf = bytearray()
    for sb in superblocks:
        scales = np.asarray(sb["scales"], dtype=np.float16)
        assert scales.size == 8
        buf += scales.tobytes()                                  # 8 * 2 = 16 bytes
        packed = np.asarray(sb["packed_nibbles"], dtype=np.uint8)
        assert packed.size == 128
        buf += packed.tobytes()                                  # 128 bytes
    return bytes(buf)


def metal_decode(raw: bytes, flat_index: int):
    """Reproduce gemv_int4_kernel's read of one weight, shift for shift.

    Returns (nibble, scale_index, value). The first two are what "indexing is
    correct" actually means; the value carries float rounding and is compared
    only within FP16 tolerance.
    """
    sb_idx = flat_index >> 8
    in_sb = flat_index & 255
    base = sb_idx * SUPERBLOCK_BYTES

    scales = np.frombuffer(raw, dtype=np.float16, count=8, offset=base)
    packed = raw[base + 16 + (in_sb >> 1)]
    nib = ((packed >> 4) & 0x0F) - 8 if (in_sb & 1) else (packed & 0x0F) - 8
    scale_idx = in_sb >> 5
    # The kernel accumulates in float, so it dequantizes in float too.
    return nib, scale_idx, float(nib) * float(scales[scale_idx])


def metal_read(raw: bytes, flat_index: int) -> float:
    return metal_decode(raw, flat_index)[2]


def test_superblock_is_144_bytes():
    w = np.random.randn(ELEMENTS_PER_SUPERBLOCK).astype(np.float32)
    q, s = quantize_weights_int4(w, group_size=GROUP_SIZE)
    raw = serialize(repack_to_superblocks(q, s, group_size=GROUP_SIZE))
    # 16-byte scale header + 128-byte payload, no padding. If the Metal struct ever
    # gains padding, every index past the first block silently shifts.
    assert len(raw) == SUPERBLOCK_BYTES


@pytest.mark.parametrize("n_superblocks", [1, 2, 8])
def test_metal_indexing_matches_reference_dequantizer(n_superblocks):
    rng = np.random.default_rng(0)
    n = ELEMENTS_PER_SUPERBLOCK * n_superblocks
    w = rng.standard_normal(n).astype(np.float32)

    q, s = quantize_weights_int4(w, group_size=GROUP_SIZE)
    raw = serialize(repack_to_superblocks(q, s, group_size=GROUP_SIZE))
    reference = lut_dequantize(q, s, group_size=GROUP_SIZE)

    # Exact: the kernel must select the same quantized value and the same scale
    # for every element. This is the indexing property; any shift error shows here.
    for i in range(n):
        nib, scale_idx, _ = metal_decode(raw, i)
        assert nib == int(q[i]), f"element {i}: kernel read nibble {nib}, packed {int(q[i])}"
        assert scale_idx == (i % ELEMENTS_PER_SUPERBLOCK) // GROUP_SIZE

    # Approximate: dequant.py builds its LUT in float16 while the kernel multiplies
    # in float, so the products differ by FP16 rounding. That is the kernel being
    # more accurate, not a layout mismatch.
    got = np.array([metal_read(raw, i) for i in range(n)], dtype=np.float32)
    np.testing.assert_allclose(got, np.asarray(reference, dtype=np.float32),
                               rtol=1e-2, atol=1e-3,
                               err_msg="Metal indexing disagrees with dequant.py beyond FP16 rounding")


def test_nibble_order_is_low_then_high():
    """Even elements live in the low nibble. Swapping these transposes pairs."""
    q = np.array([-8, 7] + [0] * 254, dtype=np.int8)
    s = np.ones(8, dtype=np.float16)
    raw = serialize(repack_to_superblocks(q, s, group_size=GROUP_SIZE))
    assert metal_read(raw, 0) == -8.0
    assert metal_read(raw, 1) == 7.0


def test_scale_changes_every_32_elements():
    """scales[in_sb >> 5] must select a new scale at each 32-element boundary."""
    q = np.ones(ELEMENTS_PER_SUPERBLOCK, dtype=np.int8)
    s = np.arange(1, 9, dtype=np.float16)
    raw = serialize(repack_to_superblocks(q, s, group_size=GROUP_SIZE))
    for group in range(8):
        assert metal_read(raw, group * GROUP_SIZE) == pytest.approx(float(s[group]))


def test_gemv_indexing_over_a_matrix():
    """flat = k * N + col, as gemv_int4_kernel computes it for B[K x N]."""
    K, N = 16, 16  # 256 elements = exactly one super-block
    rng = np.random.default_rng(7)
    B = rng.standard_normal(K * N).astype(np.float32)

    q, s = quantize_weights_int4(B, group_size=GROUP_SIZE)
    raw = serialize(repack_to_superblocks(q, s, group_size=GROUP_SIZE))
    B_deq = np.asarray(lut_dequantize(q, s, group_size=GROUP_SIZE), dtype=np.float32).reshape(K, N)

    x = rng.standard_normal(K).astype(np.float32)
    for col in range(N):
        kernel_sum = sum(x[k] * metal_read(raw, k * N + col) for k in range(K))
        # Tolerance covers FP16 scale rounding accumulated over K terms.
        assert kernel_sum == pytest.approx(float(x @ B_deq[:, col]), rel=1e-2, abs=1e-2)
