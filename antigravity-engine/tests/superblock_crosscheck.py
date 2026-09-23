#!/usr/bin/env python3
"""
Byte-for-byte cross-check: the C++ packer in src/superblock_pack.h against the
reference INT4 quantizer in src/dequant.py.

Two implementations of the same format exist in this repo — the Python one the
quantization tests are written against, and the C++ one the Metal engine actually
calls at weight-load time. Nothing tied them together, so they could drift apart
without any test noticing, and the GPU would then read bytes that no test had
ever checked. This script packs the same weights with both and compares the
resulting super-blocks byte for byte.

Usage:
    python3 tests/superblock_crosscheck.py path/to/test_superblock_pack_binary
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dequant import quantize_weights_int4, repack_to_superblocks  # noqa: E402

N_ELEMENTS = 256 * 64  # 64 super-blocks


def deterministic_weights(n: int) -> np.ndarray:
    """Mirror deterministicWeights() in tests/test_superblock_pack.cpp exactly.

    The same LCG and the same float->half rounding on both sides, so any
    difference in the packed bytes is a difference in the packers, not in the
    inputs they were handed.
    """
    # Plain ints with an explicit 32-bit mask: numpy would emit an overflow
    # warning for what is deliberate modular arithmetic.
    state = 0x13579BDF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        unit = np.float32(state >> 8) / np.float32(1 << 24)
        out[i] = (unit - np.float32(0.5)) * np.float32(0.25)
    return out.astype(np.float16)


def reference_pack(weights_fp16: np.ndarray) -> bytes:
    """Pack with src/dequant.py, laid out as the Metal SuperBlock struct."""
    q, scales = quantize_weights_int4(weights_fp16, group_size=32)
    blocks = repack_to_superblocks(q, scales, group_size=32, groups_per_superblock=8)
    buf = bytearray()
    for sb in blocks:
        buf += np.asarray(sb["scales"], dtype="<f2").tobytes()          # 16 bytes
        buf += np.asarray(sb["packed_nibbles"], dtype=np.uint8).tobytes()  # 128 bytes
    return bytes(buf)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 2
    packer = Path(sys.argv[1]).resolve()
    if not packer.is_file():
        print(f"[FAIL] packer binary not found: {packer}")
        return 2

    weights = deterministic_weights(N_ELEMENTS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        in_path, out_path = tmp / "weights.bin", tmp / "packed.bin"
        in_path.write_bytes(weights.astype("<f2").tobytes())

        result = subprocess.run(
            [str(packer), "--emit", str(in_path), str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[FAIL] C++ packer exited {result.returncode}: {result.stderr.strip()}")
            return 1
        cpp_bytes = out_path.read_bytes()

    py_bytes = reference_pack(weights)

    if len(cpp_bytes) != len(py_bytes):
        print(f"[FAIL] size mismatch: C++ {len(cpp_bytes)} bytes, dequant.py {len(py_bytes)} bytes")
        return 1

    diff = np.frombuffer(cpp_bytes, dtype=np.uint8) != np.frombuffer(py_bytes, dtype=np.uint8)
    n_diff = int(diff.sum())
    if n_diff:
        first = int(np.argmax(diff))
        block, offset = divmod(first, 144)
        where = "scale" if offset < 16 else "nibble"
        print(f"[FAIL] {n_diff} of {len(py_bytes)} bytes differ; first at byte {first} "
              f"(super-block {block}, {where} offset {offset}): "
              f"C++ 0x{cpp_bytes[first]:02X} vs dequant.py 0x{py_bytes[first]:02X}")
        return 1

    print(f"[PASS] src/superblock_pack.h and src/dequant.py produce identical bytes "
          f"for {N_ELEMENTS} weights ({len(py_bytes)} bytes, {len(py_bytes) // 144} super-blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
