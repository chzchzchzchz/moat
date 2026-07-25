# Scope: Milestone 1 - Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine

## Architecture
- INT4 Fine-Grained Quantization (group size 32)
- Superblock Repacking (8 groups = 256 elements, aligned to 128-byte SIMD register boundaries)
- FP16 Fast LUT Dequantization (`lut_dequantize_fp16`)
- Engine configuration file `config/engine_config.yaml`
- Unit test suite `tests/test_quantization.py`

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine | `config/engine_config.yaml`, `src/__init__.py`, `src/dequant.py`, `tests/test_quantization.py` | None | IN_PROGRESS |

## Interface Contracts & Specifications
1. `config/engine_config.yaml`:
   - quantization: bits: 4, group_size: 32, superblock_size: 256, alignment_bytes: 128
   - execution: batch_size: 8, reflection_threshold: 0.75, port: 8080
2. `src/dequant.py`:
   - `repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray`: Coalesces 8 fine-grained INT4 quantization groups of size 32 into a single aligned superblock of size 256 aligned to 128-byte boundaries for GPU/NPU SIMD registers.
   - `lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray`: Fast table-lookup dequantization mapping INT4 back to FP16 via vector gather operations over precomputed lookup table.
   - Additional quantization packing/unpacking & memory footprint calculations for 1.5B-3B models (~4.5GB RAM footprint limit).
3. `tests/test_quantization.py`:
   - INT4 repacking accuracy and alignment checks.
   - Speed/correctness tests for `lut_dequantize_fp16`.
   - Memory bounds verification test.
