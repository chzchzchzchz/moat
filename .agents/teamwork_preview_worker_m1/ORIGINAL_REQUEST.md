## 2026-07-25T01:17:32Z
You are Worker 1 for Milestone 1 of Project Antigravity.
Working directory: /Users/MohssineChazi2/moat/.agents/teamwork_preview_worker_m1

MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your objective is to implement Milestone 1: Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine.

Tasks to execute:
1. Read the Explorer handoff reports at:
   - /Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_1/handoff.md
   - /Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_2/handoff.md
   - /Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_3/handoff.md
   and /Users/MohssineChazi2/moat/.agents/sub_orch_m1/SCOPE.md.

2. Create the following files in /Users/MohssineChazi2/moat/:
   a. config/engine_config.yaml specifying:
      quantization:
        bits: 4
        group_size: 32
        superblock_size: 256
        alignment_bytes: 128
      execution:
        batch_size: 8
        reflection_threshold: 0.75
        port: 8080

   b. src/__init__.py exporting quantization and dequantization functions.

   c. src/dequant.py implementing:
      - `repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray`: Coalesces 8 fine-grained INT4 quantization groups of size 32 into a single aligned superblock of size 256 aligned to 128-byte boundaries for GPU/NPU SIMD registers. Ensure 128-byte memory alignment (`ctypes.data % 128 == 0`) and contiguous layout.
      - `lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray`: Fast table-lookup dequantization mapping INT4 back to FP16 via vector gather operations over precomputed lookup table. Ensure proper index offset mapping (+8 for signed int4 range [-8, 7]) to avoid NumPy negative index wrap-around bugs.
      - `quantize_fp16_to_int4`: Quantizes FP16 weights to INT4 signed values in range [-8, 7] with fine-grained group scales (group_size=32).
      - `calculate_memory_footprint`: Calculates exact memory footprint for model weights (1.5B-3B params) under INT4 quantization, verifying fit within ~4.5GB RAM ceiling.
      - Additional helper utilities if required.

   d. tests/test_quantization.py implementing unit test suite verifying:
      - INT4 repacking accuracy, shape, and 128-byte alignment.
      - LUT dequantization correctness (FP16 output, values match LUT) and speed.
      - Memory bounds verification (~4.5GB RAM limit for 1.5B and 3.0B models).
      - Engine config loading.

3. Run `pytest tests/test_quantization.py` and verify all tests pass cleanly. Document test commands and results in your handoff report at /Users/MohssineChazi2/moat/.agents/teamwork_preview_worker_m1/handoff.md.

4. Send me a completion message via send_message when finished.
