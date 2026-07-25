# Original User Request

## 2026-07-25T05:15:45Z
You are the Milestone 1 Sub-Orchestrator for Project Antigravity.
Working directory: /Users/MohssineChazi2/moat/.agents/sub_orch_m1
Parent Conversation ID: 04a01613-34ab-46c5-8005-aa56ed9b71fe

Your objective is to complete Milestone 1: Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine according to /Users/MohssineChazi2/moat/antigravity_prd.md and /Users/MohssineChazi2/moat/.agents/ORIGINAL_REQUEST.md.

Protocol & Specifications:
1. Initialize your BRIEFING.md, progress.md, and SCOPE.md in /Users/MohssineChazi2/moat/.agents/sub_orch_m1/.
2. Create config/engine_config.yaml at project root (/Users/MohssineChazi2/moat/config/engine_config.yaml) specifying model settings, INT4 quantization parameters (bits: 4, group_size: 32, superblock_size: 256, alignment_bytes: 128), batch_size: 8, reflection_threshold: 0.75, server port: 8080.
3. Create src/__init__.py and src/dequant.py at project root (/Users/MohssineChazi2/moat/src/dequant.py) implementing:
   - `repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray`: Coalesces 8 fine-grained INT4 quantization groups of size 32 into a single aligned superblock of size 256 aligned to 128-byte boundaries for GPU/NPU SIMD registers.
   - `lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray`: Fast table-lookup dequantization mapping INT4 back to FP16 via vector gather operations over precomputed lookup table.
   - Full quantization packing utilities ensuring 1.5B-3B model parameters fit within ~4.5GB RAM footprint.
4. Create unit test suite in tests/test_quantization.py verifying INT4 repacking accuracy, LUT dequantization speed/correctness, and memory bounds.
5. Perform the iteration loop: Explorer -> Worker -> Reviewer -> Challenger -> Forensic Auditor (`teamwork_preview_auditor`).
   - MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
6. Verify build & unit test results pass cleanly.
7. Upon successful audit and pass criteria, mark milestone complete in SCOPE.md, write handoff.md in your directory, and send completion message to parent (conversation ID: 04a01613-34ab-46c5-8005-aa56ed9b71fe).
