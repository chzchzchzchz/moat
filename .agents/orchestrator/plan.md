# Phase 3 Implementation Plan: Project Antigravity 🛸

## Overview
Phase 3 of Project Antigravity connects the verified INT4 super-block quantization math (`dequant.py`), safe exponential lookup table softmax (`attention.py`), and Metal `simdgroup_matrix` compute shaders (`batched_gemm.metal` / `metal_runner`) into a full parallel decoding engine running N=8 candidate reasoning traces concurrently on Apple Silicon GPU/iOS UMA memory.

## Scope & Deliverables
1. **R1: Batched Parallel Decode Rollout Coordinator (`antigravity-engine/src/batch_generator.py`)**
   - Manage N parallel candidate reasoning sequences (N=4, 8, 16).
   - Implement paged KV-cache memory manager with shared prompt prefix block mapping and isolated per-channel blocks (<= 128 MB per trace for seq len 2048).
   - Convert single-token decode steps into dense batched GEMM matrix dispatches targeting compiled Metal SIMD matrix shader (`batched_gemm.metal`) / MPS backend.
   - Temperature-based parallel sampling ($T > 0$, top-p/top-k) and candidate logit gathering.
   - Integrate with `attention.py` (`ExponentialLUT`, `safe_softmax_lut`) and `dequant.py`.

2. **R2: Model Weight Loader & Super-Block Repacker (`antigravity-engine/src/model_loader.py`)**
   - Load FP16/FP32 model weights from GGUF and Safetensors files (or format parsers/mock structures).
   - On-the-fly repacking of weights into 256-element super-blocks (144 bytes: 16-byte FP16 scale header + 128-byte packed INT4 payload).
   - Memory allocation validator guaranteeing total footprint <= 4.5 GB ceiling (and <= 2.5 GB for 1.5B parameters model weights in super-block format).

3. **R3: Integration & End-to-End Test Suite (`antigravity-engine/tests/test_batch_generator.py` & `antigravity-engine/tests/test_model_loader.py`)**
   - Micro-unit and integration tests verifying:
     - Mathematical parity between single-trace decode (N=1) and parallel trace decode (N=8).
     - Paged KV-cache memory isolation across rollout channels.
     - 0 NaN, 0 Inf, 0 memory leaks across 100+ generation steps.
     - All N=8 channels produce coherent, non-identical outputs under sampling temperature $T > 0$.
     - Correct usage of `attention.py` safe softmax LUT and Metal compute shaders.
     - N=8 batched coordinator completes 50 steps in <= 1.0s on Apple Silicon GPU.
     - 100% test pass rate via `python3 -m unittest discover`.

## Orchestration Plan & Workflow Steps

### Step 1: Exploration Phase
- Dispatch 3 parallel Explorers (`teamwork_preview_explorer`) to:
  1. Inspect existing engine components (`dequant.py`, `attention.py`, `batched_gemm.metal`, `metal_runner.cpp`, `test_batched_speed_metal.py`).
  2. Produce concrete design specifications for `batch_generator.py` (Paged KV-cache architecture, batched Metal GEMM binding, parallel temperature sampler).
  3. Produce concrete design specifications for `model_loader.py` (GGUF/Safetensors parser, on-the-fly super-block repacker, memory budget validator).
  4. Outline test cases for `test_batch_generator.py` and `test_model_loader.py`.

### Step 2: Implementation Phase
- Dispatch 1 Worker (`teamwork_preview_worker`) armed with implementation instructions to create:
  - `src/batch_generator.py`
  - `src/model_loader.py`
  - `tests/test_batch_generator.py`
  - `tests/test_model_loader.py`
  - Run `python3 -m unittest discover` and benchmarks to verify execution.

### Step 3: Review Phase
- Dispatch 2 independent Reviewers (`teamwork_preview_reviewer`) to analyze code quality, interface contracts, memory safety, and test coverage.

### Step 4: Empirical Challenge Phase
- Dispatch 2 Challengers (`teamwork_preview_challenger`) to stress-test batched decode performance (N=8 50 steps <= 1.0s), paged KV-cache isolation, and 100+ steps zero NaN/Inf/leak.

### Step 5: Forensic Integrity Audit
- Dispatch 1 Forensic Auditor (`teamwork_preview_auditor`) for mandatory binary veto integrity audit (verifying no hardcoded test outputs, no dummy implementations, authentic GPU/Metal matrix math).

### Step 6: Final Verification & Completion Report
- Verify all gate criteria pass.
- Update BRIEFING.md, PROJECT.md, progress.md, handoff.md.
- Send completion message to Sentinel.
