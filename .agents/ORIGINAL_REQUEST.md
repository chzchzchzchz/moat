# Original User Request

## Initial Request — 2026-07-25T09:12:36Z

Build **Phase 3 of Project Antigravity** — the **Batched Rollout Coordinator** (`batch_generator.py`) and **Model Weight Loader / Repacker** (`model_loader.py`) connecting the verified INT4 super-block math and native Metal `simdgroup_matrix` compute shaders to run N=8 parallel reasoning traces simultaneously on Apple Silicon GPU/iOS.

Working directory: /Users/MohssineChazi2/moat
Integrity mode: development

Reference material:
- Planning specs: /Users/MohssineChazi2/moat/planning/ (01-06)
- Existing engine & shaders: /Users/MohssineChazi2/moat/antigravity-engine/src/ (dequant.py, attention.py, shaders/batched_gemm.metal)
- Hardware benchmarks: /Users/MohssineChazi2/moat/antigravity-engine/tests/test_batched_speed_metal.py

## Requirements

### R1. Batched Parallel Decode Rollout Coordinator (batch_generator.py)
Implement the rollout coordinator that manages N parallel candidate reasoning sequences (N=4, 8, 16).
- Maintain a paged KV-cache allocation across N rollout channels.
- Convert single-token decode steps into dense batched GEMM dispatches targeting the compiled Metal SIMD matrix shader (batched_gemm.metal).
- Implement temperature-based parallel sampling and logit gathering for candidate traces.

### R2. Model Weight Loader & Super-Block Repacker (model_loader.py)
Implement a weight loader for open-weight reasoning models (GGUF / Safetensors formats).
- Parse FP16 / FP32 weight tensors from GGUF / Safetensors files.
- Repack model weights on-the-fly into 256-element super-blocks (144 bytes: 16-byte scale header + 128-byte packed INT4 payload).
- Validate memory allocations to guarantee total footprint remains under the 4.5 GB ceiling.

### R3. Integration & End-to-End Test Suite (test_batch_generator.py)
Create comprehensive micro-unit and integration tests verifying:
- Mathematical parity between single-trace decode (N=1) and parallel trace decode (N=8).
- Paged KV-cache memory isolation across channels (no cross-channel attention corruption).
- 0 NaN, 0 Inf, and 0 memory leaks across 100+ generation steps.

## Acceptance Criteria

### Performance & Scalability
- [ ] Batched rollout coordinator running N=8 channels completes 50 generation steps in <= 1.0 seconds on Apple Silicon GPU
- [ ] Paged KV-cache memory overhead per candidate trace is <= 128 MB for sequence length 2048
- [ ] Quantized model weight memory for 1.5B parameters stays within 2.5 GB in super-block format

### Functional Correctness
- [ ] All N=8 candidate reasoning channels produce coherent, non-identical outputs under sampling temperature T > 0
- [ ] batch_generator.py correctly uses attention.py safe softmax LUT and Metal compute shaders
- [ ] 100% pass rate across new test suite (test_batch_generator.py and test_model_loader.py)
- [ ] Clean build and test execution via python3 -m unittest and native C++ benchmarks
