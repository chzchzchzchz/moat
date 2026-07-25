# Implementation Plan: Project Antigravity 🛸

Hardware-accelerated, batched on-device Test-Time Scaling (TTS) engine exploiting underutilized Apple Neural Engine (ANE) and Metal GPU matrix tiles on **iPhone 15 Pro+ / Apple Silicon Macs**. Converts memory-bandwidth-bound single-token GEMV operations into compute-dense batched GEMM operations by running $N = 8$ parallel reasoning traces simultaneously.

## User Review Required

> [!IMPORTANT]
> **iPhone 15 Pro+ Memory Entitlement:** iOS allocates ~4.5GB-5.5GB RAM for app execution using the `com.apple.developer.kernel.increased-memory-limit` entitlement. To safely fit both the 1.5B reasoning model and the 1.5B list-wise verifier within this budget, the engine uses **Sequential Model Swapping** (loading the reasoner for batch rollouts, unloading it, then loading the verifier to score candidate traces).

> [!NOTE]
> **Metal GPU vs ANE Acceleration:** Deep technical research shows that autoregressive decoding with dynamic KV-cache lengths is best targeted to **Metal GPU (via Metal Performance Shaders / simdgroup_matrix)** rather than ANE private graph execution, because ANE requires strict static shapes. Embedding/prefill phases utilize CoreML/ANE.

## Key Research Findings

1. **GEMV vs GEMM Ridge Point:** Single-token decode operates at $\sim 4 \text{ FLOPs/byte}$ (severely memory-bound). Batching $N=8$ parallel candidate traces increases arithmetic intensity to $\sim 32 \text{ FLOPs/byte}$, crossing the Metal/ANE ridge point to achieve dense hardware matrix tile saturation ($3.8\times+$ speedup).
2. **Super-Block Quantization:** INT4 fine-grained group quantization (group size 32) repacked into 256-element super-blocks aligns perfectly with 128-byte hardware cache lines and SIMD registers.
3. **Table-Lookup (LUT) Dequantization & Softmax:** Precomputed 16-element FP16 lookup tables eliminate instruction cycle overhead for weight unpacking. Precomputed 32,768-entry exponential LUTs for softmax deliver a $2.2\times$ speedup in FlashAttention softmax calculations.
4. **List-Wise Verification:** Side-by-side relative scoring of candidate traces outperforms scalar scoring and prevents reward hacking. Threshold-driven adaptive reflection saves $>35\%$ in token budget.

---

## Technical Planning Documents

The architecture is fully documented in five dedicated specification files under `planning/`:
- [01_hardware_architecture.md](file:///Users/MohssineChazi2/moat/planning/01_hardware_architecture.md): ANE/Metal acceleration, GEMV-to-GEMM math, iPhone memory limits.
- [02_quantization_and_lut.md](file:///Users/MohssineChazi2/moat/planning/02_quantization_and_lut.md): INT4 group quantization, 256-element super-block repacking, LUT dequantization, precomputed Softmax LUT.
- [03_batched_decode_engine.md](file:///Users/MohssineChazi2/moat/planning/03_batched_decode_engine.md): Batched parallel rollout coordinator, paged KV-cache, SIMD matrix tiling.
- [04_verifier_and_adaptive_reflection.md](file:///Users/MohssineChazi2/moat/planning/04_verifier_and_adaptive_reflection.md): List-wise verifier, sequential model swapping protocol, adaptive reflection thresholds.
- [05_ios_app_and_api.md](file:///Users/MohssineChazi2/moat/planning/05_ios_app_and_api.md): Swift/Metal integration, OpenAI-compatible `/v1/chat/completions` server, benchmarking harness.

---

## Proposed Changes

### Core Engine (`antigravity-engine/`)

#### [NEW] [engine_config.yaml](file:///Users/MohssineChazi2/moat/antigravity-engine/config/engine_config.yaml)
Configuration file for batch size $N$, group size $g=32$, super-block size $256$, reflection threshold $\tau=0.75$, and model paths.

#### [NEW] [dequant.py](file:///Users/MohssineChazi2/moat/antigravity-engine/src/dequant.py)
INT4 symmetric group quantization (group size 32), 256-element super-block repacking (128-byte aligned), and fast vector table-lookup (LUT) FP16 dequantization.

#### [NEW] [attention.py](file:///Users/MohssineChazi2/moat/antigravity-engine/src/attention.py)
Safe Softmax implementation using row-max subtraction, 32K-entry exponential lookup table (LUT), and vector-gather indexing for FlashAttention.

#### [NEW] [batch_generator.py](file:///Users/MohssineChazi2/moat/antigravity-engine/src/batch_generator.py)
Batched parallel decode rollout coordinator. Manages parallel generation of $N=8$ candidate reasoning sequences simultaneously, converting GEMV to GEMM.

#### [NEW] [verifier.py](file:///Users/MohssineChazi2/moat/antigravity-engine/src/verifier.py)
List-wise candidate verifier and threshold-driven adaptive reflection module. Scores $N=8$ rollouts side-by-side and selects optimal output.

#### [NEW] [reasoning_template.txt](file:///Users/MohssineChazi2/moat/antigravity-engine/prompts/reasoning_template.txt)
System prompt template for zero-shot step-by-step reasoning with `<think>` tags.

#### [NEW] [verifier_template.txt](file:///Users/MohssineChazi2/moat/antigravity-engine/prompts/verifier_template.txt)
JSON-formatted list-wise critic prompt template for candidate evaluation.

#### [NEW] [run_server.py](file:///Users/MohssineChazi2/moat/antigravity-engine/run_server.py)
OpenAI-compatible HTTP server running on `localhost:8080` exposing `/v1/chat/completions` endpoint with SSE streaming.

#### [NEW] [benchmark.py](file:///Users/MohssineChazi2/moat/antigravity-engine/benchmark.py)
End-to-end performance and accuracy profiling script (GEMV vs GEMM latency, memory peak, token efficiency).

#### [NEW] [test_batched_speed.py](file:///Users/MohssineChazi2/moat/antigravity-engine/tests/test_batched_speed.py)
Unit benchmark verifying $\ge 3.0\times$ speedup of batched GEMM vs sequential GEMV.

---

## Verification Plan

### Automated Tests
- Run hardware speedup profiler:
  ```bash
  python3 antigravity-engine/tests/test_batched_speed.py
  ```
- Run quantization & repacking unit tests:
  ```bash
  python3 -m unittest antigravity-engine/tests/test_quantization.py
  ```
- Run end-to-end benchmarking harness:
  ```bash
  python3 antigravity-engine/benchmark.py
  ```

### Manual Verification
- Test OpenAI API endpoint compatibility using `curl`:
  ```bash
  curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Prove that 2^n > n^2 for n >= 5"}]}'
  ```
