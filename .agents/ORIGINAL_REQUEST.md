# Original User Request

## Initial Request — 2026-07-25T05:14:56Z

Build **"Project Antigravity"** — a native on-device inference engine for **iPhone 15 Pro and newer** (8GB RAM, latest Apple Neural Engine) that exploits underutilized ANE/GPU matrix tiles during autoregressive LLM decoding. The engine converts wasteful sequential GEMV operations into efficient batched GEMM by running N parallel reasoning traces simultaneously, achieving 4-8x parallel inference for virtually zero additional compute cost. It uses INT4 group quantization with LUT-based dequantization, precomputed softmax via memory-mapped gathers, and a local list-wise verifier to select the best output — all running entirely on-device with zero cloud dependency. The quality bar is **Apple-acquisition-grade**: this should demonstrate a genuine breakthrough in on-device inference that Apple engineering would want to acquire.

Development platform: Apple Silicon Mac.
Deployment target: iOS / iPhone 15 Pro+ (A17 Pro, A18 Pro chips — 8GB unified memory, latest ANE).

Working directory: /Users/MohssineChazi2/moat
Integrity mode: development

Reference material:
- PRD and existing code shells: /Users/MohssineChazi2/moat/antigravity_prd.md

## Requirements

### R1. Batched Parallel Decode Engine (GEMV → GEMM Conversion)

The core inference engine must run N parallel candidate reasoning traces (N=4, 8, or 16) simultaneously during autoregressive decode. By batching these traces together, the engine must convert single-token GEMV operations into dense GEMM operations, utilizing the iPhone's ANE/GPU matrix multiplication hardware that would otherwise sit idle during standard sequential decoding. The engine must demonstrate measurable speedup (target: ≥3x) when comparing batched parallel decode vs. N sequential single-pass decodes on the target iPhone hardware.

### R2. INT4 Quantization Pipeline with LUT Dequantization

The engine must compress target open-weight reasoning models (in the 1.5B–3B parameter range) to INT4 precision using fine-grained group quantization (group size 32), then repack weights into 256-element super-blocks aligned to hardware vector register boundaries (128-byte). At runtime, dequantization must use precomputed lookup tables (LUT) with native vector table-lookup operations rather than costly bit-masking/unpacking instruction sequences. The full quantized model must fit within ~4-5GB of usable iPhone memory (accounting for iOS overhead from 8GB total).

### R3. Optimized Attention with Precomputed Softmax

The attention mechanism must use a precomputed exponential lookup table (~32K entries, ~64KB) for softmax computation instead of dynamic floating-point exponentials. It must implement safe softmax (row-wise max subtraction) and use vector-gather index lookups. This must show measurable improvement over standard softmax implementations when running parallel batched attention.

### R4. Local List-Wise Verifier and Best-of-N Selection

After generating N parallel reasoning traces, the engine must run a lightweight local verifier model that compares all candidates list-wise (side-by-side comparison, not scalar scoring) and selects the highest-quality output. The system must also implement threshold-driven adaptive reflection — triggering re-generation only when a step's verifier score falls below a configurable threshold, to avoid wasting token budget on unnecessary reflection.

### R5. iPhone-Native Deployment with Local API

The complete pipeline must run natively on iPhone 15 Pro+ hardware using appropriate Apple frameworks (CoreML, Metal, or equivalent). It must expose a local API endpoint (Unix socket or localhost HTTP) that is compatible with the OpenAI `/v1/chat/completions` format, allowing any on-device app to interact with the inference engine via standard requests. The engine must operate fully offline with zero network dependency.

## Acceptance Criteria

### Performance Benchmarks
- [ ] Batched decode (N=8) achieves ≥3x wall-clock speedup over 8 sequential single-pass decodes on iPhone 15 Pro or Apple Silicon Mac (as development proxy)
- [ ] A quantized 1.5B-3B model fits within 4.5GB memory footprint when loaded on device
- [ ] End-to-end inference latency for a single user query (with N=8 parallel traces + verification) completes within 30 seconds on target hardware
- [ ] LUT-based softmax demonstrates measurable speedup (≥1.5x) over standard exp-based softmax in a standalone micro-benchmark

### Functional Correctness
- [ ] The quantized model produces coherent, correct outputs on at least 10 diverse test prompts (math, reasoning, general knowledge)
- [ ] Best-of-N selection (N=8) with list-wise verification produces higher-quality outputs than single-pass greedy decode on a set of 20+ math/reasoning problems (measured by answer correctness rate)
- [ ] The local API endpoint correctly handles OpenAI-compatible chat completion requests and returns properly formatted responses
- [ ] Threshold-based adaptive reflection uses ≥30% fewer tokens than always-reflect baseline while maintaining equivalent output quality

### Build & Run
- [ ] The project builds and runs on an Apple Silicon Mac (development proxy) without manual intervention beyond documented setup steps
- [ ] Clear documentation exists: README with architecture overview, setup instructions, and benchmark reproduction steps
- [ ] A benchmark harness script can be run to reproduce all performance claims
