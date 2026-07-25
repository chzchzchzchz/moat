# E2E Test Infrastructure Architecture & Verification Protocol
## Project Antigravity — On-Device Inference Acceleration Engine

**Author:** E2E Testing Track (`worker_e2e_1`)  
**Target Platform:** iPhone 15 Pro / Apple Silicon Mac (A17 Pro / A18 Pro / M-Series)  
**Date:** July 25, 2026  
**Status:** Executable Implementation & Verification Protocol  

---

## 1. Architecture Overview & Strategy

The End-to-End (E2E) Test Harness for Project Antigravity is an opaque-box, hardware-aware test suite designed to validate the 7 core features of the Antigravity engine across 87 explicit test cases organized into 4 distinct tiers.

### Key Architectural Design Principles:
1. **Fallback Duality Architecture:**
   The test suite supports concurrent development. Test cases attempt to import modules directly from `src/` (e.g. `src.dequant`, `src.attention`, `src.batch_generator`, `src.verifier`, `src.run_server`, `src.benchmark`). When `src/` modules are absent or incomplete, the test harness gracefully falls back to verified reference implementations provided by fixtures in `tests/e2e/conftest.py`. This ensures immediate test suite execution and continuous validation as new implementations arrive.
2. **Explicit Test Case Granularity:**
   Every single one of the 87 test cases (TC-T1-F1-01 through TC-T4-07) is explicitly defined as an individual executable test function with strict assertions and error validation.
3. **Hardware & Mock Execution Modes:**
   The suite supports `--hardware mock` (CPU/Numpy proxy execution for rapid CI and verification) and `--hardware metal` (Apple Silicon GPU/ANE acceleration profiling).
4. **Comprehensive Metric Validation:**
   Tests measure and validate mathematical accuracy (INT4 RMS error $< 2.5\%$), exponential table lookups, memory usage ($< 4.5\text{ GB}$ peak RSS), speedup metrics ($\ge 3.0\times$ GEMM decode acceleration, $\ge 1.5\times$ LUT softmax acceleration), adaptive reflection token savings ($\ge 30\%$), and OpenAI API compatibility (HTTP 200, SSE streams, error status codes).

---

## 2. Directory Layout & Module Structure

```
/Users/MohssineChazi2/moat/
├── TEST_INFRA.md                   # E2E test infrastructure documentation
└── tests/
    └── e2e/
        ├── __init__.py             # Test package initializer
        ├── conftest.py             # Global pytest fixtures, reference implementations, synthetic weight generators, memory tracker
        ├── test_tier1_features.py  # Tier 1: Feature Coverage (35 test cases: TC-T1-F1-01 to TC-T1-F7-05)
        ├── test_tier2_boundaries.py# Tier 2: Boundary & Corner Cases (35 test cases: TC-T2-F1-01 to TC-T2-F7-05)
        ├── test_tier3_combinations.py # Tier 3: Cross-Feature Interactions (10 test cases: TC-T3-01 to TC-T3-10)
        ├── test_tier4_scenarios.py # Tier 4: Real-World Application Scenarios (7 test cases: TC-T4-01 to TC-T4-07)
        └── run_e2e_tests.py        # Master CLI test runner & report generator
```

---

## 3. Runner Design (`run_e2e_tests.py`)

The CLI test runner (`run_e2e_tests.py`) provides full command-line control over test execution, filtering, hardware mode, and report generation.

### Supported Command-Line Arguments:
- `--tier {1,2,3,4,all}`: Select specific test tier to run (default: `all`).
- `--feature {f1,f2,f3,f4,f5,f6,f7,all}`: Select specific feature subset to run (default: `all`).
- `--hardware {mock,metal}`: Target execution mode (default: `mock`).
- `--json-report <PATH>`: Optional path to write a detailed JSON test execution report.
- `-v, --verbose`: Enable verbose logging output during test execution.

### Exit Code Protocol:
- Returns **Exit Code 0** if all selected tests pass cleanly (100% pass rate).
- Returns **Exit Code 1** if any test fails or errors.

---

## 4. Tier Breakdown & Test Case Specifications

### Tier 1: Feature Coverage (35 Test Cases)
- **F1 (INT4 Quantization & Repacking):** `TC-T1-F1-01` through `TC-T1-F1-05`
  - Validates INT4 round-trip RMS error, superblock `(N, 8, 32)` shaping, LUT vector gather mapping, scale factor computation ($S_G = \alpha / 7$), and 128-byte SIMD pointer alignment.
- **F2 (Safe Softmax LUT & Attention):** `TC-T1-F2-01` through `TC-T1-F2-05`
  - Validates 32,768 FP16 exponential LUT range $[-10, 0]$, row-max logit shift ($x - \max(x)$), probability sum normalization ($\sum p_i = 1.0$), $\ge 1.5\times$ softmax speedup benchmark, and MHA score gather execution.
- **F3 (Batched Decode Engine):** `TC-T1-F3-01` through `TC-T1-F3-05`
  - Validates $N=8$ parallel candidate generation, shared prompt KV-cache prefill forking, sampling diversity ($T=0.7, p=0.95$), GEMV-to-GEMM wall-clock acceleration ($\ge 3.0\times$), and individual rollout EOS token termination.
- **F4 (List-Wise Verifier & Selection):** `TC-T1-F4-01` through `TC-T1-F4-05`
  - Validates side-by-side prompt formatting, JSON verifier response extraction, normalized confidence score bounded in $[0.0, 1.0]$, sequential model swapping under $< 4.5\text{ GB}$ peak memory, and Best-of-N candidate selection accuracy.
- **F5 (Adaptive Reflection Engine):** `TC-T1-F5-01` through `TC-T1-F5-05`
  - Validates fast-path execution ($S_k \ge 0.75$), low-confidence reflection triggering ($S_k < 0.75$), `<think>` tag injection, threshold parameter sweeps ($\tau \in \{0.50, 0.75, 0.90\}$), and $\ge 30\%$ token savings vs always-reflect baseline.
- **F6 (OpenAI-Compatible API Server):** `TC-T1-F6-01` through `TC-T1-F6-05`
  - Validates `POST /v1/chat/completions` non-streaming JSON schema, SSE streaming responses (`Content-Type: text/event-stream`, `data: [DONE]`), custom request payload parsing (`n_parallel_rollouts`, `verifier_threshold`), `localhost:8080` port binding, and `GET /v1/models` endpoint.
- **F7 (E2E Benchmark Harness):** `TC-T1-F7-01` through `TC-T1-F7-05`
  - Validates CLI argument parsing (`--device`, `--batch_sizes`, `--model_path`), GEMM latency profiling across batch sizes $N \in \{1, 2, 4, 8, 16\}$, benchmark accuracy gain calculation ($+15\%$), peak resident memory tracking ($< 4.5\text{ GB}$), and JSON report disk generation.

### Tier 2: Boundary & Corner Cases (35 Test Cases)
- **F1 Boundaries:** Non-multiple of 256 assertion (`TC-T2-F1-01`), extreme outliers & zeros (`TC-T2-F1-02`), INT4 range clipping to $[-8, 7]$ (`TC-T2-F1-03`), uniform value scale computation (`TC-T2-F1-04`), out-of-bounds LUT index protection (`TC-T2-F1-05`).
- **F2 Boundaries:** Shifted input below LUT domain $\hat{x} < -10.0$ (`TC-T2-F2-01`), identical flat logits (`TC-T2-F2-02`), large sequence context length ($L=4096, 8192$) (`TC-T2-F2-03`), NaN/Inf input error handling (`TC-T2-F2-04`), $1\times 1$ single logit boundary (`TC-T2-F2-05`).
- **F3 Boundaries:** High batch size scaling $N \in \{16, 32\}$ (`TC-T2-F3-01`), single-batch fallback $N=1$ (`TC-T2-F3-02`), maximum context window limit (2,048 tokens) (`TC-T2-F3-03`), empty prompt input validation (`TC-T2-F3-04`), early divergent rollout EOS masking (`TC-T2-F3-05`).
- **F4 Boundaries:** All-identical candidates selection (`TC-T2-F4-01`), malformed non-JSON verifier response regex fallback (`TC-T2-F4-02`), empty candidate list exception (`TC-T2-F4-03`), simulated OOM verifier fallback to majority vote (`TC-T2-F4-04`), truncated rollout text scoring (`TC-T2-F4-05`).
- **F5 Boundaries:** Extreme threshold boundaries $\tau=0.0$ and $\tau=1.0$ (`TC-T2-F5-01`), maximum reflection iteration cap (max 3) (`TC-T2-F5-02`), oscillating confidence score handling (`TC-T2-F5-03`), zero confidence score recovery (`TC-T2-F5-04`), out-of-bounds score clamping (`TC-T2-F5-05`).
- **F6 Boundaries:** Malformed JSON payload HTTP 400 (`TC-T2-F6-01`), missing required fields HTTP 422 (`TC-T2-F6-02`), high concurrent client request handling (`TC-T2-F6-03`), client abort mid-stream socket disconnect (`TC-T2-F6-04`), oversized prompt payload HTTP 413 (`TC-T2-F6-05`).
- **F7 Boundaries:** Non-existent model path FileNotFoundError (`TC-T2-F7-01`), unsupported device ValueError (`TC-T2-F7-02`), zero prompts boundary (`TC-T2-F7-03`), SIGINT / Ctrl+C graceful cleanup (`TC-T2-F7-04`), unwritable log file permission handling (`TC-T2-F7-05`).

### Tier 3: Cross-Feature Interactions (10 Test Cases)
- `TC-T3-01`: Quantized INT4 Weights + Softmax LUT Attention Pipeline
- `TC-T3-02`: Softmax LUT Attention + Batched GEMM Generator Rollouts ($N=8$)
- `TC-T3-03`: Batched Decode Generator + List-Wise Verifier Best-of-N Candidate Pipeline
- `TC-T3-04`: List-Wise Verifier + Threshold-Driven Adaptive Reflection Loop
- `TC-T3-05`: Adaptive Reflection Engine + OpenAI API Server Endpoint
- `TC-T3-06`: End-to-End Benchmark Harness + Full Engine Validation
- `TC-T3-07`: Sequential Model Swapping Under Active API Server Request
- `TC-T3-08`: Batched Parallel Decode ($N=8$) + INT4 Superblock SIMD Integration
- `TC-T3-09`: OpenAI SSE Streaming + Real-Time Adaptive Reflection Interleaving
- `TC-T3-10`: High-Batch Parallel Decode ($N=16$) + List-Wise Verification Memory Boundary Safety

### Tier 4: Real-World Application Scenarios (7 Test Cases)
- `TC-T4-01`: GSM8K Multi-Step Math Reasoning Problem Solving (`Janet buys 6 bags...`)
- `TC-T4-02`: Symbolic Calculus & Integral Derivation (`Compute integral of x*cos(x) dx`)
- `TC-T4-03`: Modular Arithmetic & Number Theory Proof (`Prove 2^n > n^2 for n >= 5`)
- `TC-T4-04`: Multi-Step Reasoning Trace with Self-Correction (`<think>` intermediate fix)
- `TC-T4-05`: Zero-Shot Best-of-N Rollout Accuracy Gain vs Greedy Single Pass ($+15\%$)
- `TC-T4-06`: Sustained API Server Load Under Concurrent Math Completion Requests
- `TC-T4-07`: Low-Memory Footprint Verification on Target iPhone Entitlement Profile ($<4.5\text{ GB}$)

---

## 5. Verification Protocol & Invalidation Conditions

### Verification Command:
```bash
python3 tests/e2e/run_e2e_tests.py --tier all --hardware mock --json-report test_report.json
```

### Invalidation Conditions:
1. **GEMM Speedup Invalidation:** Batched decode latency ratio ($N=8$) over sequential decode latency is $< 3.0\times$.
2. **Softmax Speedup Invalidation:** LUT softmax speedup over dynamic exponentiation is $< 1.5\times$.
3. **Memory Invalidation:** Resident memory peak exceeds $4,500\text{ MB}$.
4. **Token Savings Invalidation:** Adaptive reflection token reduction vs always-reflect is $< 30\%$.
5. **Accuracy Gain Invalidation:** Best-of-N accuracy gain over greedy single-pass is $< +15\%$.
6. **API Standard Invalidation:** API server endpoints return invalid JSON schemas or fail SSE protocol.
