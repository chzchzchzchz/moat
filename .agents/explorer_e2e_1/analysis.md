# E2E Test Suite Architecture & Test Matrix Analysis
## Project Antigravity — On-Device Inference Acceleration Engine

**Author:** Explorer Subagent (`explorer_e2e_1`)  
**Target Platform:** iPhone 15 Pro / Apple Silicon Mac (A17 Pro / A18 Pro / M-Series)  
**Date:** July 25, 2026  
**Status:** Completed Analysis & Architecture Specification  

---

## 1. Executive Summary & Strategy

Project Antigravity is a native on-device inference optimization engine designed for consumer Apple Silicon hardware (iPhone 15 Pro+, A17 Pro/A18 Pro, 8GB unified memory). Its core innovation converts memory-bandwidth-bound single-token General Matrix-Vector Multiplication (GEMV) into compute-dense General Matrix-Matrix Multiplication (GEMM) by executing $N$ parallel candidate reasoning trajectories ($N=4, 8, 16$) simultaneously during autoregressive decoding.

To validate this high-performance engine with an **Apple-acquisition-grade quality bar**, this document establishes an **opaque-box End-to-End (E2E) Test Suite Architecture** and a **87-case Test Matrix** across 4 testing tiers. The test harness isolates external interface contracts, memory constraints, latency acceleration bounds, numerical precision stability, API standards compliance, and real-world reasoning performance without relying on internal private state access.

### Core Testing Pillars:
1. **Performance & Hardware Acceleration:** Verifying $\ge 3.0\times$ speedup of batched GEMM vs sequential GEMV, $\ge 1.5\times$ speedup of LUT-based softmax, and $<4.5$GB RAM peak allocation.
2. **Numerical & Quantization Stability:** Validating INT4 group quantization (group size 32), 256-element superblock repacking, and LUT vector gather accuracy.
3. **Reasoning & Verifier Integrity:** Ensuring list-wise candidate verification, side-by-side trajectory ranking, and adaptive reflection token savings ($\ge 30\%$).
4. **API & System Protocol Compliance:** Testing drop-in OpenAI `/v1/chat/completions` HTTP/SSE interface under single and concurrent request workloads.

---

## 2. System Requirement & PRD Feature Mapping

The E2E test suite targets 7 core features ($F1 \dots F7$), mapped directly to PRD specifications and technical planning documents:

| Feature ID | Feature Name | Core Target File | PRD / Planning Reference | Key Functional Metric |
| :--- | :--- | :--- | :--- | :--- |
| **F1** | INT4 Group Quantization & Superblock Repacking | `dequant.py` | PRD §4.A, Plan 02 | Group size 32, 256-elem (128-byte) alignment, LUT mapping |
| **F2** | Safe Softmax LUT & Attention | `attention.py` | PRD §4.B, Plan 02 | 32K entry exponential LUT, row-max shift, $\ge 1.5\times$ speedup |
| **F3** | Batched Decode Engine (GEMV $\rightarrow$ GEMM) | `batch_generator.py` | PRD §2, §4, Plan 03 | $N=8$ parallel rollouts, paged KV-cache, $\ge 3.0\times$ speedup |
| **F4** | List-Wise Verifier & Best-of-N Selection | `verifier.py` | PRD §3, Plan 04 | Side-by-side ranking, model swapping under 4.5GB RAM |
| **F5** | Threshold-Driven Adaptive Reflection Engine | `verifier.py` | PRD §5, Plan 04 | $\tau=0.75$ threshold, step reflection, $\ge 30\%$ token savings |
| **F6** | Local OpenAI-Compatible API Server | `run_server.py` | PRD §5, Plan 05 | `localhost:8080`, `/v1/chat/completions` POST/SSE stream |
| **F7** | E2E Benchmark & Performance Harness | `benchmark.py` | PRD §6, Plan 05 | Accuracy $+15\%$, latency profiling, memory verification |

---

## 3. Comprehensive 87-Case Test Matrix

The test matrix comprises **87 explicit test cases** categorized into 4 tiers:
- **Tier 1:** Feature Coverage (35 test cases: 5 per feature across 7 features) — Happy path & functional contract verification.
- **Tier 2:** Boundary & Corner Cases (35 test cases: 5 per feature across 7 features) — Extreme values, empty inputs, memory caps, error handling.
- **Tier 3:** Cross-Feature Interactions (10 test cases) — End-to-end multi-module pipeline integration.
- **Tier 4:** Real-World Application Scenarios (7 test cases) — Complex math reasoning, multi-turn rollouts, concurrent API load.

---

### Tier 1: Feature Coverage (35 Test Cases)

#### F1: INT4 Group Quantization & Superblock Repacking (`dequant.py`)
- **TC-T1-F1-01: Quantization & Dequantization Round-Trip Accuracy**
  - *Description:* Verify that FP16 weights quantized to INT4 (group size 32) and dequantized via LUT maintain relative RMS error $< 2.5\%$.
  - *Inputs:* Random Gaussian FP16 weight matrix of shape `[2048, 2048]`.
  - *Expected Outcome:* Reconstructed weights adhere to precision tolerances; shape matches original.
- **TC-T1-F1-02: Superblock Repacking Shape & Structure**
  - *Description:* Validate that `repack_weights_to_superblock` transforms `[N*256]` INT4 weights into `(N, 8, 32)` Superblocks.
  - *Inputs:* Array of 2,048 INT4 elements.
  - *Expected Outcome:* Output shape is `(8, 8, 32)`, data type is `int8`.
- **TC-T1-F1-03: FP16 LUT Dequantization Vector Gather Mapping**
  - *Description:* Confirm `lut_dequantize_fp16` performs exact vector gathering without arithmetic bit-shifts.
  - *Inputs:* Quantized indices array `[-8, -4, 0, 3, 7]` and precomputed FP16 LUT table.
  - *Expected Outcome:* Output values match exact LUT indexed positions `lut[q_weights]`.
- **TC-T1-F1-04: Group Size 32 Scale Factor Computation**
  - *Description:* Verify group scaling factor calculation $S_G = \alpha / 7$ where $\alpha = \max |w_i|$ per 32 elements.
  - *Inputs:* 32 FP16 elements with known maximum absolute value $\alpha = 14.0$.
  - *Expected Outcome:* Scale factor $S_G = 2.0$, quantized values clamped to $[-8, 7]$.
- **TC-T1-F1-05: 128-Byte Register Boundary Alignment**
  - *Description:* Check memory address offset of repacked superblock arrays for 128-byte SIMD alignment.
  - *Inputs:* Repacked superblock numpy/Metal byte buffer.
  - *Expected Outcome:* Memory buffer pointer address `% 128 == 0`.

#### F2: Safe Softmax LUT & Attention (`attention.py`)
- **TC-T1-F2-01: SafeSoftmaxLUT Initialization & Range**
  - *Description:* Verify LUT initializes 32,768 FP16 exponentials over $[-10.0, 0.0]$.
  - *Inputs:* `SafeSoftmaxLUT(size=32768)`.
  - *Expected Outcome:* `lut[0] == exp(-10.0)`, `lut[-1] == exp(0.0) == 1.0`, step size $= 10.0 / 32767$.
- **TC-T1-F2-02: Safe Softmax Row-Max Shift Subtraction**
  - *Description:* Verify input logit shift $x - \max(x)$ ensures all exponential inputs are non-positive ($\le 0$).
  - *Inputs:* Input vector $x = [12.5, 45.2, -3.1, 100.0]$.
  - *Expected Outcome:* Shifted vector $\hat{x} = [-87.5, -54.8, -103.1, 0.0]$, max element $= 0.0$.
- **TC-T1-F2-03: Probability Sum Normalization**
  - *Description:* Confirm output softmax probabilities sum to exactly $1.0 \pm 1e-4$.
  - *Inputs:* Logits matrix `[8, 128]` (Batch size 8, sequence length 128).
  - *Expected Outcome:* Row-wise sums equal $1.0$ across all batch rows.
- **TC-T1-F2-04: Standalone LUT Softmax Speedup Benchmark**
  - *Description:* Benchmark LUT vector gather softmax against `scipy.special.softmax` / dynamic `np.exp`.
  - *Inputs:* Logits matrix `[16, 2048]`, 1,000 iterations.
  - *Expected Outcome:* Speedup factor $\ge 1.5\times$ over dynamic floating-point exponentiation.
- **TC-T1-F2-05: Multi-Head Attention Score Gather Execution**
  - *Description:* Validate integration of `compute_softmax` inside multi-head attention projection matrix.
  - *Inputs:* $Q, K, V$ matrices with shape `[batch=8, heads=16, seq=64, dim=64]`.
  - *Expected Outcome:* Attention weights properly normalized, output dimension `[8, 16, 64, 64]`.

#### F3: Batched Parallel Decode Engine (`batch_generator.py`)
- **TC-T1-F3-01: Parallel Trajectory Generation (Batch Size N=8)**
  - *Description:* Run parallel decode coordinator to generate $N=8$ distinct candidate reasoning traces.
  - *Inputs:* Prompt: `"Solve for x: 2x + 6 = 14"`, `num_samples=8`.
  - *Expected Outcome:* 8 candidate string sequences returned, generated concurrently.
- **TC-T1-F3-02: Shared Prompt KV-Cache Forking**
  - *Description:* Confirm prompt prefill processes initial prompt once, sharing physical KV blocks across 8 rollouts.
  - *Inputs:* 50-token prompt prefill pass.
  - *Expected Outcome:* Prompt KV block allocation count equals single-prompt footprint, not $8\times$.
- **TC-T1-F3-03: Temperature and Top-P Sampling Diversity**
  - *Description:* Verify non-zero temperature ($T=0.7$) and top-p ($p=0.95$) generate diverse candidate traces.
  - *Inputs:* Prompt: `"What are three distinct methods to integrate x*sin(x)?"`, `N=8`.
  - *Expected Outcome:* Unique candidate count $\ge 6$ out of 8 traces.
- **TC-T1-F3-04: GEMV to GEMM Latency Acceleration Verification**
  - *Description:* Measure decode step latency for $N=8$ parallel rollouts vs 8 sequential single rollouts.
  - *Inputs:* 100 decode steps on target hardware.
  - *Expected Outcome:* Wall-clock speedup factor $\ge 3.0\times$ (Target $\sim 3.83\times$).
- **TC-T1-F3-05: Individual EOS Token Termination**
  - *Description:* Confirm individual rollouts stop generation when emitting EOS token while other channels continue.
  - *Inputs:* `num_samples=4`, variable length responses.
  - *Expected Outcome:* Completed channels freeze KV length; active channels complete generation safely.

#### F4: List-Wise Verifier & Best-of-N Selection (`verifier.py`)
- **TC-T1-F4-01: List-Wise Side-by-Side Evaluation Prompt Formatting**
  - *Description:* Verify `verifier.py` formats all $N=8$ candidate solutions into a single list-wise critic prompt.
  - *Inputs:* Problem string and list of 8 generated text rollouts.
  - *Expected Outcome:* Prompt contains structured JSON instruction with all 8 candidate strings.
- **TC-T1-F4-02: Verifier Response Parsing & Candidate Index Selection**
  - *Description:* Validate extraction of `selected_index` and `analysis` from verifier model output.
  - *Inputs:* Verifier raw output JSON: `{"index": 3, "analysis": "Step 2 is clean."}`.
  - *Expected Outcome:* Selected index extracted as integer `3`, rationale string preserved.
- **TC-T1-F4-03: Confidence Score Output Normalization**
  - *Description:* Confirm verifier confidence score is bounded in $[0.0, 1.0]$.
  - *Inputs:* Verifier evaluation payload.
  - *Expected Outcome:* Float score within range.
- **TC-T1-F4-04: Sequential Model Swapping Execution**
  - *Description:* Verify reasoner model is purged from memory before verifier model weights are loaded.
  - *Inputs:* Full generation-verification cycle.
  - *Expected Outcome:* Peak memory consumption remains $<4.5$GB during transition.
- **TC-T1-F4-05: Best-of-N Output Selection Correctness**
  - *Description:* Test selection of mathematically correct trajectory over flawed trajectories.
  - *Inputs:* 7 flawed candidates (wrong arithmetic) + 1 correct candidate.
  - *Expected Outcome:* Verifier selects the index of the correct candidate.

#### F5: Threshold-Driven Adaptive Reflection Engine (`verifier.py`)
- **TC-T1-F5-01: High-Confidence Fast-Path Execution ($S_k \ge 0.75$)**
  - *Description:* Confirm reasoning proceeds directly without reflection when step score $S_k \ge \tau = 0.75$.
  - *Inputs:* Candidate trace with step score $S_k = 0.88$.
  - *Expected Outcome:* Zero reflection prompt inserted; generation completes in fast-path mode.
- **TC-T1-F5-02: Low-Confidence Reflection Triggering ($S_k < 0.75$)**
  - *Description:* Confirm reflection is triggered when step score $S_k < \tau = 0.75$.
  - *Inputs:* Candidate trace with step score $S_k = 0.55$.
  - *Expected Outcome:* Reflection path triggered, trajectory re-evaluated.
- **TC-T1-F5-03: Reflection Prompt Tag Injection**
  - *Description:* Verify `<think> Re-evaluating previous step... </think>` tag is injected into rollout context.
  - *Inputs:* Triggered reflection state.
  - *Expected Outcome:* Generation prompt includes `<think>` tag prefix for subsequent tokens.
- **TC-T1-F5-04: Threshold Sensitivity Sweep**
  - *Description:* Evaluate system behavior across thresholds $\tau \in \{0.50, 0.75, 0.90\}$.
  - *Inputs:* Benchmark dataset of 10 math problems.
  - *Expected Outcome:* Higher $\tau$ triggers more reflections; $\tau=0.75$ balances accuracy and token usage.
- **TC-T1-F5-05: Token Reduction Measurement vs Always-Reflect Baseline**
  - *Description:* Measure total tokens consumed by adaptive reflection ($\tau=0.75$) vs always-reflect baseline.
  - *Inputs:* 20 reasoning problems.
  - *Expected Outcome:* Adaptive reflection consumes $\ge 30\%$ fewer tokens (Target $\ge 35\%$).

#### F6: Local OpenAI-Compatible API Server (`run_server.py`)
- **TC-T1-F6-01: `POST /v1/chat/completions` Non-Streaming Response**
  - *Description:* Verify standard non-streaming HTTP POST request returns valid OpenAI JSON format.
  - *Inputs:* `POST /v1/chat/completions` payload `{"model": "antigravity-1.5b", "messages": [{"role": "user", "content": "Hi"}]}`.
  - *Expected Outcome:* HTTP 200, valid JSON response with `id`, `object="chat.completion"`, `choices`.
- **TC-T1-F6-02: `POST /v1/chat/completions` SSE Streaming Response**
  - *Description:* Verify `stream=true` returns Server-Sent Events (SSE) stream.
  - *Inputs:* Request with `"stream": true`.
  - *Expected Outcome:* Response headers `Content-Type: text/event-stream`, data chunks formatted as `data: {...}`, ending with `data: [DONE]`.
- **TC-T1-F6-03: Custom API Parameters Parsing**
  - *Description:* Confirm server parses `n_parallel_rollouts` and `verifier_threshold` from request payload.
  - *Inputs:* Payload with `"n_parallel_rollouts": 4, "verifier_threshold": 0.80`.
  - *Expected Outcome:* Engine executes with $N=4$ rollouts and threshold $0.80$.
- **TC-T1-F6-04: Local Server Binding & Socket Listening**
  - *Description:* Confirm server successfully binds to `localhost:8080` or Unix socket `/tmp/antigravity.sock`.
  - *Inputs:* `python3 run_server.py --port 8080`.
  - *Expected Outcome:* Port 8080 open and responding to GET/POST requests.
- **TC-T1-F6-05: Model Info Endpoint (`/v1/models`)**
  - *Description:* Verify GET `/v1/models` returns model metadata list.
  - *Inputs:* `GET /v1/models`.
  - *Expected Outcome:* HTTP 200, JSON list containing `antigravity-qwen2.5-1.5b-tts`.

#### F7: End-to-End Benchmark Harness (`benchmark.py`)
- **TC-T1-F7-01: `benchmark.py` CLI Argument Parsing**
  - *Description:* Validate command line parameters `--device`, `--batch_sizes`, `--model_path`.
  - *Inputs:* Executing `python3 benchmark.py --device mac --batch_sizes 1,2,4,8`.
  - *Expected Outcome:* Arguments parsed correctly without syntax or type errors.
- **TC-T1-F7-02: GEMM Acceleration Profiling Across Batch Sizes**
  - *Description:* Execute latency profiling across $N \in \{1, 2, 4, 8, 16\}$.
  - *Inputs:* Benchmark execution run.
  - *Expected Outcome:* Output logs report step latency for each batch size and calculate speedup ratios.
- **TC-T1-F7-03: Benchmark Accuracy Gain Calculation**
  - *Description:* Verify accuracy gain computation comparing Best-of-N vs zero-shot greedy single pass.
  - *Inputs:* Accuracy test dataset results.
  - *Expected Outcome:* Calculates percentage accuracy gain (target $+15\%$).
- **TC-T1-F7-04: Memory Footprint Peak Tracking**
  - *Description:* Track peak resident memory (RSS) during benchmark execution.
  - *Inputs:* Full benchmark execution cycle.
  - *Expected Outcome:* Log reports peak memory usage in MB, confirming $< 4500$ MB.
- **TC-T1-F7-05: Automated Benchmark JSON Report Generation**
  - *Description:* Confirm benchmark writes detailed JSON report to disk.
  - *Inputs:* Benchmark completion run with `--output_json report.json`.
  - *Expected Outcome:* `report.json` created containing all timing, accuracy, memory, and token metrics.

---

### Tier 2: Boundary & Corner Cases (35 Test Cases)

#### F1: INT4 Group Quantization & Superblock Repacking
- **TC-T2-F1-01: Non-Multiple of 256 Weight Length Exception**
  - *Description:* Assert error is raised when input weight array length is not divisible by 256.
  - *Inputs:* Array of length 200 INT4 elements.
  - *Expected Outcome:* `AssertionError: Weight array length must be a multiple of 256.`
- **TC-T2-F1-02: Extreme Weight Values (Outliers & Uniform Zeros)**
  - *Description:* Quantize array containing all zeros and array containing extreme outliers ($1e5$).
  - *Inputs:* `weights = np.zeros(256)` and `weights[0] = 1e5`.
  - *Expected Outcome:* Handles zero scale factor without divide-by-zero NaN; clamps outliers safely.
- **TC-T2-F1-03: Quantization Range Clipping Boundary**
  - *Description:* Verify strictly enforced INT4 range limits $[-8, 7]$.
  - *Inputs:* Values mapping to $+10$ or $-12$ prior to clamping.
  - *Expected Outcome:* Output values strictly within $[-8, 7]$.
- **TC-T2-F1-04: Single-Group Uniform Value Scale Computation**
  - *Description:* Test group of 32 identical non-zero values ($w_i = 3.5$).
  - *Inputs:* 32 FP16 elements equal to $3.5$.
  - *Expected Outcome:* Scale factor $S_G = 0.5$, quantized values all equal $7$.
- **TC-T2-F1-05: Out-of-Bounds LUT Index Protection**
  - *Description:* Pass invalid quantized index (e.g. $15$ or $-10$) to `lut_dequantize_fp16`.
  - *Inputs:* Index array out of $[-8, 7]$ bounds.
  - *Expected Outcome:* Indices clipped or validated prior to lookup; no segfault or memory index exception.

#### F2: Safe Softmax LUT & Attention
- **TC-T2-F2-01: Extreme Negative Shift Input Lower Bound Clipping**
  - *Description:* Test shifted input $\hat{x} < -10.0$ (below LUT precomputed domain $[-10.0, 0.0]$).
  - *Inputs:* Shifted value $\hat{x} = -25.0$.
  - *Expected Outcome:* Index clipped to $0$ (`lut[0] = exp(-10.0)`), output probability approaches zero safely.
- **TC-T2-F2-02: Identical Input Logits (Flat Distribution)**
  - *Description:* Test vector where all logits are equal ($x = [5.0, 5.0, 5.0, 5.0]$).
  - *Inputs:* Flat vector of 4 elements.
  - *Expected Outcome:* Softmax output is uniform $[0.25, 0.25, 0.25, 0.25]$.
- **TC-T2-F2-03: Large Sequence Length Context Scaling**
  - *Description:* Test softmax execution over large sequence lengths ($L = 4096, 8192$).
  - *Inputs:* Logit matrix `[1, 4096]`.
  - *Expected Outcome:* Normalization remains numerically stable; no NaN/Inf underflow/overflow.
- **TC-T2-F2-04: NaN and Infinity Logit Error Handling**
  - *Description:* Pass logits containing `np.nan` or `np.inf`.
  - *Inputs:* Vector $[1.0, \text{NaN}, 3.0]$.
  - *Expected Outcome:* Softmax kernel raises `ValueError` or sanitizes non-finite values safely.
- **TC-T2-F2-05: Single-Element Logit Vector Boundary**
  - *Description:* Test $1 \times 1$ logit matrix $x = [[4.2]]$.
  - *Expected Outcome:* Output probability equals exactly $[[1.0]]$.

#### F3: Batched Parallel Decode Engine
- **TC-T2-F3-01: High Batch Size Scaling ($N = 16, 32$)**
  - *Description:* Test parallel decode generator with high batch sizes beyond default $N=8$.
  - *Inputs:* `num_samples=16` and `num_samples=32`.
  - *Expected Outcome:* Memory buffers scale cleanly; no OOM crash; GEMM performance monitored.
- **TC-T2-F3-02: Single-Batch Fallback ($N = 1$)**
  - *Description:* Verify system falls back gracefully to standard single-pass decode when $N=1$.
  - *Inputs:* `num_samples=1`.
  - *Expected Outcome:* Operation executes correctly as GEMV without matrix shape mismatch errors.
- **TC-T2-F3-03: Maximum Context Window Memory Bound**
  - *Description:* Generate tokens until reaching maximum context window limit (2,048 tokens).
  - *Inputs:* Long prompt + generation target exceeding context cap.
  - *Expected Outcome:* Paged KV-cache manages memory blocks cleanly; truncates or raises clean context exhaust alert.
- **TC-T2-F3-04: Empty Prompt Input Handling**
  - *Description:* Pass empty string `""` or whitespace-only prompt to generator.
  - *Inputs:* Prompt `""`.
  - *Expected Outcome:* Server/generator returns HTTP 400 or raises validation exception.
- **TC-T2-F3-05: Early Divergent Trajectory Termination**
  - *Description:* Test scenario where Rollout 1 finishes at token 5, while Rollouts 2..8 generate up to 200 tokens.
  - *Inputs:* Diverse length prompts.
  - *Expected Outcome:* Batch dimension dynamically masks inactive rollouts; GEMM continues for active sequences.

#### F4: List-Wise Verifier & Selection
- **TC-T2-F4-01: All-Identical Candidate Solutions Scoring**
  - *Description:* Pass 8 identical candidate rollout strings to list-wise verifier.
  - *Inputs:* 8 duplicate text strings.
  - *Expected Outcome:* Verifier selects index 0 without crashing or division-by-zero in relative ranking.
- **TC-T2-F4-02: Malformed Non-JSON Verifier Response Fallback**
  - *Description:* Test verifier output returning plain text instead of requested JSON format.
  - *Inputs:* Simulated verifier response: `"I think candidate 2 is best."`
  - *Expected Outcome:* Regex parser fallback extracts index 2; system does not crash.
- **TC-T2-F4-03: Empty Candidate List Exception**
  - *Description:* Pass empty candidate list `[]` to `verify_candidates`.
  - *Inputs:* `candidates=[]`.
  - *Expected Outcome:* Raises `ValueError("Candidate list cannot be empty.")`.
- **TC-T2-F4-04: Simulated OOM During Verifier Load**
  - *Description:* Simulate memory allocation failure when loading verifier weights.
  - *Inputs:* Simulated memory pressure trigger.
  - *Expected Outcome:* Clean fallback to majority-vote consensus among rollouts without crashing.
- **TC-T2-F4-05: Truncated Candidate Text Traces**
  - *Description:* Pass candidate rollouts with incomplete `<think>` tags or broken formatting.
  - *Inputs:* Candidates ending mid-sentence.
  - *Expected Outcome:* Verifier penalizes incomplete traces and ranks completed traces higher.

#### F5: Threshold-Driven Adaptive Reflection Engine
- **TC-T2-F5-01: Extreme Threshold Values ($\tau = 0.0$ and $\tau = 1.0$)**
  - *Description:* Test boundary threshold settings: $\tau=0.0$ (never reflect) and $\tau=1.0$ (always reflect).
  - *Inputs:* $\tau=0.0$ and $\tau=1.0$ on identical prompts.
  - *Expected Outcome:* $\tau=0.0$ executes 0 reflections; $\tau=1.0$ triggers reflection at every step.
- **TC-T2-F5-02: Maximum Reflection Iteration Cap**
  - *Description:* Verify system enforces maximum reflection limit per problem (e.g. max 3 reflections).
  - *Inputs:* Persistent low score $S_k = 0.20$ across all steps.
  - *Expected Outcome:* Stops reflecting after 3 attempts; prevents infinite loop and token exhaustion.
- **TC-T2-F5-03: Rapid Confidence Score Fluctuation Handling**
  - *Description:* Test sequence of scores oscillating above and below threshold ($0.80, 0.40, 0.85, 0.30$).
  - *Inputs:* Oscillating confidence input series.
  - *Expected Outcome:* System correctly toggles fast-path and reflection path without state corruption.
- **TC-T2-F5-04: Zero Confidence Score ($S_k = 0.0$) Recovery**
  - *Description:* Test verifier returning score of $0.0$ due to fatal reasoning error.
  - *Inputs:* $S_k = 0.0$.
  - *Expected Outcome:* Immediately triggers reflection and trajectory resample.
- **TC-T2-F5-05: Invalid Score Out-of-Bounds Error Handling**
  - *Description:* Pass invalid verifier score $S_k = 1.5$ or $S_k = -0.5$.
  - *Inputs:* Out-of-bounds float scores.
  - *Expected Outcome:* Score clamped to $[0.0, 1.0]$ with warning logged.

#### F6: Local OpenAI-Compatible API Server
- **TC-T2-F6-01: Malformed JSON Payload Handling (HTTP 400)**
  - *Description:* Send invalid JSON body to `/v1/chat/completions`.
  - *Inputs:* `curl POST` with body `{"messages": [invalid_json...`.
  - *Expected Outcome:* Server responds with `HTTP 400 Bad Request` and JSON error message.
- **TC-T2-F6-02: Missing Required Request Fields (HTTP 422)**
  - *Description:* Omit required `"messages"` key from POST payload.
  - *Inputs:* `{"model": "antigravity-1.5b"}`.
  - *Expected Outcome:* Server responds with `HTTP 422 Unprocessable Entity`.
- **TC-T2-F6-03: High Concurrent Client Request Handling**
  - *Description:* Send 50 concurrent request connections to `localhost:8080`.
  - *Inputs:* Concurrent load tool / multithreaded client script.
  - *Expected Outcome:* Server queues or processes requests without dropping socket connections or crashing.
- **TC-T2-F6-04: Client Abort / Socket Disconnect Mid-Stream**
  - *Description:* Terminate client connection abruptly while server is streaming SSE data.
  - *Inputs:* Client closes TCP connection after receiving 2 SSE chunks.
  - *Expected Outcome:* Server catches `BrokenPipeError` / socket closed, releases GPU/memory resources cleanly.
- **TC-T2-F6-05: Oversized Prompt Payload Handling (>1MB)**
  - *Description:* Send request containing 1.5MB text string in prompt message.
  - *Inputs:* Payload size > 1MB.
  - *Expected Outcome:* Server rejects with `HTTP 413 Payload Too Large` or truncates gracefully.

#### F7: End-to-End Benchmark Harness
- **TC-T2-F7-01: Non-Existent Model Path Handling**
  - *Description:* Run benchmark with invalid path `--model_path /tmp/nonexistent.bin`.
  - *Inputs:* Invalid file path CLI flag.
  - *Expected Outcome:* Aborts cleanly with `FileNotFoundError` and user-friendly error message.
- **TC-T2-F7-02: Unsupported Target Device Flag**
  - *Description:* Pass `--device invalid_hardware` to `benchmark.py`.
  - *Inputs:* Invalid device flag.
  - *Expected Outcome:* Raises `ValueError: Unsupported device 'invalid_hardware'.`
- **TC-T2-F7-03: Zero Prompts Input Boundary**
  - *Description:* Run benchmark with `--num_prompts 0`.
  - *Inputs:* `num_prompts=0`.
  - *Expected Outcome:* Logs warning and exits cleanly without divide-by-zero error.
- **TC-T2-F7-04: Benchmark Interruption (SIGINT / Ctrl+C) Graceful Cleanup**
  - *Description:* Send SIGINT signal while benchmark is running.
  - *Inputs:* SIGINT signal during execution.
  - *Expected Outcome:* Benchmark catches signal, releases Metal/GPU memory, writes partial results JSON.
- **TC-T2-F7-05: Log File Write Permission Failure Handling**
  - *Description:* Run benchmark with output JSON path set to read-only directory (`/sys/report.json`).
  - *Inputs:* Unwritable output path.
  - *Expected Outcome:* Prints benchmark results to stdout and logs warning regarding file write failure.

---

### Tier 3: Cross-Feature Interactions (10 Test Cases)

- **TC-T3-01: Quantized INT4 Weights + Softmax LUT Attention Pipeline**
  - *Description:* Connect `dequant.py` LUT dequantized FP16 weights into `attention.py` Safe Softmax attention layers.
  - *Inputs:* 4-bit repacked weight superblocks + attention score matrix.
  - *Expected Outcome:* End-to-end forward pass executes cleanly with accurate FP16 activations.
- **TC-T3-02: Softmax LUT Attention + Batched GEMM Generator Rollouts ($N=8$)**
  - *Description:* Integrate `SafeSoftmaxLUT` directly inside `batch_generator.py` multi-head attention step.
  - *Inputs:* Parallel decode loop running 8 sequences through LUT softmax attention.
  - *Expected Outcome:* Generates 8 rollouts concurrently while maintaining LUT softmax speedup.
- **TC-T3-03: Batched Decode Generator + List-Wise Verifier Best-of-N Candidate Pipeline**
  - *Description:* Feed 8 parallel rollouts produced by `batch_generator.py` directly into `verifier.py`.
  - *Inputs:* Raw prompt $\rightarrow$ parallel rollout $\rightarrow$ list-wise verification.
  - *Expected Outcome:* Pipeline selects top candidate trace with validated correctness.
- **TC-T3-04: List-Wise Verifier + Threshold-Driven Adaptive Reflection Loop**
  - *Description:* Execute adaptive reflection loop using verifier feedback to trigger re-generation.
  - *Inputs:* Multi-step reasoning problem with step verification.
  - *Expected Outcome:* Low step score triggers rollout resample; high score bypasses reflection.
- **TC-T3-05: Adaptive Reflection Engine + OpenAI API Server Endpoint**
  - *Description:* Expose full adaptive reflection reasoning engine via `run_server.py` `/v1/chat/completions`.
  - *Inputs:* HTTP POST request to API server requesting reasoning answer.
  - *Expected Outcome:* Server executes adaptive reflection pipeline and streams final output back via SSE.
- **TC-T3-06: End-to-End Benchmark Harness + Full Engine Validation**
  - *Description:* Execute `benchmark.py` against complete pipeline to profile latency, accuracy gain, and token savings.
  - *Inputs:* 20 benchmark reasoning problems.
  - *Expected Outcome:* Validates $\ge 3.0\times$ GEMM speedup, $+15\%$ accuracy, and $\ge 30\%$ token savings.
- **TC-T3-07: Sequential Model Swapping Under Active API Server Request**
  - *Description:* Verify memory purging and weight reloading during active API HTTP request processing.
  - *Inputs:* API request trigger while monitoring system memory footprint.
  - *Expected Outcome:* Peak memory stays $<4.5$GB throughout reasoner-to-verifier transition.
- **TC-T3-08: Batched Parallel Decode ($N=8$) + INT4 Metal Kernel Integration**
  - *Description:* Run batched decode generator using repacked INT4 weight superblocks on Metal GPU (`simdgroup_matrix`).
  - *Inputs:* Metal compute pipeline with 128-byte superblock layout.
  - *Expected Outcome:* Executes GEMM decoding with full hardware SIMD group saturation.
- **TC-T3-09: OpenAI SSE Streaming + Real-Time Adaptive Reflection Interleaving**
  - *Description:* Stream token deltas to API client while internal engine performs adaptive reflection.
  - *Inputs:* API request with `"stream": true` on hard problem.
  - *Expected Outcome:* SSE stream sends initial tokens, handles internal reflection pause, then resumes clean stream.
- **TC-T3-10: High-Batch Parallel Decode ($N=16$) + List-Wise Verification Memory Boundary Safety**
  - *Description:* Execute $N=16$ parallel rollouts followed by list-wise verification on 8GB RAM profile.
  - *Inputs:* `n_parallel_rollouts=16`.
  - *Expected Outcome:* KV-cache and verifier candidate memory remain within 4.5GB app RAM limit.

---

### Tier 4: Real-World Application Scenarios (7 Test Cases)

- **TC-T4-01: GSM8K Multi-Step Math Reasoning Problem Solving**
  - *Description:* Solve standard GSM8K word problems requiring multi-step arithmetic reasoning.
  - *Inputs:* `"Janet buys 6 bags of chips for $2 each and 3 sodas for $1.50 each. How much change does she get from $20?"`
  - *Expected Outcome:* Complete `<think>` reasoning trace deriving correct final answer `\boxed{$9.50}`.
- **TC-T4-02: Symbolic Calculus & Integral Derivation Trajectory Evaluation**
  - *Description:* Evaluate symbolic mathematical integration.
  - *Inputs:* `"Compute the indefinite integral of x * cos(x) dx."`
  - *Expected Outcome:* Correct integration by parts derivation resulting in `\boxed{x\sin(x) + \cos(x) + C}`.
- **TC-T4-03: Modular Arithmetic & Number Theory Proof Verification**
  - *Description:* Verify mathematical proof problem on target engine.
  - *Inputs:* `"Prove that 2^n > n^2 for all integers n >= 5."`
  - *Expected Outcome:* Induction proof steps validated by list-wise verifier; candidate trace selected cleanly.
- **TC-T4-04: Multi-Step Reasoning Trace with Self-Correction (`<think>`)**
  - *Description:* Verify engine generates reasoning trace with intermediate self-correction inside `<think>` block.
  - *Inputs:* Complex logic puzzle prompt.
  - *Expected Outcome:* Trajectory contains `<think>` block identifying initial error and correcting it before final `\boxed{}` answer.
- **TC-T4-05: Zero-Shot Best-of-N Rollout Accuracy Gain vs Greedy Single Pass**
  - *Description:* Compare 20 math problem solutions between $N=8$ Best-of-N engine vs $N=1$ greedy single pass.
  - *Inputs:* 20 benchmark math questions.
  - *Expected Outcome:* Best-of-N achieves $\ge 15\%$ higher answer accuracy rate than single pass.
- **TC-T4-06: Sustained API Server Load Under Concurrent Math Completion Requests**
  - *Description:* Run 10-minute sustained load test with 5 concurrent clients sending math reasoning requests.
  - *Inputs:* Continuous HTTP POST request stream to `localhost:8080`.
  - *Expected Outcome:* Zero server crashes, zero memory leaks, average request latency stable.
- **TC-T4-07: Low-Memory Footprint Verification on Target iPhone Entitlement Profile**
  - *Description:* Run full reasoning + verifier loop while monitoring memory entitlement profile (4.5GB limit).
  - *Inputs:* Continuous heavy inference execution.
  - *Expected Outcome:* Resident memory RSS does not exceed 4,096 MB at any point during run.

---

## 4. Test Infrastructure Architecture (`tests/e2e/`)

The E2E test harness is designed under `/Users/MohssineChazi2/moat/tests/e2e/` with strict modularity, high mock/hardware duality, and automated assertion capabilities:

```
/Users/MohssineChazi2/moat/
├── TEST_INFRA.md                   # Full test infrastructure architectural guide
└── tests/
    └── e2e/
        ├── __init__.py
        ├── conftest.py             # Global pytest fixtures, synthetic weights, memory trackers
        ├── test_tier1_features.py  # Tier 1 Feature Coverage tests (35 test cases)
        ├── test_tier2_boundaries.py# Tier 2 Boundary & Corner Case tests (35 test cases)
        ├── test_tier3_combinations.py # Tier 3 Cross-Feature Interaction tests (10 test cases)
        ├── test_tier4_scenarios.py # Tier 4 Real-World Application Scenario tests (7 test cases)
        └── run_e2e_tests.py        # Master CLI test runner & reporter
```

### Module Responsibilities:

1. **`conftest.py` (Pytest Configuration & Fixtures):**
   - `synthetic_fp16_weights`: Generates deterministic random FP16 matrices for quantization testing.
   - `mock_engine_config`: Provides standardized `engine_config.yaml` parameters ($N=8$, group size $32$, $\tau=0.75$).
   - `softmax_lut_instance`: Returns initialized `SafeSoftmaxLUT` instance.
   - `mock_api_server`: Fixture spinning up local HTTP server on dynamic test port for API client tests.
   - `memory_tracker`: Context manager monitoring peak resident set size (RSS) during test runs.

2. **`test_tier1_features.py`:**
   - Implements isolated unit/integration tests for F1 through F7 (`test_tc_t1_f1_01` $\dots$ `test_tc_t1_f7_05`).

3. **`test_tier2_boundaries.py`:**
   - Implements extreme input, out-of-bounds, memory cap, and error handling tests (`test_tc_t2_f1_01` $\dots$ `test_tc_t2_f7_05`).

4. **`test_tier3_combinations.py`:**
   - Implements cross-module interaction tests (`test_tc_t3_01` $\dots$ `test_tc_t3_10`).

5. **`test_tier4_scenarios.py`:**
   - Implements GSM8K, calculus, proof, and API server load tests (`test_tc_t4_01` $\dots$ `test_tc_t4_07`).

6. **`run_e2e_tests.py` (Master Execution & CLI Harness):**
   - Command-line runner supporting flags: `--tier {1,2,3,4,all}`, `--feature {f1..f7}`, `--hardware {mock,metal}`, `--json-report report.json`.
   - Executes tests, aggregates assertions, formats colored console tables, and outputs JSON metrics.

---

## 5. Verification Protocol & Invalidation Criteria

To ensure independent verifiability by downstream agents or human reviewers, the test suite defines clear **Verification Methods** and **Invalidation Conditions**:

### Verification Method:
Execute master test runner with full coverage on local machine:
```bash
python3 tests/e2e/run_e2e_tests.py --tier all --hardware mock
```

### Key Performance Invalidation Conditions:
1. **GEMM Speedup Failure:** If batched decode latency ($N=8$) divided by sequential single decode latency is $< 3.0\times$.
2. **Softmax Acceleration Failure:** If LUT softmax execution speedup over standard dynamic softmax is $< 1.5\times$.
3. **Memory Footprint Exceeded:** If peak app memory allocation exceeds $4,500$ MB ($4.5$ GB).
4. **Token Savings Violation:** If adaptive reflection token consumption savings vs always-reflect baseline is $< 30\%$.
5. **Accuracy Degradation:** If Best-of-N selection accuracy gain on math benchmark set is $< +15\%$ over greedy single pass.
6. **API Standard Violation:** If POST `/v1/chat/completions` payload response format deviates from standard OpenAI JSON schema or breaks SSE streaming protocol.

---
*Report completed by `explorer_e2e_1`. Saved to `/Users/MohssineChazi2/moat/.agents/explorer_e2e_1/analysis.md`.*
