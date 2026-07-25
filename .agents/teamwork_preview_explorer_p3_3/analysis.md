# Phase 3 Architecture & Test Suite Specification Analysis

**Author:** Explorer 3 (Phase 3 Integration & End-to-End Test Suite Architect)  
**Date:** 2026-07-25  
**Target Subsystem:** Project Antigravity Phase 3 — Batched Rollout Coordinator & Model Weight Loader  
**Scope:** `antigravity-engine/tests/test_batch_generator.py` & `antigravity-engine/tests/test_model_loader.py`

---

## 1. Executive Summary & Context

Phase 3 of Project Antigravity scales single-sequence inference ($N=1$) to parallel candidate reasoning traces ($N=8, 16$) executing concurrently on Apple Silicon GPU hardware via Metal `simdgroup_matrix` intrinsics. 

To ensure mathematical precision, numerical stability, memory efficiency, and execution performance, R3 defines a comprehensive integration and end-to-end test suite (`test_batch_generator.py` and `test_model_loader.py`). This document details the architectural analysis, mathematical bounds, memory budget calculations, and test specifications required for Phase 3.

---

## 2. Review of Existing Micro-Unit Test Suites

| Test Module | Coverage & Focus | Key Verification Mechanisms | Relevance to Phase 3 R3 |
| :--- | :--- | :--- | :--- |
| `test_quantization.py` | INT4 symmetric group quantization, 144-byte super-block repacking, FP16 LUT dequantization | Strict error bounds ($\le S_G / 2 + \epsilon$), bitwise nibble packing validation, dead-zone sign preservation, LUT vs arithmetic dequant identity | Serves as baseline math for `test_model_loader.py` weight repacking and dequantization correctness. |
| `test_attention.py` | 32,768-entry exponential LUT (~64 KB), safe softmax max-subtraction | Softmax probability axioms (sum to 1.0, non-negative, $\le 1.0$), argmax agreement ($\ge 98\%$), extreme logit range stability ($[-1000, 1000]$) | Basis for `test_batch_generator.py` attention scoring & parallel sampling. |
| `test_batched_speed_metal.py` | Metal GPU MPS vs sequential GEMV speedup benchmarks | SIMD matrix tile saturation, execution timing via `torch.mps.synchronize()`, throughput scaling across $N \in [1, 32]$ | Reference timing & GPU hardware harness for Phase 3 rollout coordinator performance benchmark. |

---

## 3. Detailed Test Specification: `test_batch_generator.py`

`batch_generator.py` coordinates $N=8$ parallel candidate reasoning trajectories, managing a Paged KV-Cache and dispatching dense matrix activations through Metal SIMD GEMM kernels and safe LUT softmax.

### 3.1 Test Hierarchy & Method Specifications

#### Test 1: Single-Trace (N=1) vs Parallel Trace Decode (N=8) Mathematical Parity
* **Objective:** Verify that evaluating token $t$ in channel 0 of an $N=8$ batched rollout produces identical unnormalized logits and hidden states as a standalone $N=1$ decode pass.
* **Mathematical Parity Formulation:**
  $$\mathbf{z}_{N=1}^{(t)} = \text{Forward}(x_0^{(t)}, \text{KV}_0) \quad \text{vs} \quad \mathbf{Z}_{N=8}^{(t)}[0, :] = \text{BatchedForward}(\mathbf{X}^{(t)}, \text{KV}_{0..7})[0, :]$$
* **Assertion Threshold:**
  $$\max_{k} |\mathbf{z}_{N=1}^{(t)}[k] - \mathbf{Z}_{N=8}^{(t)}[0, k]| \le 10^{-2} \quad (\text{FP16 precision limit})$$
* **Implementation Pattern:**
  - Initialize identical model parameters and prompt prefix $P$.
  - Run 1 step with single-sequence decode ($N=1$).
  - Run 1 step with 8-sequence batched decode ($N=8$) where row 0 has identical inputs and history.
  - Assert `np.testing.assert_allclose(logits_n8[0], logits_n1[0], rtol=1e-2, atol=1e-2)`.

#### Test 2: Paged KV-Cache Memory Isolation Across Channels
* **Objective:** Ensure no cross-channel attention corruption occurs when candidate traces diverge.
* **Architectural Mechanics:**
  - All $N=8$ channels share physical memory blocks for prompt tokens $[0 \dots P-1]$.
  - Generated tokens ($t \ge P$) allocate independent physical blocks per channel via block tables.
* **Verification Protocol:**
  - Initialize KV-cache with shared prompt (32 tokens).
  - Write distinct synthetic Key/Value vectors to Channel $i$ and Channel $j$ ($i \neq j$) over 10 generation steps.
  - Direct Block Inspection: Verify block indices owned by Channel $i$ are disjoint from Channel $j$ ($BlockTable_i \cap BlockTable_j = \emptyset$ for $t \ge P$).
  - Attention Score Parity: Compute multi-head attention output for Channel $i$ using the shared Paged KV-Cache versus an isolated single-channel reference cache. Assert bitwise identical output (`atol=1e-5`).

#### Test 3: Numerical Stability & Zero Memory Leaks (100+ Generation Steps)
* **Objective:** Guarantee zero NaN, zero Inf, and zero memory accumulation across long generation runs.
* **Stress Protocol:**
  - Execute a continuous 100-step generation loop with $N=8$ active channels.
  - Per-Step Sanity Assertion:
    - `assert not np.isnan(logits).any()`, `assert not np.isinf(logits).any()`
    - `assert not np.isnan(kv_cache.k_pool).any()`, `assert not np.isnan(kv_cache.v_pool).any()`
  - Memory Footprint Tracking:
    - Record initial GPU/host memory allocation $M_{\text{start}}$ via PyTorch MPS / system memory counters.
    - Run 100 generation cycles (allocating and freeing trace instances).
    - Trigger garbage collection and KV-cache block table resets.
    - Measure final memory allocation $M_{\text{end}}$.
    - Assert $(M_{\text{end}} - M_{\text{start}}) \le 0.1 \text{ MB}$.

#### Test 4: Sampling Diversity & Coherence under Temperature $T > 0$
* **Objective:** Confirm all $N=8$ reasoning channels produce coherent, non-identical outputs when $T > 0$.
* **Sampling Setup:** Temperature $T = 0.7$, Top-$p = 0.95$.
* **Diversity Assertion:**
  - Generate 20 tokens across all 8 parallel channels simultaneously.
  - Collect generated sequences $S_0, S_1, \dots, S_7$.
  - Pairwise Uniqueness: Assert `len(set(tuple(s) for s in sequences)) == 8` (no duplicate reasoning paths).
* **Coherence Assertion:**
  - Every token ID $s_{i, t} \in [0, V-1]$.
  - For every sampled token $s_{i, t}$, verify its pre-sampled logit $z_{i, t}$ satisfied $P(s_{i, t}) > 0$ in the top-$p$ truncated distribution.

#### Test 5: Apple Silicon GPU Performance Benchmark (N=8, 50 Steps in $\le 1.0$ s)
* **Objective:** Validate Phase 3 latency requirement ($\le 1.0 \text{ s}$ for 50 decode steps with $N=8$).
* **Execution Harness:**
  - Use PyTorch MPS or native Metal compute runner.
  - Warm up GPU pipeline for 5 steps.
  - Benchmark loop:
    ```python
    torch.mps.synchronize()
    start_time = time.perf_counter()
    for step in range(50):
        coordinator.step(batched_tokens)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start_time
    ```
  - Assert `elapsed <= 1.0` seconds (equivalent to $\le 20 \text{ ms}$ per 8-trace step, or $\ge 400 \text{ tok/s}$ aggregate throughput).

#### Test 6: Paged KV-Cache Memory Overhead Budget ($\le 128 \text{ MB}$ / trace at sequence length 2048)
* **Mathematical Proof:**
  - Architecture parameters (1.5B model): $L = 28$ layers, $n_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 64$ ($d_{\text{kv}} = 512$), dtype = FP16 (2 bytes).
  - Bytes per token per trace:
    $$\text{Bytes/token} = 2 (\text{K and V}) \times 28 \text{ layers} \times 512 \text{ dim} \times 2 \text{ bytes} = 57,344 \text{ bytes/token} \approx 56 \text{ KB/token}$$
  - Total KV data footprint for sequence length 2048:
    $$\text{Footprint}_{2048} = 2048 \times 57,344 \text{ bytes} = 117,440,512 \text{ bytes} \approx 112.0 \text{ MB}$$
  - Block table metadata overhead: $< 0.5 \text{ MB}$.
  - Total per-trace footprint: $\approx 112.5 \text{ MB} \le 128 \text{ MB}$.
* **Assertion:** `assert kv_cache.get_allocated_bytes(channel_id=0) <= 128 * 1024 * 1024`.

---

## 4. Detailed Test Specification: `test_model_loader.py`

`model_loader.py` parses open-weight model files (GGUF / Safetensors), quantizes/repacks weight matrices into 144-byte super-blocks, and enforces memory budget constraints.

### 4.1 Test Hierarchy & Method Specifications

#### Test 1: GGUF / Safetensors Tensor Loading and Repacking Accuracy
* **Objective:** Ensure model tensor parsing and repacking into super-blocks preserves numerical weight values.
* **Verification Protocol:**
  - Mock GGUF / Safetensors tensor buffers containing known FP16 weight matrices ($[K, M]$).
  - Invoke `model_loader.load_and_repack_tensor(mock_tensor)`.
  - Unpack super-blocks via `unpack_superblock` and dequantize via `lut_dequantize`.
  - Calculate per-element reconstruction error $e_i = |w_i - \hat{w}_i|$.
  - Assert max error $\le S_G / 2 + 10^{-3}$ and mean absolute error (MAE) $< 0.01$.

#### Test 2: Super-Block 144-Byte Binary Structure Validation
* **Objective:** Validate the exact binary memory alignment of packed super-block buffers.
* **Binary Structure Rules:**
  - Total size per super-block: exactly 144 bytes.
  - Header: 8 FP16 floats (16 bytes, offset 0..15).
  - Payload: 128 uint8 bytes (128 bytes, offset 16..143) containing 256 packed INT4 nibble pairs.
* **Assertion Suite:**
  - `assert buffer.nbytes % 144 == 0`
  - Extract header: `scales = np.frombuffer(buffer[:16], dtype=np.float16)`. Assert `len(scales) == 8` and `np.all(scales > 0)`.
  - Extract payload: `nibbles = np.frombuffer(buffer[16:144], dtype=np.uint8)`. Assert `len(nibbles) == 128`.
  - Unpack nibbles and verify all INT4 values fall in $[-8, 7]$.

#### Test 3: 1.5B Parameters Memory Footprint Calculation & Budget Assertion ($\le 2.5 \text{ GB}$)
* **Mathematical Footprint Proof:**
  - Parameter count: $N_{\text{params}} = 1,500,000,000$ (1.5B).
  - Super-block compression efficiency: 256 parameters stored in 144 bytes.
  - Compression ratio:
    $$\text{Bytes/param} = \frac{144 \text{ bytes}}{256 \text{ params}} = 0.5625 \text{ bytes/param} \quad (4.5 \text{ bits/param})$$
  - Quantized Weight Memory Footprint:
    $$\text{Memory}_{\text{weights}} = 1.5 \times 10^9 \times 0.5625 \text{ bytes} = 843,750,000 \text{ bytes} \approx 843.75 \text{ MB} \approx 0.804 \text{ GB}$$
  - Non-quantized FP16 parameters (Embeddings, LayerNorms, LM Head): $\approx 150 \text{ MB}$.
  - Total Loaded Model Memory: $\approx 0.95 \text{ GB} \le 2.5 \text{ GB}$ (Budget Margin: $> 1.5 \text{ GB}$ headroom).
* **Assertion:** `assert loader.calculate_total_memory_bytes(config_1_5b) <= 2.5 * 1024 * 1024 * 1024`.

---

## 5. Integration with Discovery & Test Execution

All test modules will be located in `/Users/MohssineChazi2/moat/antigravity-engine/tests/` and discoverable via standard unittest execution:

```bash
python3 -m unittest discover -s /Users/MohssineChazi2/moat/antigravity-engine/tests -p "test_*.py" -v
```

### Unittest Integration Guidelines:
1. Every test file inherits from `unittest.TestCase`.
2. Hardware-specific benchmarks (Metal GPU) use conditional skip decorators (`@unittest.skipUnless(HAS_TORCH_MPS, "PyTorch MPS required")`).
3. Path configuration automatically prepends `../src` to `sys.path`.
4. Zero external side effects or uncleaned state across test runs.
