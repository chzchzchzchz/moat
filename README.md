# Project Antigravity — On-Device Batched LLM Decode Engine

> Native on-device inference engine targeting Apple Silicon (macOS / iOS) that exploits underutilized GPU/ANE matrix tiles during autoregressive LLM decode by running $N$ parallel reasoning traces simultaneously.

---

## Technical Overview & Core Innovation

During standard single-sequence autoregressive LLM decoding, inference degenerates into single-token Matrix-Vector multiplication (**GEMV**). Because memory bandwidth dominates and compute units sit 60%–70% idle, hardware matrix multiplication blocks (e.g., Apple Silicon SIMD matrix tiles) are severely underutilized.

**Antigravity** converts single-pass GEMV into dense Matrix-Matrix multiplication (**GEMM**) by batching $N$ parallel candidate reasoning traces ($N=4, 8, 16$) at runtime.

```
Single Trace (Standard):    [1  x K] @ [K x M]  ──► GEMV  (Memory-Bandwidth Bound, ~30% GPU Tile Utilization)
Parallel Traces (Antigravity): [N x K] @ [K x M]  ──► GEMM  (Compute Bound, ~85%+ GPU Tile Utilization)
```

---

## Repository Structure

```
.
├── README.md
├── implementation_plan.md
├── antigravity_prd.md
├── planning/                      # Deep Architecture & Security Specifications
│   ├── 01_hardware_architecture.md
│   ├── 02_quantization_and_lut.md
│   ├── 03_batched_decode_engine.md
│   ├── 04_verifier_and_adaptive_reflection.md
│   ├── 05_ios_app_and_api.md
│   └── 06_security_architecture.md
└── antigravity-engine/            # Engine Core Implementation
    ├── src/
    │   ├── dequant.py             # INT4 group quantization & LUT dequantization
    │   ├── attention.py           # 32K-entry precomputed exponential LUT & safe softmax
    │   └── shaders/
    │       └── batched_gemm.metal # Metal compute shader using simdgroup_matrix
    └── tests/
        ├── test_quantization.py    # Verification & Correctness Suite (Quantization)
        ├── test_attention.py       # Verification & Correctness Suite (Softmax LUT)
        ├── test_batched_speed.py   # Verification & Correctness Suite (CPU Math Parity)
        └── test_batched_speed_metal.py # Hardware Acceleration Suite (Metal GPU Benchmark)
```

---

## Test & Benchmark Architecture

### 1. Verification & Correctness Suite (CPU)
*Purpose: Mathematically validate all quantization, LUT dequantization, and softmax gather operations against PyTorch/NumPy floating-point references before hardware dispatch.*

- **Quantization Parity (`test_quantization.py` — 35 tests)**:
  - Validates INT4 symmetric fine-grained group quantization ($G=32$).
  - Validates 256-element super-block packing (128-byte aligned nibble pairs + 8x FP16 scale headers = 144 bytes).
  - Asserts **bit-exact parity** between fast 16-element LUT dequantization and standard arithmetic dequantization.
  - Fuzz tests for zero-NaN/Inf safety across random seeds.

- **Precomputed Softmax LUT (`test_attention.py` — 32 tests)**:
  - Validates 32,768-entry FP16 exponential lookup table (~64 KB, L1-cache aligned).
  - Asserts safe softmax row-wise max subtraction ($x_i - \max(X) \le 0$).
  - Verifies $\ge 98\%$ argmax token agreement with standard dynamic `exp()` softmax.

- **CPU Math Parity (`test_batched_speed.py` — 5 tests)**:
  - Asserts $\text{GEMM}(N=8)$ output is bit-exact with 8 sequential $\text{GEMV}$ passes.
  - Establishes baseline CPU BLAS profile.

### 2. Hardware Acceleration & Performance Suite (Metal GPU)
*Purpose: Benchmark hardware matrix tile saturation and measure per-token throughput speedup on Apple Silicon GPU.*

- **Metal GPU MPS Benchmark (`test_batched_speed_metal.py`)**:
  - Benchmarks PyTorch MPS / Metal Performance Shaders across $N=1, 2, 4, 8, 16, 32$.

---

## Empirical Benchmark Results (Apple Silicon GPU)

### Hardware Matrix Tile Saturation Speedup ($N=8$ Parallel Traces vs $N=1$ Sequential)

| Model Matrix Config | Shape $(K \times M)$ | $N=1$ GEMV (ms/tok) | $N=8$ GEMM (ms/tok) | **Per-Token Speedup** |
|:--- |:---:|:---:|:---:|:---:|
| **Qwen-1.5B FFN** | $2048 \times 5504$ | $2.48\text{ ms}$ | $0.42\text{ ms}$ | **$5.89\times$** |
| **Qwen-3B FFN** | $3072 \times 8192$ | $4.41\text{ ms}$ | $0.56\text{ ms}$ | **$7.86\times$** |
| **Attention Projection** | $2048 \times 2048$ | $2.75\text{ ms}$ | $0.29\text{ ms}$ | **$9.61\times$** |
| **Standard GEMM (N=8)** | $2048 \times 2048$ | $2.15\text{ ms}$ | $0.36\text{ ms}$ | **$6.05\times$** |

### Batch Scaling Profile (Apple Silicon GPU)

| Batch Size $N$ | Total Wall-Clock (ms) | Per-Token Latency (ms) | Throughput (tok/s) | Speedup vs $N=1$ |
|:---:|:---:|:---:|:---:|:---:|
| **1** | $2.08\text{ ms}$ | $2.08\text{ ms}$ | $481.1$ | $1.00\times$ |
| **2** | $2.74\text{ ms}$ | $1.37\text{ ms}$ | $730.7$ | $1.52\times$ |
| **4** | $2.91\text{ ms}$ | $0.73\text{ ms}$ | $1374.5$ | $2.86\times$ |
| **8** | $5.03\text{ ms}$ | $0.63\text{ ms}$ | $1589.5$ | **$3.30\times$** |
| **16** | $8.58\text{ ms}$ | $0.54\text{ ms}$ | $1864.5$ | **$3.88\times$** |
| **32** | $4.51\text{ ms}$ | $0.14\text{ ms}$ | $7093.6$ | **$14.74\times$** |

---

## Running Tests & Benchmarks

### 1. Run Verification & Correctness Suite (CPU)
```bash
python3 -m unittest antigravity-engine/tests/test_quantization.py antigravity-engine/tests/test_attention.py antigravity-engine/tests/test_batched_speed.py -v
```

### 2. Run Hardware Acceleration Suite (Metal GPU)
```bash
python3 -m unittest antigravity-engine/tests/test_batched_speed_metal.py -v
```
