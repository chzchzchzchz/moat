# Architectural Document 01: Hardware Acceleration & Memory Layout

## 1. Executive Summary & Hardware Context

Project Antigravity targets **iPhone 15 Pro and newer (A17 Pro / A18 Pro)** devices featuring:
- **Unified Memory Architecture (UMA):** 8GB unified memory shared between CPU, GPU, and Apple Neural Engine (ANE).
- **Available App RAM:** ~4.5GB to 5.5GB maximum (with `com.apple.developer.kernel.increased-memory-limit` entitlement).
- **Compute Accelerators:**
  - **16-Core ANE:** Rated at 35 INT8 TOPS (~19 FP16 TFLOPS). Best suited for static shape operations.
  - **Apple Metal GPU:** 6-core (A17 Pro) / 6-core (A18 Pro) with SIMD execution units and hardware SIMD-group matrix multiply instructions (`simdgroup_matrix`).
  - **CPU AMX / SME:** Apple Matrix Coprocessor / Scalable Matrix Extension blocks.

---

## 2. The GEMV Bottleneck on Mobile Hardware

### 2.1 Why Autoregressive Decode Degenerates into GEMV
During standard single-token generation:
- Batch size $N = 1$.
- Input shape: $[1, K]$ (activation vector).
- Weight matrix shape: $[K, M]$.
- Operation: Vector-Matrix multiplication (**GEMV**).

### 2.2 Arithmetic Intensity Breakdown
$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Bytes Transferred}}$$

For a 1.5B INT4 model layer ($K = M = 2048$):
- **GEMV (N=1):** $\text{FLOPs} = 2 \times 2048 \times 2048 \approx 8.39 \times 10^6$.
- **Bytes transferred (INT4 weights):** $\approx 2.1 \text{ MB}$.
- **Arithmetic Intensity:** $\approx 4 \text{ FLOPs/byte}$.

The Apple Metal GPU / ANE ridge point is $\approx 25 \text{ FLOPs/byte}$. Therefore, **GEMV operates at less than 16% of hardware compute capacity**, spending 84%+ of cycle time stalled on memory fetch!

---

## 3. The Breakthrough: Batched Parallel Decode (GEMV → GEMM)

### 3.1 Batching Candidate Trajectories
By executing $N = 8$ candidate reasoning traces concurrently:
- Input shape changes from $[1, K]$ to $[8, K]$ (matrix).
- Weight matrix shape remains $[K, M]$.
- Operation converts from GEMV to **GEMM** (General Matrix-Matrix Multiplication).

### 3.2 Compute Math Comparison
- **Bytes transferred:** Identical ($2.1 \text{ MB}$ weights loaded once!).
- **FLOPs:** $8 \times 8.39 \times 10^6 \approx 6.71 \times 10^7$.
- **Arithmetic Intensity:** $\approx 32 \text{ FLOPs/byte}$ — crossing the hardware ridge point!

**Result:** The hardware processes 8 parallel reasoning rollouts in virtually the same wall-clock latency as a single rollout!

---

## 4. Hardware Dispatch Strategy (Metal vs. ANE)

| Architectural Component | Target Engine | Rationale |
| :--- | :--- | :--- |
| **Quantized Weight Matrix Math** | Metal GPU (`simdgroup_matrix` / MPS) | Dynamic sequence lengths, custom LUT gathering, zero copy via UMA |
| **Batched Attention KV-Cache** | Metal Compute Shaders | Paged KV cache, memory-mapped exponential LUT softmax |
| **Embedding & Prefill Phase** | CoreML / ANE (or Metal GPU) | Fixed shape prefill fits ANE static graph requirements |
| **Verifier Ranking** | Metal GPU (sequential model swap) | High throughput batch scoring of candidate traces |

---

## 5. Memory Management Strategy on 8GB iPhone

```
Total iPhone RAM: 8.0 GB
├── iOS System & Neural Engine Reserve: ~2.5 GB
└── App Available RAM: ~5.5 GB (with entitlement)
    ├── Reasoner Model (Qwen2.5-1.5B INT4): ~1.1 GB
    ├── Batched KV Cache (N=8, Context=2048): ~0.8 GB
    ├── Verifier Model (Skywork-1.5B PRM INT4): ~1.1 GB (Sequential Load/Swap)
    └── App Buffer & Metal Command Buffers: ~0.5 GB
```

### Sequential Model Swapping Protocol
To avoid triggering iOS `Jetsam` OOM termination:
1. **Phase A:** Load Reasoner Model into UMA $\rightarrow$ Run Batched Generation ($N=8$).
2. **Phase B:** Purge Reasoner weights from active Metal buffer (keep text rollouts in RAM).
3. **Phase C:** Load Verifier Model into UMA $\rightarrow$ Score and rank all 8 candidate traces.
4. **Phase D:** Select optimal trace and stream back to API response.
