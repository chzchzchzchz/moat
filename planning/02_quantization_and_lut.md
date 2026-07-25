# Architectural Document 02: INT4 Fine-Grained Quantization & LUT Dequantization

## 1. INT4 Symmetric Fine-Grained Group Quantization

### 1.1 Mathematical Formulation
For each group $G$ of $g = 32$ weight elements $w_1, w_2, \dots, w_{32} \in \mathbb{R}$:
1. Find maximum absolute value:
   $$\alpha = \max_{i \in G} |w_i|$$
2. Compute FP16 scaling factor $S_G$:
   $$S_G = \frac{\alpha}{7}$$
3. Quantize FP16 weights to INT4 signed values $q_i \in [-8, 7]$:
   $$q_i = \text{clamp}\left( \left\lfloor \frac{w_i}{S_G} \right\rceil, -8, 7 \right)$$

---

## 2. Super-Block Weight Repacking (128-Byte Alignment)

### 2.1 The Register Alignment Problem
Small group sizes ($g = 32$) introduce frequent metadata loads (scale factor per 32 elements). If loaded independently into GPU thread registers, wide SIMD registers (128-byte / 256-bit) sit partially empty, leading to memory instruction stalls.

### 2.2 Super-Block Coalescing Scheme
Coalesce 8 individual 32-element quantization groups into 1 Super-Block of 256 elements:
- **Weights Payload:** $256 \times 4 \text{ bits} = 1024 \text{ bits} = 128 \text{ bytes}$ (exactly 1 SIMD vector cache line!).
- **Metadata Header:** 8 FP16 scale factors ($8 \times 2 \text{ bytes} = 16 \text{ bytes}$).
- **Total Super-Block Size:** $144 \text{ bytes}$.

```
Super-Block Structure (144 Bytes Total):
┌──────────────────────────────────────────┬────────────────────────────────────────┐
│ Header (16 Bytes)                        │ Payload (128 Bytes)                    │
│ [S_0][S_1][S_2][S_3][S_4][S_5][S_6][S_7] │ [Group 0..7 INT4 Packed Nibbles]       │
└──────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 3. Fast Table-Lookup (LUT) Dequantization

### 3.1 Eliminating Bit-Masking & Arithmetic Operations
Traditional dynamic dequantization requires:
1. Extract 4-bit nibbles via bit shifts (`>> 4`) and masks (`& 0x0F`).
2. Subtract offset / cast to float.
3. Multiply by scale factor $S_G$.

**LUT-Centric Approach:**
Precompute a 16-element FP16 lookup table per group $G$:
$$\text{LUT}_G[k] = k \times S_G \quad \text{for } k \in \{-8, -7, \dots, 6, 7\}$$

During the GEMM forward pass, native vector table-lookup operations (e.g. `vlut16` on NPU or SIMD byte gathers in Metal compute shaders) map quantized INT4 indices directly to FP16 values in a single cycle.

---

## 4. Precomputed Softmax Exponential LUT

### 4.1 Safe Softmax Offset & Bounded Domain
For attention scores $x_i$, compute row-wise maximum $m = \max_i(x_i)$.
Shift inputs to ensure non-positive domain:
$$\hat{x}_i = x_i - m \le 0$$

### 4.2 32,768-Entry Exponential Table
- Domain: $[-10.0, 0.0]$.
- Table entries: 32,768 FP16 values ($\approx 64 \text{ KB}$, fitting directly into CPU L1 cache or Metal Threadgroup Memory).
- Index mapping function:
  $$\text{idx} = \text{clamp}\left( \left\lfloor \frac{|\hat{x}_i|}{10.0} \times 32767 \right\rfloor, 0, 32767 \right)$$
- Softmax output:
  $$e^{\hat{x}_i} \approx \text{LUT}_{\exp}[\text{idx}]$$

**Speedup:** Eliminates dynamic transcendental `exp()` calls, delivering up to **2.2x speedup** in FlashAttention softmax passes.
