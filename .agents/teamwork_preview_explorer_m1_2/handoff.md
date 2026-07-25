# Technical Exploration Handoff Report: INT4 Quantization, Superblock Repacking & LUT Dequantization Engine

**Agent**: Explorer 2 (Milestone 1)  
**Working Directory**: `/Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_2`  
**Date**: 2026-07-25T05:21:00Z  

---

## 1. Observation

### 1.1 Source Documents & Codebase State
- **Scope File**: `/Users/MohssineChazi2/moat/.agents/sub_orch_m1/SCOPE.md`
  - Defines Milestone 1 targets: `config/engine_config.yaml`, `src/__init__.py`, `src/dequant.py`, `tests/test_quantization.py`.
  - Lines 16-22 specify interface contracts:
    - Group size $g = 32$, superblock size $= 256$, alignment $= 128 \text{ bytes}$.
    - `repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray`
    - `lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray`
    - Memory calculations for 1.5B–3B models fitting in ~4.5GB RAM.
- **PRD Document**: `/Users/MohssineChazi2/moat/antigravity_prd.md`
  - Lines 84-108 contain shell code for `repack_weights_to_superblock` and `lut_dequantize_fp16`.
- **Architectural Planning Documents**:
  - `/Users/MohssineChazi2/moat/planning/01_hardware_architecture.md`: Details 8GB UMA budget (~4.5GB app RAM limit) and GPU/ANE 128-byte SIMD cache-line alignment.
  - `/Users/MohssineChazi2/moat/planning/02_quantization_and_lut.md`: Details symmetric INT4 group quantization formula, 144-byte superblock packing structure, and 16-element LUT gather indexing.
- **Codebase Directory Status**:
  - `src/` and `tests/` do not exist yet in `/Users/MohssineChazi2/moat/`. They are to be created during Milestone 1 implementation.

---

## 2. Logic Chain

### 2.1 Algorithmic & Numerical Specification for INT4 Quantization

#### 2.1.1 Fine-Grained Symmetric Group Quantization Math
For a tensor of weights $W \in \mathbb{R}^N$ partitioned into groups $G_j$ of size $g = 32$ elements:
1. **Group Max Absolute Value**:
   $$\alpha_j = \max_{i \in [0, 31]} |w_{j, i}|$$
2. **Dynamic FP16 Scale Factor ($S_j$)**:
   $$S_j = \begin{cases} \frac{\alpha_j}{7.0} & \text{if } \alpha_j > 10^{-8} \\ 1.0 & \text{if } \alpha_j \le 10^{-8} \end{cases}$$
   - *Scale precision*: Stored as `np.float16` (2 bytes).
3. **Quantized Signed INT4 Values ($q_{j, i} \in [-8, 7]$)**:
   $$q_{j, i} = \text{clamp}\left( \text{round}\left( \frac{w_{j, i}}{S_j} \right), -8, 7 \right)$$
   - Data type: `np.int8` (values restricted to range $[-8, 7]$).
4. **Unsigned LUT Index Representation ($u_{j, i} \in [0, 15]$)**:
   $$u_{j, i} = q_{j, i} + 8 \in \{0, 1, 2, \dots, 15\}$$
   - Data type: `np.uint8` (stored packed as 4-bit nibbles or unpacked).

#### 2.1.2 The NumPy Indexing Pitfall (Critical Discovery)
In standard Python/NumPy, array indexing with negative integers performs wrapped indexing from the tail (e.g., `lut[-1]` returns `lut[15]`, NOT the value for $-1 \times S_j$).
- **Naive code shell in PRD**: `return lut[q_weights]` with signed `q_weights` in $[-8, 7]$ causes silent data corruption!
- **Required Fix**: `q_weights` MUST be formatted as unsigned `np.uint8` indices in $[0, 15]$ (where index 0 maps to $-8 \times S_j$, index 8 maps to $0.0$, and index 15 maps to $+7 \times S_j$), or dequantization must explicitly use `lut[q_weights + 8]`.

---

### 2.2 Superblock Repacking & SIMD Alignment (128-Byte)

#### 2.2.1 Superblock Structure & Memory Layout
A **Superblock** coalesces 8 fine-grained quantization groups of 32 elements into 256 weight elements:
- **Element Count**: 256 weights.
- **Scale Factors**: 8 FP16 values ($8 \times 2 = 16 \text{ bytes}$).
- **Weight Payload**:
  - *Unpacked*: 256 bytes (`np.int8` or `np.uint8`).
  - *Packed*: 128 bytes (`np.uint8`, 2 nibbles per byte: low nibble = even index, high nibble = odd index).
  - $256 \times 4 \text{ bits} = 1024 \text{ bits} = 128 \text{ bytes}$ — perfectly matches 1 SIMD cache line / 128-byte GPU vector register boundary.

```
Superblock Memory Layout (Total Packed: 144 Bytes):
┌──────────────────────────────────────────┬────────────────────────────────────────┐
│ Header (16 Bytes)                        │ Packed Payload (128 Bytes)             │
│ [S_0, S_1, S_2, S_3, S_4, S_5, S_6, S_7] │ [128 Bytes: 256 x INT4 Nibble Pairs]   │
└──────────────────────────────────────────┴────────────────────────────────────────┘
```

#### 2.2.2 Exact Function Signatures & Array Shapes

```python
import numpy as np

def quantize_to_int4_groups(weights: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantizes float weights into INT4 indices and FP16 scale factors per group.
    
    Inputs:
        weights: np.ndarray of shape (N,) or (M, K), dtype np.float32 or np.float16.
                 Length MUST be a multiple of group_size (32).
        group_size: int, default 32.
        
    Returns:
        q_indices: np.ndarray of shape (N_groups, group_size), dtype np.uint8 (values in 0..15).
        scales: np.ndarray of shape (N_groups,), dtype np.float16.
    """

def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Coalesces 8 fine-grained INT4 quantization groups of size 32 into a
    single aligned super-block of size 256 to ensure wide vector register access.
    
    Inputs:
        weights_int4: np.ndarray of shape (N,), dtype np.int8 or np.uint8.
                      N MUST be a multiple of 256.
        group_size: int, default 32.
        
    Returns:
        repacked: np.ndarray of shape (num_superblocks, 8, group_size), dtype np.uint8 or np.int8.
                  Guaranteed contiguous memory layout aligned to 128-byte SIMD boundaries.
    """
```

#### 2.2.3 Memory Alignment Verification
To ensure 128-byte alignment for Apple Metal / GPU SIMD access:
```python
def check_128b_alignment(arr: np.ndarray) -> bool:
    """Verifies that the array's underlying data memory address is aligned to 128 bytes."""
    return arr.ctypes.data % 128 == 0

def align_array_128b(arr: np.ndarray) -> np.ndarray:
    """Ensures 128-byte memory alignment for the given numpy array."""
    if check_128b_alignment(arr):
        return arr
    # Allocate aligned buffer if unaligned
    buf = np.empty(arr.nbytes + 128, dtype=np.uint8)
    offset = (128 - (buf.ctypes.data % 128)) % 128
    aligned_view = buf[offset:offset + arr.nbytes].view(dtype=arr.dtype).reshape(arr.shape)
    np.copyto(aligned_view, arr)
    return aligned_view
```

---

### 2.3 LUT Gather Dequantization to FP16

#### 2.3.1 Lookup Table Construction & Gathering Math
For group $j$ with scale factor $S_j \in \text{FP16}$, construct a 16-element FP16 lookup table:
$$\text{LUT}_j[u] = (u - 8) \times S_j \quad \text{for } u \in \{0, 1, \dots, 15\}$$

- **Global LUT Shape**: `(N_groups, 16)` of dtype `np.float16`.
- **Dequantization Signature**:
```python
def build_dequant_lut(scales: np.ndarray) -> np.ndarray:
    """
    Precomputes 16-element FP16 lookup tables for each quantization group.
    
    Inputs:
        scales: np.ndarray of shape (N_groups,), dtype np.float16.
        
    Returns:
        lut: np.ndarray of shape (N_groups, 16), dtype np.float16.
    """
    # Base multiplier vector [-8, -7, ..., 7]
    base_levels = np.arange(-8, 8, dtype=np.float16)  # shape (16,)
    # Broadcast multiply: (N_groups, 1) * (16,) -> (N_groups, 16)
    return scales[:, None] * base_levels[None, :]

def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Fast table-lookup dequantization mapping INT4 back to FP16.
    Avoids expensive unpacking instruction cycles in the compute loop.
    
    Inputs:
        q_weights: np.ndarray of shape (N_groups, group_size) or (num_superblocks, 8, group_size),
                   dtype np.uint8 (unsigned indices 0..15).
        lut: np.ndarray of shape (N_groups, 16) or (num_superblocks, 8, 16), dtype np.float16.
        
    Returns:
        dequantized: np.ndarray of shape identical to q_weights, dtype np.float16.
    """
    # Perform vector gather along the 16-level axis
    if q_weights.ndim == 2 and lut.ndim == 2:
        return np.take_along_axis(lut, q_weights.astype(np.intp), axis=1)
    elif q_weights.ndim == 3 and lut.ndim == 3:
        return np.take_along_axis(lut, q_weights.astype(np.intp), axis=2)
    else:
        # Flattened indexing fallback
        num_groups = lut.shape[0] if lut.ndim == 2 else lut.shape[0] * lut.shape[1]
        flat_lut = lut.reshape(num_groups, 16)
        flat_q = q_weights.reshape(num_groups, -1)
        return flat_lut[np.arange(num_groups)[:, None], flat_q].reshape(q_weights.shape)
```

---

### 2.4 Rigorous Memory Footprint Calculations (~4.5GB Budget)

#### 2.4.1 iPhone 15 Pro System Memory Allocation
- **Total System RAM**: 8,192 MB (8.0 GB).
- **iOS & Neural Engine Reserved**: ~2,500 MB to 3,500 MB.
- **Entitled App Memory Ceiling**: ~4,500 MB to 5,500 MB.
- **Target Engine Upper Ceiling**: $\le 4,500 \text{ MB}$ ($4.5 \text{ GB}$).

#### 2.4.2 Model Weight Memory Math (1.5B vs 3.0B Parameters)

| Metric / Layer Component | Qwen2.5-1.5B (1.54B Params) | Llama-3.2-3B / Qwen2.5-3B (3.09B Params) |
| :--- | :--- | :--- |
| **FP16 Base Weights** | $1.54\text{B} \times 2\text{B} = \mathbf{3,080\text{ MB}}$ | $3.09\text{B} \times 2\text{B} = \mathbf{6,180\text{ MB}}$ |
| **INT4 Payload (4 bits / param)** | $\frac{1.54\text{B} \times 4}{8} = \mathbf{770.0\text{ MB}}$ | $\frac{3.09\text{B} \times 4}{8} = \mathbf{1,545.0\text{ MB}}$ |
| **FP16 Scales ($g=32$)** | $\frac{1.54\text{B}}{32} \times 2\text{B} = \mathbf{96.25\text{ MB}}$ | $\frac{3.09\text{B}}{32} \times 2\text{B} = \mathbf{193.13\text{ MB}}$ |
| **Total INT4 Superblock Weights** | $\mathbf{866.25\text{ MB}}$ ($0.866\text{ GB}$) | $\mathbf{1,738.13\text{ MB}}$ ($1.738\text{ GB}$) |
| **Effective Bits per Parameter** | $4.50\text{ bits/param}$ | $4.50\text{ bits/param}$ |
| **Compression Ratio vs FP16** | **3.56x** | **3.56x** |

#### 2.4.3 Paged KV Cache Memory Math ($N=8$ Batched Rollouts)
- Model: Qwen2.5-1.5B ($L=28$ layers, $H_{KV}=2$ GQA heads, $d_k=128$).
- Key + Value elements per token across all 28 layers:
  $$2 \times L \times H_{KV} \times d_k = 2 \times 28 \times 2 \times 128 = 14,336 \text{ elements/token} = 28.67 \text{ KB/token (FP16)}$$
- Context Length $C = 2,048$ tokens per sequence (512 shared prompt + 1,536 generated):
  - Shared Prompt KV ($N=8$ share 512 tokens): $512 \times 28.67 \text{ KB} = \mathbf{14.68 \text{ MB}}$.
  - Generated KV ($8 \times 1,536$ tokens): $8 \times 1,536 \times 28.67 \text{ KB} = \mathbf{352.25 \text{ MB}}$.
  - Total Paged KV Cache Footprint: $\mathbf{366.93 \text{ MB}}$.

#### 2.4.4 Sequential Model Swapping & Peak Allocation Summary

```
Sequential Execution Timeline:
─────────────────────────────────────────────────────────────────────────────►
[ Phase 1: Batched Reasoner Rollouts (N=8) ]
- Reasoner INT4 Model Weights: 866.25 MB
- Paged KV Cache (N=8, 2048 tokens): 366.93 MB
- Compute Scratchpad & Activation Buffers: 200.00 MB
- Phase 1 Memory Peak: 1,433.18 MB (1.43 GB)

[ Phase 2: Weight Purge & Swap ]
- Free Reasoner Weights: -866.25 MB
- Retain candidate text tokens in RAM: ~0.05 MB

[ Phase 3: List-Wise Verifier Scoring ]
- Verifier INT4 Model Weights: 866.25 MB
- Candidate Evaluation Buffer: 50.00 MB
- Phase 3 Memory Peak: 1,116.25 MB (1.12 GB)

Max Engine RAM Peak across all phases: 1,433.18 MB (1.43 GB)
Available Safety Margin under 4.5 GB limit: 4,500 MB - 1,433.18 MB = 3,066.82 MB (3.07 GB headroom!)
```

Even for a 3.0B model, peak RAM is $1,738.13 + 366.93 + 200.00 = 2,305.06 \text{ MB}$ ($2.31 \text{ GB}$), which is still under 52% of the 4.5GB ceiling!

---

### 2.5 Unit Testing & Edge Cases Specification

The test suite `tests/test_quantization.py` must cover the following test cases:

1. **Repacking Shape & Alignment Test (`test_superblock_repacking_shape_and_alignment`)**:
   - Verify array lengths not divisible by 256 raise `AssertionError` or `ValueError`.
   - Verify input `(256 * K,)` produces output `(K, 8, 32)`.
   - Verify memory contiguity and 128-byte alignment via `check_128b_alignment`.
2. **LUT Dequantization Accuracy Test (`test_lut_dequantize_fp16_accuracy`)**:
   - Verify `lut_dequantize_fp16` on known INT4 indices and LUT values returns exact FP16 matches.
   - Verify output dtype is strictly `np.float16`.
   - Compare reconstruction Mean Squared Error (MSE) against original FP16 weights: target $\text{MSE} < 10^{-3}$ for standard normal weights.
3. **Boundary & Edge Case Handling (`test_quantization_edge_cases`)**:
   - *Zero Scale*: All weights in a group are `0.0`. Verify scale is handled safely without `NaN` / `Inf` or division by zero.
   - *Extreme Range / Clamping*: Weights exceeding $[-8 \cdot S, 7 \cdot S]$. Verify indices are clamped strictly to $[0, 15]$ (or $[-8, 7]$).
   - *Unsigned Indexing*: Verify negative indices do not cause wrapped NumPy reverse lookup bugs.
4. **Memory Footprint Bounds Test (`test_memory_footprint_bounds`)**:
   - Calculate theoretical memory size of a 1.5B and 3B model under INT4 group 32 quantization.
   - Assert total model weight footprint is $< 900 \text{ MB}$ for 1.5B model and $< 1800 \text{ MB}$ for 3B model.

---

## 3. Caveats

- **Read-Only Exploration**: Explorer 2 operated strictly in read-only analysis mode. No project code files in `src/` or `tests/` were created or modified.
- **NumPy CPU Golden Reference vs Metal Kernel**: This report specifies NumPy algorithms and data structures. In downstream production milestones (Metal/iOS deployment), `simdgroup_matrix` compute shaders will ingest the repacked 144-byte superblocks. The NumPy logic serves as the ground-truth specification for unit tests.
- **No Further Caveats**.

---

## 4. Conclusion

1. **Exact Data Types & Shapes**:
   - Weights INT4: `np.uint8` in $[0, 15]$ or `np.int8` in $[-8, 7]$. Unpacked group shape: `(N_groups, 32)`.
   - Superblock shape: `(num_superblocks, 8, 32)`, aligned to 128 bytes.
   - Group scales: `np.float16`, shape `(N_groups,)`.
   - LUT table: `np.float16`, shape `(N_groups, 16)`.
2. **Memory Footprint Assessment**:
   - 1.5B model INT4 weights require **866.25 MB**.
   - 3.0B model INT4 weights require **1,738.13 MB**.
   - Batched $N=8$ Paged KV Cache requires **366.93 MB**.
   - Peak RAM usage is **~1.43 GB** (for 1.5B) and **~2.31 GB** (for 3.0B), leaving $>2.1\text{ GB}$ of headroom within the 4.5GB iPhone 15 Pro RAM limit.
3. **Key Finding / Bug Prevention**:
   - Naive signed index table lookup `lut[q_weights]` in NumPy triggers reverse indexing bugs for negative values.
   - Solved by mapping INT4 signed indices $[-8, 7]$ to unsigned indices $[0, 15]$ via $u = q + 8$, ensuring direct, zero-overhead LUT indexing.

---

## 5. Verification Method

To verify the recommendations of this exploration report:
1. **Inspect Handoff Document**:
   Verify `/Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_2/handoff.md` exists and contains all 5 required protocol sections.
2. **Implementation Verification (Worker Phase)**:
   When `src/dequant.py` and `tests/test_quantization.py` are implemented by the worker:
   - Run unit test suite: `pytest tests/test_quantization.py -v`.
   - Confirm all alignment checks (`arr.ctypes.data % 128 == 0`), LUT accuracy checks, and memory bounds assertions pass cleanly.
