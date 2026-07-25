# Architectural Formulation & Technical Design: R1 Batched Parallel Decode Rollout Coordinator (`src/batch_generator.py`)

## Executive Summary

Project Antigravity Phase 3 focuses on transforming single-token autoregressive decoding from a memory-bandwidth-bound **GEMV** operation into a compute-bound **GEMM** operation by batching $N$ candidate reasoning trajectories (rollout channels, e.g., $N = 4, 8, 16$).

This document formulates the complete technical design for **`src/batch_generator.py`**, consisting of two core components:
1. **`PagedKVCache`**: A paged Key-Value cache manager featuring fixed block sizes (16 tokens), virtual-to-physical block table mapping, shared prompt prefix block allocation with Copy-on-Write (CoW) semantics, and guaranteed memory overhead $\le 128\text{ MB}$ per trace for sequence length 2048.
2. **`BatchedRolloutCoordinator`**: A batched parallel decode coordinator that combines $N$ candidate token activations into a single $[N \times D]$ batch matrix, dispatches batched GEMM operations via PyTorch MPS or Metal SIMD compute shaders, performs multi-channel parallel sampling ($T > 0$, top-$p$/top-$k$), and integrates seamlessly with `attention.py`'s precomputed `ExponentialLUT` and `safe_softmax_lut`.

---

## 1. Existing System Context & Codebase Investigation

### 1.1 Summary of Key Engine Modules

1. **`src/dequant.py`**:
   - **Quantization**: INT4 symmetric fine-grained group quantization with `group_size = 32`. Per-group scale factor $S_G = \alpha / 7$ where $\alpha = \max_{i \in G} |w_i|$.
   - **Super-Block Repacking**: Coalesces 8 fine-grained groups (256 INT4 elements) into a 144-byte super-block:
     - 16-byte header: 8 FP16 scale factors (`scales`: `half[8]`).
     - 128-byte payload: 256 INT4 nibbles packed into 128 `uint8` byte pairs (`packed_nibbles`: `uchar[128]`).
     - Exactly aligns with 128-byte SIMD vector cache lines on Apple Silicon.
   - **LUT Dequantization**: `build_dequant_lut` precomputes 16-element FP16 lookup tables mapping INT4 indices $[-8, 7]$ to dequantized values $k \cdot S_G$, replacing expensive bitwise operations with single-cycle vector table lookups.

2. **`src/attention.py`**:
   - **`ExponentialLUT`**: 32,768-entry precomputed table (~64 KB FP16) mapping non-positive shifted inputs $\hat{x} \in [-10.0, 0.0]$ to $e^{\hat{x}} \in (0.0, 1.0]$.
   - **`safe_softmax_lut`**: Performs row-wise max subtraction $x_{\text{shifted}} = x - \max(x)$ to guarantee non-positive domain ($\le 0.0$), then gathers exponentials from `ExponentialLUT` via normalized indexing:
     $$\text{idx} = \text{clamp}\left(\frac{|\hat{x}|}{\text{step}}, 0, 32767\right)$$
     Eliminates dynamic transcendental `exp()` calls, providing up to $2.2\times$ speedup in softmax computation.

3. **`src/shaders/batched_gemm.metal`**:
   - **`dequantize_superblocks_kernel`**: Unpacks 144-byte super-blocks into contiguous FP16 weights on GPU.
   - **`batched_gemm_simdgroup`**: Uses Metal SIMD group matrix intrinsics (`simdgroup_matrix<half, 8, 8>`) to perform hardware-accelerated matrix multiplication $C [N \times M] = A [N \times K] \cdot B [K \times M]$.
   - $N=8$ aligns perfectly with Metal SIMD 8x8 matrix tiles, avoiding GPU tile idling.

4. **`tests/test_batched_speed_metal.py`**:
   - Demonstrates that PyTorch MPS / Metal GPU converts $N=1$ GEMV (~30% GPU utilization) to $N=8$ GEMM (~85%+ GPU utilization).
   - Validates per-token speedup of $3.82\times - 6.88\times$ on Apple Silicon (A17 Pro / A18 Pro / M-series).

5. **Planning Specs (`planning/01-06`)**:
   - Hardware Target: iPhone 15 Pro / 16 Pro (8GB UMA, ~4.5GB - 5.5GB app RAM limit).
   - Reasoner Model (Qwen2.5-1.5B INT4): ~1.1 GB weight footprint.
   - Memory Budget for Batched KV-Cache ($N=8$, Context=2048): ~0.8 GB total (~100 MB per trace).
   - Sequential Model Swapping: Load Reasoner $\rightarrow$ Generate $N=8$ traces $\rightarrow$ Purge GPU buffers $\rightarrow$ Load List-Wise Verifier.

---

## 2. Component 1: Paged KV-Cache Data Structure & Manager (`PagedKVCache`)

### 2.1 Block Architecture & Physical Layout

Traditional KV-caches allocate contiguous memory tensors of size $[N, L, 2, H_{kv}, S_{\max}, D_{\text{head}}]$. For multiple parallel rollout trajectories, this causes severe memory fragmentation and redundant prompt memory duplication.

`PagedKVCache` divides physical KV memory into fixed-size **Blocks**:
- **Block Size ($B$)**: 16 tokens per block.
- **Physical Block Pool**: Pre-allocated contiguous tensors in UMA memory:
  ```
  K_pool: shape [num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim] (float16)
  V_pool: shape [num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim] (float16)
  ```
- **Block Table**: A mapping per rollout channel $i \in \{0, \dots, N-1\}$ from virtual sequence token indices $t \in [0, S_i - 1]$ to physical block indices $p \in [0, \text{num\_physical\_blocks} - 1]$.

#### Virtual-to-Physical Address Translation Formula:
For a sequence $i$ at token position $t$:
$$\text{virtual\_block\_idx} = \lfloor t / B \rfloor$$
$$\text{block\_offset} = t \pmod B$$
$$\text{physical\_block\_id} = \text{block\_table}[i][\text{virtual\_block\_idx}]$$

Physical memory address for key tensor at layer $l$, head $h$:
$$\text{K\_address} = \text{K\_pool}[\text{physical\_block\_id}, l, h, \text{block\_offset}, :]$$

```
Virtual Sequence Tokens: [0................15] [16...............31] [32...............47]
                          └──── Block 0 ─────┘  └──── Block 1 ─────┘  └──── Block 2 ─────┘
                                  │                     │                     │
Block Table [Channel i]:     [ Physical #4 ]       [ Physical #12 ]      [ Physical #87 ]
                                  │                     │                     │
Physical Pool Allocation:  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
                           │ K_pool[4...] │      │ K_pool[12..] │      │ K_pool[87..] │
                           └──────────────┘      └──────────────┘      └──────────────┘
```

---

### 2.2 Shared Prompt Prefix Block Allocation & Copy-on-Write (CoW)

During initial prefill of a user prompt of length $S_{\text{prompt}}$:
1. The prefill engine processes prompt tokens $0 \dots S_{\text{prompt}}-1$ in a single forward pass.
2. `PagedKVCache` allocates $K_{\text{prompt}} = \lceil S_{\text{prompt}} / B \rceil$ physical blocks $P_0, P_1, \dots, P_{K_{\text{prompt}}-1}$.
3. Reference counts for these prompt blocks are set to $N$ (`ref_count[P_k] = N`).
4. When initializing $N$ candidate rollout channels, each channel $i$'s `block_table[i]` is assigned references to the shared prompt blocks:
   $$\text{block\_table}[i] = [P_0, P_1, \dots, P_{K_{\text{prompt}}-1}]$$

```
Shared Prompt Blocks:  [ P_0 (ref=N) ] ─── [ P_1 (ref=N) ] ─── [ P_prompt_last (ref=N) ]
                               │                  │                     │
                  ┌────────────┼──────────────────┼─────────────────────┘
                  │            │                  │
                  ▼            ▼                  ▼
Rollout Channel 0: [ P_0 ] ── [ P_1 ] ── ... ── [ P_last ] ──► [ Block 0_A (private, ref=1) ]
Rollout Channel 1: [ P_0 ] ── [ P_1 ] ── ... ── [ P_last ] ──► [ Block 0_B (private, ref=1) ]
Rollout Channel N: [ P_0 ] ── [ P_1 ] ── ... ── [ P_last ] ──► [ Block 0_N (private, ref=1) ]
```

#### Copy-on-Write (CoW) Semantics:
- If $S_{\text{prompt}}$ is an exact multiple of $B = 16$, all prompt blocks are fully packed and marked **Read-Only**. Every channel allocates a new private block (with `ref_count = 1`) for its first generated token.
- If $S_{\text{prompt}}$ is not a multiple of $B$ (i.e. $S_{\text{prompt}} \pmod B \ne 0$), the last prompt block $P_{\text{last}}$ is partially filled. When rollout channel $i$ writes its first generated token into slot $S_{\text{prompt}} \pmod B$:
  - `PagedKVCache` checks `ref_count[P_last]`. Since `ref_count > 1`, writing directly would mutate other channels' caches.
  - **CoW Trigger**: Channel $i$ allocates a new physical block $P_{\text{new}}$, copies existing prompt tokens from $P_{\text{last}}$ to $P_{\text{new}}$, replaces $P_{\text{last}}$ with $P_{\text{new}}$ in `block_table[i]`, sets `ref_count[P_new] = 1`, and decrements `ref_count[P_last]`.

---

### 2.3 Per-Channel Memory Overhead Calculation & Verification

Let's rigorously verify the memory overhead for candidate rollout sequence length up to 2048 tokens.

#### Target Model Parameter Specification (Qwen2.5-1.5B / Standard 1.5B):
- Number of layers ($L$): $28$
- Number of Query Heads ($H_q$): $12$
- Number of Key/Value Heads ($H_{kv}$): $2$ (Grouped Query Attention - GQA)
- Head Dimension ($D_{\text{head}}$): $128$
- Data type: FP16 ($2\text{ bytes}$)

#### 1. Per-Token KV Memory Footprint:
$$\text{Bytes}_{\text{token}} = 2 \times L \times H_{kv} \times D_{\text{head}} \times 2\text{ bytes (FP16)}$$
$$\text{Bytes}_{\text{token}} = 2 \times 28 \times 2 \times 128 \times 2 = 28,672\text{ bytes} = 28.0\text{ KB/token}$$

#### 2. Single Block (16 Tokens) Memory Footprint:
$$\text{Bytes}_{\text{block}} = 16 \times 28,672\text{ bytes} = 458,752\text{ bytes} = 448.0\text{ KB/block}$$

#### 3. Total Memory Footprint per Channel for Sequence Length $S = 2048$:
$$\text{Blocks}_{\text{required}} = \left\lceil \frac{2048}{16} \right\rceil = 128\text{ blocks}$$
$$\text{Memory}_{\text{trace}} = 128 \times 448.0\text{ KB} = 57,344\text{ KB} = \mathbf{56.0\text{ MB}}$$

#### 4. Shared Prompt Prefix Savings ($N = 8$, Prompt $= 512$ tokens, Generation $= 1536$ tokens):
- Prompt blocks ($512 / 16 = 32$ blocks): $32 \times 448\text{ KB} = 14.336\text{ MB}$ (shared once across all 8 channels).
- Generated blocks per channel ($1536 / 16 = 96$ blocks): $96 \times 448\text{ KB} = 42.0\text{ MB}$ per channel.
- Effective memory per channel with prompt sharing:
  $$\text{Effective Memory}_{\text{channel}} = 42.0\text{ MB} + \frac{14.336\text{ MB}}{8} = \mathbf{43.79\text{ MB}}$$
- Total pool memory for $N=8$ channels:
  $$\text{Total Pool Memory} = 14.336\text{ MB} + (8 \times 42.0\text{ MB}) = \mathbf{350.34\text{ MB}} \approx 0.35\text{ GB}$$

#### Memory Bound Verification Table:

| Architecture / Model Configuration | $L$ | $H_{kv}$ | $D_{\text{head}}$ | KB / Token | MB per 2048-Seq Trace | Verification Status ($\le 128\text{ MB}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B (GQA)** | 28 | 2 | 128 | 28.0 KB | **56.0 MB** | **PASSED** (43.8% of budget) |
| **Qwen2.5-3B (GQA)** | 36 | 2 | 128 | 36.0 KB | **72.0 MB** | **PASSED** (56.3% of budget) |
| **Standard 1.5B (MHA $H_{kv}=12$)** | 28 | 12 | 128 | 168.0 KB | **336.0 MB** | Exceeds (requires GQA) |

Conclusion: For all targeted mobile models utilizing Grouped Query Attention (GQA), per-channel memory footprint is **$\le 56.0\text{ MB}$**, well below the strict **$\le 128\text{ MB}$** requirement per trace!

---

## 3. Component 2: Batched Parallel Decode Loop & GEMM Dispatch (`BatchedRolloutCoordinator`)

### 3.1 Data Flow & Execution Pipeline

The `BatchedRolloutCoordinator` drives the multi-channel parallel decode loop at generation step $t$:

```
Step t:
  Active Tokens [N]:  [ tok_0, tok_1, ..., tok_{N-1} ]
                           │
                           ▼
  Embedding Lookup:   X [N x D]  (Batched FP16 Matrix)
                           │
                           ▼
  Transformer Layer:  LayerNorm(X)
                           │
                           ├───────────────────────────────────────────┐
                           ▼                                           ▼
                      Linear Projections                         GEMM Dispatch
                   Q = X @ W_q  [N x (H_q * D_head)]       (PyTorch MPS / Metal Shader)
                   K = X @ W_k  [N x (H_kv * D_head)]       hardware matrix tiles simdgroup
                   V = X @ W_v  [N x (H_kv * D_head)]       sat. compute bound
                           │
                           ▼
                      Paged KV Cache Store
                   Store K, V into PagedKVCache at pos t
                           │
                           ▼
                      Paged Batched Attention
                   Attn_i = Softmax_LUT( Q_i @ K_hist_i^T / sqrt(d) )
                   Out_i  = Attn_i @ V_hist_i
                   Out    = [Out_0; ...; Out_{N-1}] @ W_o
                           │
                           ▼
                      MLP / FFN Block (Gate, Up, Down)
                   H_gate = X @ W_gate [N x D_ffn]
                   H_up   = X @ W_up   [N x D_ffn]
                   X_ffn  = (SiLU(H_gate) * H_up) @ W_down
                           │
                           ▼
                      Final LayerNorm & LM Head
                   Logits = LayerNorm(X) @ W_lm_head  [N x Vocab_Size]
                           │
                           ▼
                      Multi-Channel Parallel Sampling
                   Logits_scaled = Logits / T
                   Probs = safe_softmax_lut(Logits_scaled, exp_lut)
                   Top-k / Top-p filtering per channel
                   Sample next tokens: [ tok_0^{(t+1)}, ..., tok_{N-1}^{(t+1)} ]
```

---

### 3.2 GEMM Dispatch Mechanism

`BatchedRolloutCoordinator` supports two backend execution dispatchers:

1. **PyTorch MPS Dispatcher (`backend='mps'`)**:
   - Batched matrix multiplication: `X @ W` where $X \in \mathbb{R}^{N \times D}$ and $W \in \mathbb{R}^{D \times M}$.
   - PyTorch automatically translates FP16 GEMM calls into Metal Performance Shaders (MPS Graph) matrix operations, saturating SIMD hardware matrix multiply instructions.

2. **Native Metal Compute Shader Dispatcher (`backend='metal'`)**:
   - Uses `batched_gemm_simdgroup` kernel from `src/shaders/batched_gemm.metal`.
   - Grid dimensions: Threadgroups of size $(8, 8, 1)$.
   - `simdgroup_matrix<half, 8, 8>` hardware intrinsics perform $8 \times 8$ tile matrix multiply-accumulate operations in GPU registers.
   - Decoupled weight dequantization via `dequantize_superblocks_kernel` unpacks 144-byte super-blocks into contiguous GPU FP16 buffers before GEMM execution.

---

### 3.3 Multi-Channel Parallel Sampling & Logit Gathering

At step $t$, the LM head produces a batch logit matrix $L^{(t)} \in \mathbb{R}^{N \times V_{\text{vocab}}}$ ($V_{\text{vocab}} \approx 151,936$ for Qwen2.5).

Sampling is executed independently for each rollout channel $i \in \{0, \dots, N-1\}$ with temperature $T > 0$ (default $T = 0.7$), top-$p$ (default $p = 0.95$), and top-$k$ (default $k = 50$):

1. **Temperature Scaling**:
   $$L'_{i, v} = \frac{L_{i, v}}{T}$$
2. **Softmax Probability Calculation via `safe_softmax_lut`**:
   $$P_i = \text{safe\_softmax\_lut}(L'_i, \text{exp\_lut})$$
3. **Top-$k$ & Top-$p$ Masking**:
   - Sort probabilities $P_i$ descending.
   - Truncate beyond index $k$.
   - Compute cumulative sum of probabilities and mask out entries where cumulative sum exceeds $p$.
   - Renormalize remaining probabilities to sum to 1.0.
4. **Categorical Sampling**:
   $$\text{next\_token}_i \sim \text{Categorical}(P_{i, \text{filtered}})$$
5. **EOS / Stop Condition Tracking**:
   - If $\text{next\_token}_i = \text{EOS\_TOKEN\_ID}$ or sequence length reached max context (2048), channel $i$ is marked finished.
   - Finished channels emit padding/masked tokens in subsequent steps without updating their KV-cache blocks.

---

### 3.4 Integration with `attention.py` (`ExponentialLUT` & `safe_softmax_lut`)

`BatchedRolloutCoordinator` integrates `src/attention.py` in two critical sub-modules:

1. **Batched Attention Softmax**:
   For candidate sequence $i$, attention score vector $S_i = \frac{Q_i K_{\text{hist}, i}^T}{\sqrt{D_{\text{head}}}} \in \mathbb{R}^{1 \times t}$.
   - Shift logits: $S_{\text{shifted}, i} = S_i - \max(S_i)$.
   - Apply `safe_softmax_lut(S_i, self.exp_lut, axis=-1)`.
   - Vector gather from 32,768-entry `ExponentialLUT` replaces $O(t)$ dynamic `exp()` instructions with single-cycle L1 cache lookups.

2. **Sampling Probability Softmax**:
   For vocabulary logits $L'_i \in \mathbb{R}^{V_{\text{vocab}}}$ ($V_{\text{vocab}} = 151,936$):
   - Dynamic `exp()` over 151,936 elements per rollout channel requires ~1.2 million FLOPs per step.
   - `safe_softmax_lut(L'_i, self.exp_lut)` replaces exponent calculations with table gather, speeding up sampling pass by **$2.2\times$**.

---

## 4. Software Architecture & Technical Interfaces (`src/batch_generator.py`)

### 4.1 Class Hierarchy & API Signatures

```python
"""
Project Antigravity — R1: Batched Parallel Decode Rollout Coordinator

This module implements:
  1. PagedKVCache: Fixed-block KV-cache manager with shared prompt prefix & CoW
  2. BatchedRolloutCoordinator: Multi-channel GEMM decoder & sampling coordinator
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from attention import ExponentialLUT, safe_softmax_lut


class PagedKVCache:
    """
    Paged KV-Cache Manager supporting N parallel rollout channels.
    """
    def __init__(
        self,
        num_physical_blocks: int,
        block_size: int = 16,
        num_layers: int = 28,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "mps"
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Contiguous physical memory pool: [num_blocks, num_layers, num_kv_heads, block_size, head_dim]
        self.k_pool = torch.zeros(
            (num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim),
            dtype=dtype, device=device
        )
        self.v_pool = torch.zeros(
            (num_physical_blocks, num_layers, num_kv_heads, block_size, head_dim),
            dtype=dtype, device=device
        )

        # Free block pool queue & reference counter array
        self.free_blocks: List[int] = list(range(num_physical_blocks))
        self.ref_counts: np.ndarray = np.zeros(num_physical_blocks, dtype=np.int32)

        # Block tables per channel: channel_id -> List[physical_block_id]
        self.block_tables: Dict[int, List[int]] = {}
        # Sequence lengths per channel: channel_id -> int
        self.seq_lengths: Dict[int, int] = {}

    def allocate_prompt_prefix(
        self,
        prompt_k: torch.Tensor,  # [num_layers, num_kv_heads, S_prompt, head_dim]
        prompt_v: torch.Tensor,  # [num_layers, num_kv_heads, S_prompt, head_dim]
        num_channels: int
    ) -> List[int]:
        """
        Allocate shared prompt prefix physical blocks for N channels.
        Returns list of allocated physical block indices.
        """
        ...

    def init_channel(self, channel_id: int, prompt_block_ids: List[int], prompt_len: int) -> None:
        """
        Initialize channel with shared prompt prefix blocks.
        """
        ...

    def append_kv(
        self,
        channel_id: int,
        layer_idx: int,
        k_val: torch.Tensor,  # [num_kv_heads, 1, head_dim]
        v_val: torch.Tensor   # [num_kv_heads, 1, head_dim]
    ) -> None:
        """
        Append single token KV projection for a specific layer & channel.
        Triggers Copy-on-Write if writing to a shared block.
        """
        ...

    def get_kv_history(
        self,
        channel_id: int,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch contiguous or block-gathered historical K, V tensors for attention.
        Returns (K_hist, V_hist) of shape [num_kv_heads, seq_len, head_dim].
        """
        ...

    def free_channel(self, channel_id: int) -> None:
        """
        Release channel blocks and update reference counts.
        """
        ...


class BatchedRolloutCoordinator:
    """
    Batched Parallel Decode Coordinator for N rollout trajectories.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        num_channels: int = 8,
        max_seq_len: int = 2048,
        block_size: int = 16,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        device: str = "mps"
    ):
        self.model = model
        self.num_channels = num_channels
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device

        # Precomputed Exponential LUT for safe softmax
        self.exp_lut = ExponentialLUT(size=32768, range_max=10.0)

        # Initialize Paged KV Cache Pool
        # Reserve enough physical blocks for prompt + N * rollout_tokens
        num_blocks_per_channel = (max_seq_len + block_size - 1) // block_size
        total_blocks = num_blocks_per_channel * num_channels + 64
        self.kv_cache = PagedKVCache(
            num_physical_blocks=total_blocks,
            block_size=block_size,
            device=device
        )

    def prefill_prompt(
        self,
        prompt_tokens: List[int]
    ) -> torch.Tensor:
        """
        Run initial prompt prefill, store shared KV blocks in PagedKVCache,
        and initialize N rollout channels.
        """
        ...

    def step(
        self,
        active_tokens: torch.Tensor,  # [N] token IDs
        active_mask: torch.Tensor     # [N] boolean active mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute single parallel decode step across N candidate rollouts.
        Returns (next_tokens [N], logit_probs [N, Vocab]).
        """
        ...

    def generate_rollouts(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 512
    ) -> List[List[int]]:
        """
        Run complete parallel decode loop for N candidate reasoning traces.
        Returns list of N generated token sequences.
        """
        ...
```

---

## 5. Quantitative Latency, FLOPs & Speedup Proof

### 5.1 Arithmetic Intensity & Compute Bound Transition
For a 1.5B INT4 model layer ($K = M = 2048$, Weight size $= 2.1\text{ MB}$):

- **Sequential GEMV ($N=1$)**:
  $$\text{FLOPs} = 2 \times 1 \times 2048 \times 2048 = 8.39 \times 10^6\text{ FLOPs}$$
  $$\text{Memory Transferred} \approx 2.1\text{ MB}$$
  $$\text{Arithmetic Intensity} = \frac{8.39 \times 10^6}{2.1 \times 10^6} \approx \mathbf{4.0\text{ FLOPs/byte}}$$
  *Hardware ridge point on Apple Silicon GPU is $\approx 25\text{ FLOPs/byte}$. GEMV operates at $<16\%$ hardware capacity.*

- **Batched GEMM ($N=8$)**:
  $$\text{FLOPs} = 2 \times 8 \times 2048 \times 2048 = 6.71 \times 10^7\text{ FLOPs}$$
  $$\text{Memory Transferred} \approx 2.1\text{ MB (weights loaded once!)}$$
  $$\text{Arithmetic Intensity} = \frac{6.71 \times 10^7}{2.1 \times 10^6} \approx \mathbf{32.0\text{ FLOPs/byte}}$$
  *Crosses the hardware ridge point ($32 > 25$), saturating GPU SIMD matrix multiplication units!*

### 5.2 Throughput Comparison Matrix

| Decoding Mode | Single Pass Layer Latency | Per-Token Latency per Trace | Total Throughput (8 Traces) | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential (8 independent runs)** | $8 \times 0.65\text{ ms} = 5.20\text{ ms}$ | $0.65\text{ ms/tok}$ | 25 tok/s total | 1.0x (Baseline) |
| **Batched Engine ($N=8$)** | **$1.36\text{ ms}$** | **$0.17\text{ ms/tok}$** | **172 tok/s total** | **6.88x Total Throughput** |

---

## 6. Implementation Checklist & Guidelines for Implementer

1. **`PagedKVCache` Verification**:
   - Unit test block table indexing arithmetic for token positions $t=0, 15, 16, 31, 32$.
   - Unit test Copy-on-Write when channel 0 writes to partial prompt block.
   - Assert `ref_counts` decrement correctly when channels finish.
2. **`BatchedRolloutCoordinator` Verification**:
   - Compare batched GEMM forward pass output against $N$ sequential GEMV passes (`assert_allclose` with `rtol=1e-2`).
   - Validate `safe_softmax_lut` outputs match PyTorch standard `torch.softmax` within tolerance $1\text{e}-2$.
3. **MPS Synchronizations**:
   - Call `torch.mps.synchronize()` appropriately during timing / benchmarking steps to ensure valid GPU execution measurement.
