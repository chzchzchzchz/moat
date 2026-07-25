# Project: Project Antigravity 🛸 — Phase 3

## Architecture Overview
Phase 3 builds the core runtime decoding engine connecting INT4 super-block quantization (`src/dequant.py`), precomputed safe softmax LUT (`src/attention.py`), and hardware-accelerated Metal GPU SIMD matrix compute shaders (`src/shaders/batched_gemm.metal`).

Phase 3 Core Modules:
1. **R1: Batched Parallel Decode Rollout Coordinator (`src/batch_generator.py`)**:
   - Manages N parallel candidate reasoning sequences (N=4, 8, 16).
   - Maintains a paged KV-cache allocation across N channels with shared prompt prefix block mapping.
   - Converts single-token decode steps into dense batched GEMM matrix dispatches targeting compiled Metal SIMD matrix shader (`batched_gemm.metal`) / MPS backend.
   - Implements temperature-based parallel sampling ($T > 0$, top-p / top-k) and logit gathering for candidate traces.
2. **R2: Model Weight Loader & Super-Block Repacker (`src/model_loader.py`)**:
   - Parses FP16 / FP32 weight tensors from GGUF and Safetensors files (or format parsers/mock structures).
   - On-the-fly repacking of model weights into 256-element super-blocks (144 bytes: 16-byte scale header + 128-byte packed INT4 payload).
   - Validates memory allocations guaranteeing total footprint stays under the 4.5 GB ceiling (and <= 2.5 GB for 1.5B parameters model weights in super-block format).
3. **R3: Integration & End-to-End Test Suite (`tests/test_batch_generator.py` & `tests/test_model_loader.py`)**:
   - Micro-unit and integration tests verifying single vs batched decode mathematical parity, paged KV-cache memory isolation across channels, 0 NaN/Inf/leaks across 100+ generation steps, non-identical temperature sampling, 100% test pass rate, and execution performance (N=8 50 steps <= 1.0s).

## Code Layout
```text
/Users/MohssineChazi2/moat/antigravity-engine/
├── config/
│   └── engine_config.yaml         # Configuration for quantization, batch sizes, KV cache blocks
├── src/
│   ├── __init__.py
│   ├── dequant.py                 # INT4 quantization & superblock repacking math
│   ├── attention.py               # SafeSoftmaxLUT & attention utilities
│   ├── batch_generator.py         # R1: Batched Parallel Decode Rollout Coordinator
│   ├── model_loader.py            # R2: GGUF / Safetensors Weight Loader & Repacker
│   ├── metal_runner.cpp           # Native Metal runner C++ bridge
│   └── shaders/
│       ├── batched_gemm.metal     # Metal compute shader (simdgroup_matrix)
│       └── batched_gemm.metallib  # Compiled Metal library
├── tests/
│   ├── __init__.py
│   ├── test_quantization.py       # Micro-unit tests for dequant.py
│   ├── test_attention.py          # Micro-unit tests for attention.py
│   ├── test_batched_speed.py      # Benchmark script for GEMV vs GEMM
│   ├── test_batched_speed_metal.py # Real Metal GPU benchmark
│   ├── test_batch_generator.py    # R3: Integration test suite for rollout coordinator
│   └── test_model_loader.py       # R3: Unit test suite for model weight loader & repacker
```

## Milestones Table
| # | Name | Scope | Dependencies | Status | Conversation ID |
|---|------|-------|-------------|--------|-----------------|
| P3.1 | Phase 3 Architecture Investigation | Requirements analysis, module design specs for R1, R2, R3 | None | IN_PROGRESS | TBD |
| P3.2 | R1: Batched Parallel Decode Rollout Coordinator | `src/batch_generator.py` | P3.1 | PLANNED | TBD |
| P3.3 | R2: Model Weight Loader & Super-Block Repacker | `src/model_loader.py` | P3.1 | PLANNED | TBD |
| P3.4 | R3: Integration Test Suite & Verification | `tests/test_batch_generator.py`, `tests/test_model_loader.py` | P3.2, P3.3 | PLANNED | TBD |
| P3.5 | Review, Challenge & Forensic Integrity Audit | Independent code review, empirical stress benchmark, binary veto audit | P3.4 | PLANNED | TBD |

## Interface Contracts

### 1. `src/batch_generator.py`
```python
class PagedKVCache:
    def __init__(self, num_channels: int, block_size: int = 16, max_seq_len: int = 2048, head_dim: int = 64, num_heads: int = 16): ...
    def allocate_prompt(self, prompt_tokens: list[int]) -> None: ...
    def append_tokens(self, channel_indices: list[int], token_ids: list[int]) -> None: ...
    def get_memory_bytes((self) -> int: ...

class BatchedRolloutCoordinator:
    def __init__(self, model_config: dict, num_channels: int = 8, device: str = 'auto'): ...
    def decode_step(self, active_tokens: np.ndarray, temperature: float = 0.7, top_p: float = 0.95) -> np.ndarray: ...
    def generate_rollouts(self, prompt_tokens: list[int], max_new_tokens: int = 50, temperature: float = 0.7) -> list[list[int]]: ...
```

### 2. `src/model_loader.py`
```python
class ModelWeightLoader:
    def __init__(self, max_memory_bytes: int = 4_831_838_208): ... # 4.5 GB ceiling
    def load_and_repack_gguf(self, file_path: str) -> dict: ...
    def load_and_repack_safetensors(self, file_path: str) -> dict: ...
    def validate_memory_footprint(self, num_params: int = 1_500_000_000) -> bool: ...
```

### 3. Acceptance Benchmarks (`tests/test_batch_generator.py` & `tests/test_model_loader.py`)
- Single vs Batched decode parity: `N=1` decode matches `N=8` rollout channel 0 within `atol=1e-2`.
- Memory isolation: Writing to channel $i$ does not alter channel $j \neq i$ KV-cache blocks.
- Performance: 50 generation steps for $N=8$ completes in $\le 1.0$s on Apple Silicon GPU.
- 0 NaN, 0 Inf, 0 memory leaks across 100+ steps.
