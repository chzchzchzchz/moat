# Project: Project Antigravity 🛸
### Hardware-Accelerated, Batched On-Device Test-Time Scaling (TTS) Engine

## Architecture Overview
Project Antigravity transforms single-token vector-vector autoregressive decoding (GEMV) into dense parallel matrix blocks (GEMM) by batching N parallel candidate reasoning traces (N=4, 8, 16). Built for Apple Silicon / iOS hardware, it achieves 4-8x parallel inference at zero extra compute cost.

The architecture comprises:
1. **INT4 Quantization & Superblock LUT Dequantization (`src/dequant.py`)**: Quantizes 1.5B–3B model weights to INT4 with group size 32, coalescing 8 groups into 256-element SIMD superblocks (128-byte aligned). Runtime dequantization uses precomputed table lookup (LUT) with vector gather operations.
2. **Precomputed Safe Softmax & Fast Attention (`src/attention.py`)**: Uses a 32K-entry FP16 exponential lookup table (`SafeSoftmaxLUT`) with row-max offset mapping to eliminate exponentials bottleneck during parallel decode.
3. **Batched Decode & GEMM Parallel Rollout Coordinator (`src/batch_generator.py`)**: Manages parallel reasoning trajectories using structured prompts (`prompts/reasoning_template.txt`). Converts N single decodes into dense GEMM passes.
4. **List-Wise Verifier & Adaptive Reflection Engine (`src/verifier.py`)**: Evaluates N candidate rollouts side-by-side using `prompts/verifier_template.txt`. Triggers step-level reflection dynamically when scores fall below a configurable threshold (saving ≥30% tokens).
5. **Native OpenAI-Compatible API Server (`run_server.py`)**: Exposes an HTTP API at `http://localhost:8080/v1/chat/completions` with zero external dependencies.
6. **E2E Benchmark Harness & Diagnostic Suite (`benchmark.py`, `tests/`)**: Measures GEMM speedup ratio (target ≥3x), memory footprint (target ≤4.5GB), end-to-end latency (<30s), LUT softmax acceleration (≥1.5x), accuracy gain (≥15%), and token savings (≥30%).

## Code Layout
```text
/Users/MohssineChazi2/moat/
├── config/
│   └── engine_config.yaml         # Quantization parameters, batch size (N), threshold settings
├── src/
│   ├── __init__.py
│   ├── dequant.py                 # Table-lookup (LUT) INT4 quantization & superblock repack
│   ├── attention.py               # Precomputed SafeSoftmaxLUT & attention kernels
│   ├── batch_generator.py         # Parallel decoding (GEMM) rollout coordinator
│   └── verifier.py                # List-wise selection & threshold adaptive reflection
├── prompts/
│   ├── reasoning_template.txt     # System prompt for parallel reasoning rollouts
│   └── verifier_template.txt      # Structured prompt for list-wise verifier agent
├── tests/
│   ├── test_quantization.py       # Unit tests for INT4 weight repacking & LUT dequant
│   └── test_batched_speed.py      # Profiling script verifying GEMV vs GEMM speedup
├── benchmark.py                   # E2E performance and accuracy benchmark harness
├── run_server.py                  # Drop-in OpenAI-compatible API server (localhost:8080)
└── README.md                      # Architecture overview, setup, and benchmark guide
```

## Milestones Table
| # | Name | Scope | Dependencies | Status | Conversation ID |
|---|------|-------|-------------|--------|-----------------|
| E2E | E2E Testing Track | Requirement-driven test suite (Tiers 1-4) & TEST_READY.md | None | IN_PROGRESS | ed335a16-f1cf-414c-b708-5188c82a55d9 |
| M1 | INT4 Quantization & Superblock LUT Dequantization | `config/engine_config.yaml`, `src/dequant.py`, `tests/test_quantization.py` | None | IN_PROGRESS | 99d44cd1-3e5c-4614-b779-8476d60a7b44 |
| M2 | Precomputed Softmax & Fast Attention Engine | `src/attention.py` micro-benchmarks & integration | M1 | PLANNED | TBD |
| M3 | Batched Parallel Decode & GEMV-to-GEMM Parallel Rollouts | `src/batch_generator.py`, `prompts/reasoning_template.txt`, `tests/test_batched_speed.py` | M1, M2 | PLANNED | TBD |
| M4 | List-Wise Verifier & Adaptive Reflection Engine | `src/verifier.py`, `prompts/verifier_template.txt` | M3 | PLANNED | TBD |
| M5 | Native OpenAI-Compatible API Server | `run_server.py` | M3, M4 | PLANNED | TBD |
| M6 | Final Integration, Benchmark Harness & Documentation | `benchmark.py`, `README.md`, full E2E test verification | M1-M5, E2E | PLANNED | TBD |

## Interface Contracts

### 1. `config/engine_config.yaml`
```yaml
engine:
  model_name: "Qwen2.5-Math-1.5B"
  quantization:
    bits: 4
    group_size: 32
    superblock_size: 256
    alignment_bytes: 128
  batch_size: 8 # N candidates (4, 8, 16)
  softmax_lut_size: 32768
  reflection_threshold: 0.75
server:
  host: "0.0.0.0"
  port: 8080
```

### 2. `src/dequant.py`
```python
def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
    """Coalesce 8 fine-grained INT4 groups of 32 into a 256 super-block."""
    pass

def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Fast table-lookup dequantization mapping INT4 to FP16."""
    pass
```

### 3. `src/attention.py`
```python
class SafeSoftmaxLUT:
    def __init__(self, size: int = 32768): ...
    def compute_softmax(self, x: np.ndarray) -> np.ndarray: ...
```

### 4. `src/batch_generator.py`
```python
class BatchedDecodeGenerator:
    def generate_candidates(self, problem: str, num_samples: int = 8) -> list[dict]: ...
```

### 5. `src/verifier.py`
```python
class ListWiseVerifier:
    def select_best_candidate(self, problem: str, candidates: list[dict]) -> dict: ...
    def should_reflect(self, score: float, threshold: float) -> bool: ...
```

### 6. `run_server.py`
API Endpoint: `POST http://localhost:8080/v1/chat/completions`
Response format: Standard OpenAI JSON response schema.
