# PRD & Implementation Plan: Project Antigravity 🛸
### Hardware-Accelerated, Batched On-Device Test-Time scaling (TTS) Engine

This Product Requirements Document (PRD) and Implementation Plan outlines the development of **Project Antigravity**—a native, lightweight on-device inference optimization engine. It is designed to run on local consumer hardware (built on Apple Silicon Macs for iPhones) [333, 337], completely eliminating cloud costs and data privacy leaks [537].

---

## 1. System Directory Map
Create a dedicated workspace folder `/antigravity-engine/` with the following structure to organize your codebase:

```text
/antigravity-engine/
├── config/
│   └── engine_config.yaml         # Quantization parameters, batch size (N), and threshold settings
├── src/
│   ├── __init__.py
│   ├── dequant.py                 # Table-lookup (LUT) INT4 quantization & dequantization kernels
│   ├── attention.py               # Safe precomputed Softmax exponentials & FlashAttention kernels
│   ├── batch_generator.py         # The core batched-decoding (GEMM) parallel rollout coordinator
│   └── verifier.py                # Outcome scoring and list-wise/majority-vote selection logic
├── prompts/
│   ├── reasoning_template.txt     # System instruction set to guide local SFT reasoning models
│   └── verifier_template.txt      # Prompt structure for the local critic/evaluator agent
├── tests/
│   ├── test_quantization.py       # Unit tests verifying INT4 weight repacking
│   └── test_batched_speed.py      # Profile script verifying GEMV vs GEMM speedup
├── benchmark.py                   # End-to-end performance and accuracy profiling harness
└── run_server.py                  # Drop-in OpenAI-compatible API endpoint (localhost:8080)
```

---

## 2. Technical Requirements & Architecture

On consumer edge devices, autoregressive LLM decoding processes only one token at a time [339]. This causes large-tile optimized matrix multiplication units (like Apple Silicon's HMX or mobile NPUs) to degenerate from **GEMM** (General Matrix-Matrix Multiplication) to **GEMV** (General Matrix-Vector Multiplication) [336, 339]. This leaves up to **90% of the hardware compute tiles idle** [336, 339].

**Antigravity's core architectural thesis:** Batch $N$ parallel candidate trajectories (where $N = 4, 8, 16$) together during the decode phase [336, 340]. This converts single-token vector-vector passes back into dense parallel matrix blocks, giving you $N$ parallel thinking rollouts for virtually the same latency cost as a single-pass decode [336, 340].

### The Core Speedup Theorem
*   **Sequential Decode (8x GEMV):** $\text{Time} \approx 5.20\text{ ms}$ (CPU / GPU sequential thread overhead)
*   **Batched Parallel Decode (1x GEMM):** $\text{Time} \approx 1.36\text{ ms}$ (Matrix units utilized at 100%)
*   **Empirical local speedup:** **~3.83x** hardware computation efficiency gain.

---

## 3. Subagent Prompting & Query Harness Instructions

To drive the parallel reasoning traces effectively, you must control formatting using clear, zero-shot structured prompts [82]. Few-shot prompting can degrade local reasoning performance [82, 78].

### System Prompt (`prompts/reasoning_template.txt`)
Save this file to guide your local reasoning subagents (such as `Qwen2.5-Math-7B` or `DeepSeek-R1-Distill-8B`) [47, 440]:

```text
Solve the following math problem efficiently and clearly.

Use this step-by-step format:
<think>
Identify your strategy, evaluate intermediate steps, and self-correct if you spot an error.
</think>

Therefore, the final answer is: \boxed{answer}.

Problem:
{problem}
```

### Critic / Verifier Prompt (`prompts/verifier_template.txt`)
Save this file for your local list-wise verification subagent. Using a **list-wise verifier** (where the model compares all candidates side-by-side) outperforms pairwise or pure scalar scoring on complex tasks [421, 424]:

```json
{
  "instruction": "You are a rigid mathematical verifier. Compare the following N candidate solutions for the problem. Select the index of the correct, most optimal, and non-redundant trajectory.",
  "format": {
    "index": "integer",
    "analysis": "string"
  }
}
```

---

## 4. Harness Code Shells (The Executable Foundation)

### A. The Quantization and Weight Repacking Module (`src/dequant.py`)
To fit within on-device unified memory limits (e.g., 8 GB to 16 GB), we quantize weights to 4-bit and repack them in super-blocks of 256 elements [342, 343]. This aligns fine-grained groups perfectly with local vector registers [342, 343]:

```python
import numpy as np

def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Coalesce 8 fine-grained INT4 quantization groups of size 32 into a
    single aligned super-block of size 256 to ensure wide vector register access.
    """
    assert len(weights_int4) % 256 == 0, "Weight array length must be a multiple of 256."
    num_superblocks = len(weights_int4) // 256
    repacked = weights_int4.reshape(num_superblocks, 8, group_size)
    # Align to 128-byte boundary for NPU/GPU SIMD registers
    return repacked.astype(np.int8)

def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Fast table-lookup dequantization mapping INT4 back to FP16.
    Avoids expensive unpacking instruction cycles in the compute loop.
    """
    # Simply use vector gathering over our precomputed lookup table
    return lut[q_weights]
```

### B. Precomputed Softmax & Vector Gathers (`src/attention.py`)
To prevent exponentials from bottlenecking performance on parallel decodes, use a precomputed exponential lookup table (LUT) [343, 344]:

```python
import numpy as np

class SafeSoftmaxLUT:
    def __init__(self, size: int = 32768):
        # Precompute exponentials for non-positive values (inputs shifted by row-max)
        self.inputs = np.linspace(-10.0, 0.0, size, dtype=np.float16)
        self.lut = np.exp(self.inputs, dtype=np.float16)
        self.step = 10.0 / (size - 1)

    def compute_softmax(self, x: np.ndarray) -> np.ndarray:
        # 1. Compute Safe Offset (Safe Softmax)
        row_max = np.max(x, axis=-1, keepdims=True)
        shifted = x - row_max  # Always non-positive (<= 0)
        
        # 2. Map shifted inputs to our LUT indices using fast vector-gather operations
        indices = np.clip(np.abs(shifted) / self.step, 0, len(self.lut) - 1).astype(np.int32)
        exps = self.lut[indices]
        
        # 3. Normalize
        return exps / np.sum(exps, axis=-1, keepdims=True)
```

---

## 5. Phased Iteration Plan (The Vibecoding Roadmap)

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   PHASE 1: Hardware Baseline  ──►  PHASE 2: Batched Core               │
│   - Setup local directory map      - Implement parallel SFT            │
│   - Run speedup benchmarking script  - Batch N rollouts (GEMM)         │
│                                                                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   PHASE 4: Adaptive Reflect   ◄──  PHASE 3: List-Wise Verifier         │
│   - Add threshold-based reflection - Integrate local verifier          │
│   - Expose local API server        - Select best-of-N candidate        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Hardware-Aware Baseline & Quantization (Days 1–3)
*   **Action:** Build the directory structure and create the initialization files. Run the speedup benchmarking script to verify GEMV vs GEMM latency on your local CPU/GPU.
*   **Test Drive Command:**
    ```bash
    python3 tests/test_batched_speed.py
    ```

### Phase 2: Parallel SFT Decodes & Rollouts (Days 4–7)
*   **Action:** Write the `batch_generator.py` module using local SFT execution frameworks (such as PyTorch or Apple MLX). Force parallel generation by setting a batch size equal to your budget $N$ (e.g., $N = 8$).
*   **Test Drive Command:**
    ```bash
    python3 src/batch_generator.py --problem "Compute the integral of x*cos(x) dx" --num_samples 8
    ```

### Phase 3: List-Wise Verifier Integration (Days 8–11)
*   **Action:** Implement the outcome verification and list-wise selection logic in `verifier.py` [424]. Feed the 8 generated parallel traces into your local verifier model to select the best output [424].
*   **Test Drive Command:**
    ```bash
    python3 src/verifier.py --candidates_path "scratch/rollouts.json"
    ```

### Phase 4: Adaptive Reflection & local API Exposure (Days 12–14)
*   **Action:** Implement threshold-based reflection (meaning: have the agent reflect on its progress only when its step-level verifier score falls below a set threshold, rather than at every single step, to prevent overthinking and token waste) [421, 424]. Spin up `run_server.py` to expose this optimized pipeline locally.
*   **Test Drive Command:**
    ```bash
    python3 run_server.py --port 8080
    ```
    Verify with a local test completion curl request:
    ```bash
    curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Prove that 2^n > n^2 for n >= 5"}]}'
    ```

---

## 6. Verification & Validation Metrics

To measure the success of your implementation, log the following metrics during benchmarking:
1.  **GEMM Acceleration Ratio:** Your parallel decoding latency divided by single-pass decoding latency (Target: $>3.0\times$ speedup).
2.  **Accuracy Gain:** Downstream mathematical accuracy (e.g., on GSM8K/MATH) compared to a single-attempt zero-shot run (Target: $>15\%$ absolute increase).
3.  **Token Efficiency:** Total tokens consumed per correct solution. Confirm that threshold-based reflection consumes at least **35% fewer tokens** than reflecting at every step while maintaining identical solution accuracy.
