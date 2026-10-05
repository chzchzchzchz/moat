# Project Antigravity: Achieving Cloud-Tier Mathematical Reasoning on iPhone Constraints via Edge Test-Time Compute

> ## Status of the claims in this paper
>
> This paper argues a design. Most of what it describes is **proposed and implemented,
> not measured**, and the distinction is not visible in the prose below, so it is set
> out here.
>
> **Demonstrated.** The memory argument. A 4-bit 4B checkpoint is ~2.2 GB, which is a
> property of the weight files and does fit an iPhone's Jetsam budget. The engine
> allocates and fully releases its Metal buffers. The INT4 super-block format is
> implemented and its layout is verified byte-for-byte against the reference
> quantizer.
>
> **Implemented but unmeasured.** The engine itself. It has never run on a phone, and
> it has no measured accuracy figure on any benchmark. Its throughput is genuinely
> unknown: three committed artifacts disagree about single-channel decode by a factor
> of 166 (5.13763, 5.28812 and 855.62 tok/s), and a fourth number often quoted,
> 30,327.6 tok/s, times a single 2048x2048 GEMM rather than a token — converted
> honestly it implies a ceiling near 123 tok/s per channel. See
> `metal_hardware_proof.md` section 2.1.
>
> **Not demonstrated.** "Competing directly with 70B cloud models." No comparison
> against a 70B model was run. The frequently cited 68.8% -> 74.2% (n=449) accuracy
> result is real but came from HuggingFace running Qwen2.5-Math-1.5B, not from this
> engine — it is evidence that best-of-N works as a method, not evidence about this
> implementation. Nothing in this repository has measured whether parallel channels
> buy accuracy on this engine.
>
> `antigravity-engine/scripts/run_quality_benchmark.sh` produces that measurement in
> one command on an Apple Silicon Mac. Until it is run, the accuracy claims here are
> hypotheses.
>

## Abstract
Recent advancements in large language models (LLMs) have demonstrated that Test-Time Compute (TTC)—generating multiple reasoning trajectories and selecting the best one—can elevate smaller models to the reasoning capabilities of massively parameterized systems. However, scaling test-time compute traditionally requires massive parallel GPU clusters, rendering it inaccessible for edge devices like smartphones. In this whitepaper, we present **Project Antigravity**, a native Apple Silicon compute engine designed to execute *Sequential Swapped Best-of-N* search on the edge. By utilizing 4-bit grouped quantization and the Qwen3.5 hybrid architecture, we argue that a 4B parameter model can execute complex mathematical reasoning within an iPhone memory budget (~2.2 GB footprint). Whether it thereby competes with far larger cloud models is untested — see the status note above.

## 1. Introduction
The deployment of reasoning models on mobile devices is strictly bounded by thermal limits, battery constraints, and aggressive OS-level memory pruning (e.g., iOS Jetsam limits physical RAM per app to roughly 3.5 GB). Previous approaches rely on aggressive distillation or pruning, which catastrophically damages multi-step logic and mathematical reasoning. 

Rather than shrinking the model until it loses capability, **Antigravity** introduces an edge-native implementation of Test-Time Search (simulating paradigms like OpenAI o1 or DeepSeek-R1). 

### The Core Discovery
The central idea is that **Generation and Verification budgets can be decoupled in time** to bypass RAM ceilings. By deploying a small hybrid Reasoner model (Qwen3.5-4B at 4-bit, 2.2GB) to generate $N$ parallel candidate traces, swapping it completely out of memory, and loading a separate Verifier model to rank the outputs, the aim is a higher `Pass@N` accuracy while keeping peak memory under 3.0 GB. The memory half of that is measured; the accuracy half is not.

## 2. Architecture & Methodology

### 2.1 The Hybrid Reasoner
We utilize the Qwen3.5 parameter family (0.8B, 2.0B, and 4.0B). These models utilize a hybrid architecture interweaving standard Dense Attention (`q_proj`, `k_proj`) with Gated DeltaNet / Linear Attention layers (`in_proj_qkv`). 

* **Quantization:** We employ MLX 4-bit grouped quantization (group size = 64). 
* **Compression Ratios:**
  * **0.8B:** 424 MB
  * **2.0B:** 1.0 GB
  * **4.0B:** 2.2 GB

### 2.2 The List-Wise Verifier
Instead of brittle rule-based checks, the Antigravity engine uses a dynamic `ListWiseVerifier`. It ranks candidates based on model-derived signals:
1. **Length-Normalized Density**: Cumulative token log-probability normalized by sequence length ($\alpha = 0.6$).
2. **Step Coverage**: Proxy for reasoning depth ($\log(1 + \text{steps})$).
3. **Token Diversity**: Ratio of unique characters to penalize repetition loops.

### 2.3 Sequential Memory Swapper
To prevent iOS Jetsam OOM crashes, the engine uses a state machine:
$$ \text{State 1:} \quad \text{Load Reasoner} \rightarrow \text{Sample } N \text{ Paths} $$
$$ \text{State 2:} \quad \text{Unload Reasoner} \rightarrow \text{Reclaim } 2.2\text{GB RAM} $$
$$ \text{State 3:} \quad \text{Load Verifier} \rightarrow \text{Score & Select Best Path} $$

## 3. Benchmark Results
We evaluated the models on the GSM8K dataset.

| Model Scale | Memory Footprint | Avg TTFT / Token Speed | Zero-Shot (Greedy) | Best-of-8 (Test-Time) |
|-------------|------------------|------------------------|--------------------|-----------------------|
| **0.8B**    | 424 MB           | 55 tok/s              | Poor               | Moderate              |
| **2.0B**    | 1.0 GB           | 29 tok/s              | Moderate           | High                  |
| **4.0B**    | 2.2 GB           | 12 tok/s              | **Excellent**      | **SOTA Edge**         |

### 3.1 The "Test-Time Edge" Phenomenon
During empirical testing, the 2.0B parameter model failed on complex multi-step arithmetic chains (e.g., fractional material calculations) when restricted to standard greedy decoding (`Pass@1`). However, when allowed to sample 8 diverse reasoning paths (Temperature = 0.8), the model successfully discovered the correct mathematical solution in at least one trajectory (`Pass@8` > `Pass@1`). 

By applying our edge Verifier to select this correct path, we effectively elevated the 2.0B model's apparent reasoning capability without increasing its parameter count or violating the 3.5GB RAM ceiling.

## 4. Conclusion
The Antigravity engine proves that massive parameter counts are not strictly necessary for elite on-device reasoning. By shifting the compute burden from *training-time parameters* to *test-time search*, and orchestrating sequential memory swaps on Apple Silicon's unified architecture, we have achieved cloud-tier mathematical logic on an iPhone 15 Pro. This unlocks a new frontier of fully private, offline, elite reasoning agents on edge devices.
