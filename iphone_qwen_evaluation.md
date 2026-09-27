# Qwen3.5 Model Family Evaluation (0.8B vs 2B vs 4B) — MLX on a Mac, not iOS, not this engine

> **Two corrections to how this report reads, both verifiable from the scripts that
> produced it.**
>
> **1. None of this used the Antigravity engine.** The measurements come from
> `mlx_lm` (`run_gsm8k_mlx.py`, `run_gsm8k_mlx_batch.py` — `from mlx_lm import load,
> generate`), which is Apple's own framework. So nothing here is evidence about this
> project's Metal engine, its INT4 super-blocks, or its parallel channels. The engine
> has no measured accuracy figure; `antigravity-engine/scripts/run_quality_benchmark.sh`
> is what produces one.
>
> **2. No iPhone ran anything.** MLX runs on macOS. The report says it targeted a
> memory ceiling "to simulate an iPhone 15 Pro iOS application environment", and that
> is what happened — a simulated budget on a Mac. The footprints below are properties
> of the model files and do carry over. **The throughput figures do not.** An M-series
> Mac has several times an A-series iPhone's memory bandwidth and far more sustained
> thermal headroom, and decode is bandwidth-bound, so a phone will be slower by a
> margin this report cannot tell you. Quoting "~55 tokens/sec" as an iPhone figure is
> not supported by anything measured here.
>
> The accuracy section is a qualitative read of a handful of outputs — "Very poor",
> "Exceptional" — with no sample count and no interval. Treat it as an impression, not
> a measurement.

This report evaluates the **Qwen3.5** architecture across three parameter scales under a simulated iOS memory (Jetsam) budget. All models were downloaded, loaded, and compressed to 4-bit grouped quantization using the MLX framework on Apple Silicon.

## 1. Physical Footprint & iPhone Viability
We targeted a strict ~3.5GB memory budget ceiling to simulate an iPhone 15 Pro iOS application environment.

* **Qwen3.5-0.8B (4-bit):** 424 MB
* **Qwen3.5-2B (4-bit):** 1.0 GB 
* **Qwen3.5-4B (4-bit):** 2.2 GB

**Conclusion:** *All three models completely fit inside the iPhone memory ceiling.* The 4B is the largest viable local model, while the 0.8B footprint is almost negligible.

## 2. Speed and Throughput (on a Mac, via MLX)
Tested generating exactly 256 tokens of `<think>` reasoning chain on an Apple Silicon
Mac via MLX. These are not iPhone figures and not this engine's figures; see the note
at the top.

* **0.8B:** ~4.5 seconds per prompt (~55 tokens/sec)
* **2B:** ~8.8 seconds per prompt (~29 tokens/sec)
* **4B:** ~20.5 seconds per prompt (~12.5 tokens/sec)

**Conclusion:** 0.8B crushes 2B and 4B in pure speed on the hardware, offering real-time streaming speeds well above human reading capability. 

## 3. Mathematical Reasoning Accuracy (Zero-Shot) — impressions, not measurements
The models were run against GSM8K to see whether 4-bit quantization destroyed their
capacity to reason. What follows is a qualitative read of sample outputs with no
sample count, no accuracy percentage and no confidence interval, so it cannot support
a comparison between the three models beyond "the 4B was visibly better".

* **0.8B Accuracy:** Very poor. The model got stuck in loops, outputting generic text, or over-explaining the prompt logic without actually finishing the math before hitting the 256 token limit.
* **2B Accuracy:** Moderate. It successfully answered simpler logic questions (e.g. 3 bolts of fiber) perfectly, but failed to complete the longer arithmetic pipelines before running out of steam.
* **4B Accuracy:** **Exceptional.** The 4B model successfully generated flawless logical chains of thought, identifying individual variables, subtracting steps correctly, and outputting the perfect final answer identical to the 70B cloud models (e.g., exactly matching the Ground Truth of 18 eggs).

## Final Recommendation
1. **If Speed/Memory is absolute priority:** `Qwen3.5-0.8B` runs very lightly but acts
   more as a conversational parser than a strict reasoner.
2. **The likely target for this engine:** `Qwen3.5-4B (4-bit)` at 2.2 GB fits an
   iPhone's memory budget, which is a real and useful finding — memory footprint is a
   property of the weights and transfers directly.

**What this does not establish.** That the 4B is usable on an iPhone rests on
throughput and thermals, and this report measured neither on a phone. The 20-second
figure is a Mac running MLX. Whether the same model is tolerable on an A-series chip,
under a phone's bandwidth and sustained-load limits, through *this* engine rather than
MLX, is unmeasured — and it is the question that decides whether the product works.
Answering it needs a run on an actual device.
