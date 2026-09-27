# What needs an Apple Silicon Mac

Everything in this file is blocked on hardware, not on effort. It cannot be done in
CI: `MTLCreateSystemDefaultDevice()` returns nil on GitHub's `macos-14` runners — they
are arm64 VMs without GPU passthrough — and `macos-14-xlarge` is not available on this
account. Both were measured rather than assumed; the `INT4 kernels on a real GPU
(macOS)` CI job reports the result on every run.

The items are in priority order, and the order matters. The first one decides whether
the rest is worth doing.

---

## 1. Does the engine produce input-dependent output? (one minute)

```
cd antigravity-engine
scripts/run_quality_benchmark.sh
```

Before grading anything, that script runs `tools/check_engine_sanity.py`: three prompts
sharing almost no tokens, which must not produce the identical token sequence, must not
be one token or one character repeated, and must generate something. It exits 1 and
stops if any of those fail.

**Why this is first.** `gsm8k_full_checkpoint.json` holds 587 GSM8K problems produced
by this engine — `run_full_gsm8k.py` constructs
`NativeMetalEngine(dylib_path='libantigravity_engine.dylib')`. In 413 of 587 the
generated text is a single character repeated, the **same** character in 100% of them,
across 587 different prompts, and accuracy is 0.3% at 1, 2, 4 and 8 channels. Output
that does not vary with its input is a broken forward pass, not a weak model.

Run `python3 tools/analyze_existing_artifacts.py` for the full breakdown. No GPU needed.

If the check passes, the shader fix was the cause and the benchmark below is worth
having. If it fails, the forward pass has a defect that none of the work on this branch
addressed, and nothing else here should be attempted first.

### Already ruled out, so do not re-investigate

| candidate | why it is out |
| :--- | :--- |
| weights failing to load | a failed load returns −1 and `native_bridge` raises, so the script would have stopped rather than writing 587 rows |
| BF16 → FP16 conversion | `bf16_to_fp16` tested against ground truth (BF16 is the top 16 bits of a float32) over every bit pattern in the weight range: 6,822 patterns, all exact |
| decode buffer ping-pong | 22 layers lands the final hidden state in `batch_hidden_1`, which is what `final_hidden` selects |
| attention indexing | dispatched per channel, so `batch_idx` in `gqa_attention_scores_kernel` is always 0 — consistent with `kvCaches_[l][c].k_cache` having no batch dimension |
| the causal mask | `q_pos = seq_len - q_len + q_idx` is correct for decode |
| the sampler | it *was* broken and is fixed; it masked the failure rather than causing it |

## 2. If it still fails, localise it

Two cheap checks before anything invasive:

- **Read the non-finite logit counters.** `MetalTransformerEngine::nonFiniteLogitCount()`
  and `emptyDistributionCount()` are exposed and `generate()` prints both at the end.
  Non-zero means the forward pass is producing NaN or Inf, and that is where to look.
  This used to be invisible: `std::discrete_distribution` given NaN weights returns
  index 0, so one NaN anywhere pinned every sampled token to id 0 — measured at token 0
  on 400 of 400 draws with a single NaN among 1,999 healthy logits.
- **Compare `ANTIGRAVITY_INT4=1` against FP16.** If one path is clean and the other is
  not, that halves the search space.

Beyond that, the fastest localisation is a layer-by-layer diff against `transformers`
running the same weights, which needs API surface that does not exist yet — there is no
way to read intermediate activations or raw logits through `antigravity_c_api.h`. A
debug entry point returning the logits for one forward pass would be worth the small
amount of new surface.

## 3. The measurement that has never been made

The project's central claim — that N parallel reasoning channels buy accuracy — has
never been measured on this engine.

```
PROBLEMS=200 scripts/run_quality_benchmark.sh
```

Reports Wilson intervals, an exact paired McNemar test, and refuses to call a
difference a lift it cannot distinguish from chance. **Commit the JSON it writes.** It
carries the hardware profile, the weights file and every per-problem record, which is
what makes a number checkable rather than quotable, and it measures throughput in the
same session so accuracy and tok/s finally share one hardware profile.

For calibration: `full_benchmark_results.json` reports pass@1 = 2/30 against
pass@8 = 6/30, which reads as threefold but is p = 0.1250 with heavily overlapping
intervals. About 45 problems at the same disagreement rate would settle it. Note that
pass@8 asks whether the right answer is *anywhere* among 8 samples — selecting it needs
an oracle, so it bounds what best-of-N could reach rather than what ships.

## 4. Throughput, and only after the above

`MetalTransformerEngine::generate()`'s prefill loop creates a command buffer per prompt
token and blocks on `waitUntilCompleted` each time, so a 100-token prompt costs 100
CPU–GPU round trips. `[cmdBuf computeCommandEncoder]` gives a serial-dispatch encoder,
so those dispatches can be encoded into one command buffer and committed once — verify
first that nothing in that loop reads GPU results back between tokens.

`generateMultimodal()` is worse: its loop is `for t { for c { … commit;
waitUntilCompleted; } }`, and it forwards one channel at a time rather than batching
them, which gives up the bandwidth sharing the project's whole premise rests on.

`metal_hardware_proof.md` §2.1 works out that the GEMM microbenchmark implies a ceiling
near 123 tok/s per channel while the committed artifacts report about 5.14 — a ~24×
gap, which says arithmetic is not the bottleneck and the surrounding execution is.
These two items are the leading candidates.

**Do not start here.** Optimising a path whose correctness is unestablished is how this
repository arrived at a 0.3% benchmark that nobody noticed.
