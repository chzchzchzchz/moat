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
| BF16 → FP16 conversion | out, but the earlier claim here was overstated: "6,822 patterns, all exact" described a *Python reimplementation* in `tools/analyze_existing_artifacts.py`, not the shipping C++, which no test had ever run. `bf16_to_fp16` now lives in `src/weight_convert.h` and is checked against an independent reference over all 65,536 patterns; that found two real defects in it (NaN → `+Inf`, and truncation instead of rounding in the subnormal range), both now fixed and both incapable of producing constant output |
| decode buffer ping-pong | 22 layers lands the final hidden state in `batch_hidden_1`, which is what `final_hidden` selects |
| attention indexing | dispatched per channel, so `batch_idx` in `gqa_attention_scores_kernel` is always 0 — consistent with `kvCaches_[l][c].k_cache` having no batch dimension |
| the causal mask | `q_pos = seq_len - q_len + q_idx` is correct for decode |
| the sampler | it *was* broken and is fixed; it masked the failure rather than causing it |
| the safetensors header parser | four defects fixed, all ruled out **for this checkpoint** by its own header — see below |
| F32 read as FP16 | real, and not latent, but this checkpoint is 201/201 BF16, so it never took that path |
| shape read past its `data_offsets` span | every one of the 201 tensors' shapes matches its span exactly |

#### The weight-loading path is now ruled out by measurement, not by argument

`run_full_gsm8k.py` names `TinyLlama/TinyLlama-1.1B-Chat-v1.0`'s `model.safetensors`, so
that is the checkpoint behind `gsm8k_full_checkpoint.json`. Its 23,088-byte header is
committed at `tests/fixtures/tinyllama-1.1b-chat-v1.0.header.json` — header bytes only, no
weights — and `tests/test_safetensors_header.cpp` asserts the properties that decide
whether each defect could have applied:

| property | value | the defect it rules out |
| :--- | :--- | :--- |
| dtypes | 201 of 201 **BF16** | F32 read as FP16, which copies half the tensor and misreads every value |
| `__metadata__` | `{"format":"pt"}`, the first key | the `find('}')` skip: flat metadata is the one shape it survives |
| nested object in metadata | none | the same skip's misparse, which loses the real tensor |
| `}` inside a metadata value | none | the string-literal variant of it |
| shape vs `data_offsets` span | consistent for all 201 | reading past a tensor in the conversion or transposing loop |
| `data_start` + largest `offset_end` | 2,200,119,864 = the file size | the missing bounds check against a truncated file |

The two `bf16_to_fp16` defects found by testing it for the first time — a NaN with a
mantissa below `0x10` becoming `+Inf`, and the subnormal path truncating instead of
rounding — are also out. Neither produces constant output: the first needs a NaN already in
the checkpoint, and the second is ±1 ulp on magnitudes below 6.1e-05.

**So nothing in the weight-loading path explains the degenerate run.** Every defect there
is real and each fails by producing a model that loads and runs, which is why they were
worth finding; none of them is this. Item 1 below is still the first thing to do, and a
reader who assumed the weight-loading audit had closed the question would skip it.


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

~~`MetalTransformerEngine::generate()`'s prefill loop creates a command buffer per prompt
token~~ — **done.** It now encodes up to `antigravity::prefillChunkTokens(n_layers)` prompt
tokens into one command buffer: 31 tokens for a 22-layer model, so a 100-token prompt costs 4
round trips instead of 99.

The verification that condition asked for came out clean, and is worth keeping: nothing in
the loop reads a GPU result back (the only CPU-side input is `prompt_tokens[t]`, known before
the loop, and `setBytes:` copies it into the encoder at encode time rather than aliasing a
variable the next iteration overwrites); `MTLComputeCommandEncoder` is serial unless created
with `MTLDispatchTypeConcurrent`, so token *t*'s attention still reads the KV entries token
*t−1* wrote; and `forwardLayer()` takes `cmdBuf` but never uses it — only the encoder — so it
never needed a per-token boundary. The same dispatches in the same order, with only the commit
granularity changed, which is why it was safe to do without a device.

It is chunked rather than one buffer for the whole prompt because a command buffer holds every
dispatch until committed: a 2048-token prompt over 22 layers would otherwise encode around
half a million dispatches before anything started executing. The chunk size and the ranges
live in `engine_limits.h` and are tested, because a wrong bound here means prefill silently
skips prompt tokens — a partially ignored prompt producing confident output that does not
follow from its input.

**This is not measured.** It removes 95% of the prefill round trips by construction; whether
prefill was a material share of end-to-end time is a question for a device.

`generateMultimodal()` was worse: `for t { for c { … commit; waitUntilCompleted } }`, so
prefill cost `prefill_len * n_channels` round trips — eight times the text path for the same
prompt. **Its prefill loop is fixed:** the channels now share one command buffer per token,
because they share nothing else (channel *c* touches only `hidden_bufs[c]`, `hidden_bufs2[c]`
and `kvCaches_[l][c]`). The token loop is deliberately *not* merged there, unlike in
`generate()`: `hidden_bufs[c]` is reused for the next token and written from the CPU for image
patches, so merging tokens would let the CPU overwrite a buffer the GPU was still reading.

**Its decode loop is still per-channel, and this is the one throughput item left specified but
not done.** Every channel's `lm_head` writes the same `scratchLogits_`, and the CPU samples
from it before the next channel runs, so merging the channels would have channel *c+1*
overwrite logits that channel *c* has not been sampled from yet. The fix is a logits buffer
per channel — `vocab_size * 2 * n_channels`, about 512 KB at a 32k vocabulary and 2.4 MB at
Qwen's 151,936 — after which all channels share one command buffer per step, as `generate()`
already does.

It is left undone on purpose. This path is exposed through the C API and `native_bridge`, but
no test or benchmark calls it, so the change could be validated by nothing at all. Given what
this repository's history is made of, an unverifiable restructure of an unexercised path is
the wrong trade; it wants a caller and a test first.

`metal_hardware_proof.md` §2.1 works out that the GEMM microbenchmark implies a ceiling
near 123 tok/s per channel while the committed artifacts report about 5.14 — a ~24×
gap, which says arithmetic is not the bottleneck and the surrounding execution is.
These two items are the leading candidates.

**Do not start here.** Optimising a path whose correctness is unestablished is how this
repository arrived at a 0.3% benchmark that nobody noticed.

---

## Appendix: shader audit coverage

Every kernel in `src/shaders` has been read looking for the cause of the degenerate
587-problem run. Recorded so the ground is not re-covered, and so the gaps are visible.

| kernel | outcome |
| :--- | :--- |
| `rmsnorm_kernel` | **fixed** — tree reduction only summed a power-of-two prefix of its threads |
| `rope_kernel` | **fixed** — read the frequency tables with no bound on `absolute_pos` |
| `kv_cache_append_kernel` | **fixed** — took `max_seq` and never used it as a bound; overrun lands in the next head's cache |
| `fused_batched_gemm_int4` | **fixed** — dequantized every tile 32x over, raced the staging tile, and stored edge tiles out of bounds |
| `gemv_int4_kernel` | verified against a host reference: RMS error 0.017914 vs an analytic 0.017972 |
| `moe_router` | **fixed earlier** — `thread float logits[64]` indexed by a runtime count, and a softmax with no maximum subtracted |
| `batched_gemm_simdgroup` | correct, including its bounds-checked edge store |
| `gqa_attention_scores_kernel` | correct — `batch_idx` is always 0 because the grid's x extent is `n_heads`, matching a per-channel cache; mask is right for decode |
| `softmax_kernel` | correct — dispatched at 32 threads so the cross-SIMD reduction degenerates properly; in-place is safe |
| `attention_value_kernel` | correct — probs stride matches what the scores kernel wrote |
| `silu_elementwise_mul_kernel` | correct for finite inputs; guards `gid >= size` |
| `residual_add_kernel` | correct; guards `gid >= size` |
| `embedding_lookup_kernel` | bounds check added at both multimodal call sites |
| `gemv_kernel` | correct; `B[k * N + col]` is contiguous across adjacent threads |
| `dequantize_superblocks_kernel` | **dead, and should stay dead** — materialising FP16 weights undoes the bandwidth saving INT4 exists for |
| `deltanet_forward`, `moe_router` pipelines | **created, never dispatched** — `forwardLayer()` is dense for every layer, so the hybrid architecture is not implemented; `loadWeights()` now warns on such a checkpoint |

Three pipelines are created and never dispatched: `deltanetPipeline_`, `moeRouterPipeline_`
and `antigravity_c_api.cpp`'s `dequantPipeline`. None is deleted — each is the start of
real work and two have had genuine bugs fixed in them — but each is now annotated where
it is created, so none of them reads as a working feature.

**No defect found in this audit explains the degenerate run.** Six of the fixes are
latent: they need a dimension that is not a multiple of 8, a `hidden_dim` below 256, a
sequence past `max_seq_len`, or a non-dense checkpoint, and none of those held for the
TinyLlama run that produced it. Which means the cause is still open, and item 1 above is
still the first thing to do.
