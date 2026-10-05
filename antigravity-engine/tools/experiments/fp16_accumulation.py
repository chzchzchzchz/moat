"""Does FP16 accumulation in the GEMMs reproduce the degenerate run?

Runs TinyLlama-1.1B-Chat on CPU three ways, fed exactly what run_full_gsm8k.py fed the engine
(the bare question, no BOS, no chat template), greedy so the conditions are comparable:

  A  fp32 throughout                                  - the reference
  B  every linear output and hidden state rounded to FP16, fp32 accumulation
  C  as B, and every linear layer ACCUMULATES in FP16, rounding once per 8-term block

C models batched_gemm_simdgroup: simdgroup_matrix<half,8,8> acc, one multiply-accumulate per
8-wide K block. Rounding once per block rather than per product is the generous assumption.

Usage:  python3 tools/experiments/fp16_accumulation.py <TinyLlama dir> [n_problems] [new_tokens]

RESULT (2026-10-02, 2 problems, 32 greedy tokens each): A, B and C produced IDENTICAL text,
coherent in all three ("Jane's ducks lay 16 eggs per day, which means she sells 16…"), with no
non-finite logits. So FP16 accumulation in the GEMMs, at this granularity, does NOT reproduce
the degenerate run's word salad or its NaN onset, and neither does FP16 storage. The run also
shows the missing BOS and chat template are not the cause: the reference model answers
sensibly without them. Kept so the hypothesis is not chased again; see NEXT_ON_HARDWARE.md.

Caveat: per-product rounding inside each 8x8x8 multiply-accumulate is not modelled. Given that
C did not flip a single greedy token over 64, the margin is large, but it is not zero.
"""
import json, sys, time
import torch
torch.set_num_threads(1)               # never starve the benchmark running beside this
from transformers import AutoModelForCausalLM, AutoTokenizer

D = sys.argv[1]
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 32

tok = AutoTokenizer.from_pretrained(D)
model = AutoModelForCausalLM.from_pretrained(D, dtype=torch.float32).eval()

def fp16(x):
    return x.half().float()

def fp16_accum_linear(x, weight):
    """y = x @ weight.T with the running sum kept in FP16, rounded after each 8-term block."""
    K = x.shape[-1]
    assert K % 8 == 0
    xs = fp16(x).reshape(*x.shape[:-1], K // 8, 8)              # [..., B, 8]
    ws = fp16(weight).reshape(weight.shape[0], K // 8, 8)       # [N, B, 8]
    # Each block's 8-term partial dot product, computed exactly, then rounded to FP16.
    partial = fp16(torch.einsum('...bk,nbk->b...n', xs, ws))    # [B, ..., N]
    acc = torch.zeros_like(partial[0])
    for b in range(partial.shape[0]):
        acc = fp16(acc + partial[b])
    return acc

MODE = {"m": "A"}
orig_linear_forward = torch.nn.Linear.forward
def linear_forward(self, x):
    if MODE["m"] == "A":
        return orig_linear_forward(self, x)
    if MODE["m"] == "B":
        return fp16(orig_linear_forward(self, fp16(x)))
    return fp16_accum_linear(x, self.weight)                   # C
torch.nn.Linear.forward = linear_forward

# FP16 storage of the residual stream between layers (B and C).
def round_hidden(_mod, _inp, out):
    if MODE["m"] == "A":
        return out
    if isinstance(out, tuple):
        return (fp16(out[0]),) + tuple(out[1:])
    return fp16(out)
for layer in model.model.layers:
    layer.register_forward_hook(round_hidden)

from pathlib import Path
DATASET = Path(__file__).resolve().parents[3] / "gsm8k_test_set.jsonl"
qs = [json.loads(l)["question"] for l in open(DATASET)][:N_PROMPTS]
for qi, q in enumerate(qs):
    ids = tok.encode(q, add_special_tokens=False)              # no BOS, as the engine got it
    print(f"\n=== problem {qi}: {len(ids)} prompt tokens ===", flush=True)
    for m in ("A", "B", "C"):
        MODE["m"] = m
        t0 = time.time()
        cur = torch.tensor([ids])
        past = None
        out_ids, nonfinite_at = [], None
        with torch.no_grad():
            for step in range(NEW):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits = o.logits[0, -1]
                if nonfinite_at is None and not torch.isfinite(logits).all():
                    nonfinite_at = step
                nxt = int(torch.argmax(torch.nan_to_num(logits, nan=-1e30)))
                out_ids.append(nxt)
                cur = torch.tensor([[nxt]])
        text = tok.decode(out_ids)
        print(f"  {m}  ({time.time()-t0:5.1f}s)  non-finite logits first at step: {nonfinite_at}", flush=True)
        print(f"     {text[:150]!r}", flush=True)
