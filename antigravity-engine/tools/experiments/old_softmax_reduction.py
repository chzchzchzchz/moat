"""Does the pre-fix softmax kernel's tree reduction reproduce the degenerate GSM8K run?

The engine that produced quality_gsm8k_full_checkpoint.json (c7d5196) dispatched
softmax_kernel with `min(cur_seq_len, 256)` threads per head, and the kernel (source at
3c5382e, the commit whose metallib was loaded) reduced the per-thread max and sum with

    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1)
        if (tid < s) shared[tid] = op(shared[tid], shared[tid + s]);

That loop is only correct when the thread count is a power of two. For any other count
it silently drops partials: with 6 threads, s = 3 then 1, so shared[2] never reaches
shared[0]. Every position below 256 that is not a power of two therefore normalises its
attention with a max and a sum that miss some keys. A missed max makes exp(score - max)
exceed 1, the missed sum makes the row sum exceed 1, and the probability is stored as
half, so a large enough miss overflows to inf and then NaN downstream.

This script runs TinyLlama (the benchmark's model) with that exact reduction emulated in
the attention softmax, and with the correct softmax, on the benchmark's prompt format,
and reports what each generates. No Metal needed.

    PYTHONPATH=src python3 tools/experiments/old_softmax_reduction.py [--model DIR] [--problems N] [--bare]
"""
import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama


def buggy_tree_reduce(partials, op):
    """The kernel's reduction loop, verbatim, over a 1-D tensor of per-thread partials."""
    shared = partials.clone()
    s = shared.shape[-1] // 2
    while s > 0:
        shared[..., :s] = op(shared[..., :s], shared[..., s:2 * s])
        s >>= 1
    return shared[..., 0]


def old_softmax_row(scores, n):
    """softmax_kernel at 3c5382e for one head-row of n valid half scores ([heads, n])."""
    scores = scores.to(torch.float16).float()
    threads = min(n, 256)
    pad = (-n) % threads
    padded_max = torch.nn.functional.pad(scores, (0, pad), value=-1e9)
    local_max = padded_max.view(scores.shape[0], -1, threads).amax(dim=1)
    local_max = torch.maximum(local_max, torch.full_like(local_max, -1e9))
    max_val = buggy_tree_reduce(local_max, torch.maximum)
    e = torch.exp(scores - max_val[:, None])
    padded_e = torch.nn.functional.pad(e, (0, pad), value=0.0)
    local_sum = padded_e.view(scores.shape[0], -1, threads).sum(dim=1)
    sum_val = buggy_tree_reduce(local_sum, torch.add)
    return (e / sum_val[:, None]).to(torch.float16)


def make_eager(old):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        k = modeling_llama.repeat_kv(key, module.num_key_value_groups)
        v = modeling_llama.repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, k.transpose(2, 3)) * scaling  # [b, h, q, kv]
        b, h, q, kv = scores.shape
        assert b == 1
        probs = torch.zeros_like(scores)
        past = kv - q
        for i in range(q):
            n = past + i + 1  # causal: the engine scores exactly these keys
            row = scores[0, :, i, :n]
            if old:
                probs[0, :, i, :n] = old_softmax_row(row, n).to(scores.dtype)
            else:
                probs[0, :, i, :n] = torch.softmax(row.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous()
        return out, probs
    return attention


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--problems", type=int, default=3)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--bare", action="store_true",
                    help="feed the bare question with no BOS and no chat template, exactly as "
                         "run_full_gsm8k.py fed the engine")
    ap.add_argument("--max-new", type=int, default=120)
    args = ap.parse_args()

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from benchmark_quality_reference import PROMPT, load_problems  # noqa: E402

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 attn_implementation="eager")
    model.eval()
    repo = pathlib.Path(__file__).resolve().parents[2]
    problems = load_problems(pathlib.Path(args.dataset or repo.parent / "gsm8k_test_set.jsonl"), args.problems)
    for p in problems:
        if args.bare:
            ids = tok(p["question"], return_tensors="pt", add_special_tokens=False)
        else:
            text = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(question=p["question"])}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt")
        print(f"=== problem {p['index']} ({ids.input_ids.shape[1]} prompt tokens) gold {p['gold']}")
        for old in (False, True):
            modeling_llama.eager_attention_forward = make_eager(old)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                     output_logits=True, return_dict_in_generate=True)
            gen = tok.decode(out.sequences[0, ids.input_ids.shape[1]:], skip_special_tokens=False)
            nonfinite = [i for i, l in enumerate(out.logits) if not torch.isfinite(l).all()]
            first = nonfinite[0] if nonfinite else None
            print(f"--- {'OLD softmax (c7d5196 engine)' if old else 'correct softmax'}: "
                  f"{len(nonfinite)}/{len(out.logits)} steps with non-finite logits (first at step {first})\n{gen!r}\n")


if __name__ == "__main__":
    main()
