#!/usr/bin/env python3
"""
Best-of-N on GSM8K with a PyTorch reference model — NOT the Metal engine.

Two jobs, and it is worth being exact about which.

1. It validates the grading machinery on real model output. src/quality_scoring.py
   decides what accuracy this project claims, and until now every test of it ran on
   synthetic strings like "#### 18". Real chain-of-thought output is messier: stray
   numbers after the answer, unfinished reasoning, no answer at all. If the
   extraction or the voting mishandles that, the engine's eventual number is wrong
   and nothing would say so.

2. It measures whether best-of-N helps at all, on these problems, with this scoring
   code — the baseline the engine will later be compared against, graded the same
   way rather than quoted from a different paper with a different model.

It is emphatically NOT a measurement of the Metal engine: it runs torch on the CPU.
The engine's own number comes from scripts/run_quality_benchmark.sh on a Mac.
Every output line says so, because a file like this one is exactly how an
engine-shaped claim gets made from a non-engine run.

    python3 tools/benchmark_quality_reference.py --limit 20 --samples 8
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quality_scoring import (  # noqa: E402
    answers_match,
    compare_conditions,
    extract_gold_answer,
    extract_model_answer,
    majority_vote,
    min_detectable_problems,
    sanity_checks,
    min_detectable_problems_paired,
    observed_discordance,
)

PROMPT = ("Solve the problem step by step, then give the final answer after ####.\n\n"
          "Problem: {question}\n\nSolution:")


def load_problems(path: Path, limit: int) -> list:
    problems = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(problems) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gold = extract_gold_answer(record.get("answer", ""))
            if gold is not None:
                problems.append({"index": index, "question": record["question"], "gold": gold})
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset", default=str(REPO.parent / "gsm8k_test_set.jsonl"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--samples", type=int, default=8,
                    help="samples per problem; stands in for the engine's channels")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="quality_gsm8k_reference.json")
    args = ap.parse_args()

    if args.samples < 2:
        print("--samples must be at least 2, or the two conditions are the same run",
              file=sys.stderr)
        return 2

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2
    problems = load_problems(dataset, args.limit)
    if not problems:
        print(f"no usable problems in {dataset}", file=sys.stderr)
        return 2

    import torch                                    # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: PLC0415

    torch.manual_seed(args.seed)
    print(f"loading {args.model} on CPU (this is the REFERENCE model, not the engine)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    records, errors = [], []
    baseline_correct, candidate_correct = [], []
    started = time.time()

    for position, problem in enumerate(problems):
        messages = [{"role": "user", "content": PROMPT.format(question=problem["question"])}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")

        try:
            with torch.no_grad():
                # One call with num_return_sequences=N, which is what the engine's N
                # channels are: N independent samples from the same prompt.
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.samples,
                    pad_token_id=tok.eos_token_id,
                )
        except Exception as exc:                     # noqa: BLE001
            errors.append({"problem_index": problem["index"], "error": repr(exc)})
            continue

        prompt_len = inputs["input_ids"].shape[1]
        generated = [seq[prompt_len:] for seq in out]
        texts = [tok.decode(g, skip_special_tokens=True) for g in generated]
        # A sample that used its whole budget did not stop on its own, so its last
        # number is an intermediate step rather than an answer. See
        # extract_model_answer: reading it anyway makes a cut-off sample vote.
        # Truncation must be judged per sample, by whether the model emitted EOS.
        # Length cannot do it: generate() with num_return_sequences pads every
        # sequence to the batch's longest, so a length test is uniformly true or
        # false for the whole batch. A first run marked all 8 samples truncated on
        # 35 of 50 problems for that reason, discarded every answer on those
        # problems, and produced exactly zero disagreements between the two
        # conditions — an impossible result that is what exposed it.
        #
        # pad_token_id is EOS here, but padding only lands on sequences that already
        # finished, so "EOS present" still means "stopped on its own".
        eos_id = tok.eos_token_id
        truncated = [eos_id not in g.tolist() for g in generated]
        answers = [extract_model_answer(t, truncated=c) for t, c in zip(texts, truncated)]
        selected = majority_vote(answers)

        base_ok = answers_match(answers[0] if answers else None, problem["gold"])
        cand_ok = answers_match(selected, problem["gold"])
        baseline_correct.append(base_ok)
        candidate_correct.append(cand_ok)

        records.append({
            "problem_index": problem["index"],
            "gold": problem["gold"],
            "sample_answers": answers,
            "baseline_answer": answers[0] if answers else None,
            "selected_answer": selected,
            "baseline_correct": base_ok,
            "candidate_correct": cand_ok,
            "n_samples_with_no_answer": sum(1 for a in answers if a is None),
            "n_samples_truncated": sum(1 for c in truncated if c),
            "sample_texts": texts,
        })

        done = len(baseline_correct)
        print(f"  {position + 1}/{len(problems)}  "
              f"baseline {sum(baseline_correct)}/{done}  "
              f"best-of-{args.samples} {sum(candidate_correct)}/{done}  "
              f"({time.time() - started:.0f}s)", flush=True)

    if not records:
        print(f"every problem failed ({len(errors)} error(s))", file=sys.stderr)
        return 1

    comparison = compare_conditions(baseline_correct, candidate_correct)
    no_answer = sum(r["n_samples_with_no_answer"] for r in records)
    total_samples = len(records) * args.samples

    result = {
        "benchmark": "gsm8k",
        "engine": "pytorch-cpu-reference",
        "IS_THIS_THE_METAL_ENGINE": False,
        "note": ("Reference measurement with a PyTorch model on CPU. This validates the "
                 "grading code in src/quality_scoring.py against real model output and "
                 "establishes the best-of-N baseline. It is NOT a measurement of the "
                 "Metal engine; for that run scripts/run_quality_benchmark.sh on a Mac."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "python": platform.python_version(), "device": "cpu"},
        "config": {
            "model": args.model,
            "dataset": str(dataset),
            "samples": args.samples,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "baseline": "sample 0 alone",
            "candidate": f"majority vote over {args.samples} samples",
        },
        "elapsed_seconds": time.time() - started,
        "comparison": comparison,
        "extraction": {
            "samples_with_no_extractable_answer": no_answer,
            "total_samples": total_samples,
            "fraction": no_answer / total_samples if total_samples else 0.0,
        },
        "errors": errors,
        "records": records,
    }
    if not comparison["significant"]:
        # A null result means nothing without the sample size it would have taken. The
        # figure is computed for McNemar's test, which is the test actually run, and
        # from the discordance this run observed rather than an assumed rate — the
        # two-sample formula ignores discordance and overstates the requirement by
        # roughly 2.5x, which invites the wrong conclusion that a feasible run is
        # pointless.
        discordance = observed_discordance(comparison)
        note = {
            "problems_run": comparison["n_problems"],
            "observed_discordance": discordance,
            "test": "exact McNemar (paired)",
        }
        for effect in (0.05, 0.10, 0.15):
            key = f"problems_needed_for_{int(effect * 100)}_point_effect"
            try:
                note[key] = min_detectable_problems_paired(effect, discordance)
            except ValueError as exc:
                # Happens when this run's discordance is too small to carry an effect
                # that size at all; saying so is more useful than a number.
                note[key] = f"not reachable at this discordance: {exc}"
        note["conservative_two_sample_bound_5_point"] = min_detectable_problems(0.05)
        result["power_note"] = note

    # A broken measurement and a null result read identically — "no difference
    # distinguishable from chance" says nothing about whether the method did nothing
    # or the harness discarded the data. These checks name the difference.
    warnings = sanity_checks(records, args.samples, comparison)
    result["sanity_warnings"] = warnings

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    base, cand = comparison["baseline"], comparison["candidate"]
    print()
    print("THIS IS THE PYTORCH CPU REFERENCE, NOT THE METAL ENGINE")
    print(f"model        {args.model}")
    print(f"graded       {comparison['n_problems']} problems in {result['elapsed_seconds']:.0f}s")
    print(f"baseline     {base['correct']}/{comparison['n_problems']} = {base['accuracy']*100:.1f}%"
          f"  [{base['ci95'][0]*100:.1f}, {base['ci95'][1]*100:.1f}]")
    print(f"best-of-{args.samples:<4} {cand['correct']}/{comparison['n_problems']} = {cand['accuracy']*100:.1f}%"
          f"  [{cand['ci95'][0]*100:.1f}, {cand['ci95'][1]*100:.1f}]")
    print(f"verdict      {comparison['verdict']}")
    print(f"extraction   {no_answer}/{total_samples} samples produced no number "
          f"({no_answer/total_samples*100:.1f}%)")
    if errors:
        print(f"errors       {len(errors)} problem(s) failed")
    if warnings:
        print()
        print("DO NOT TRUST THIS RESULT — the run itself looks wrong:")
        for w in warnings:
            print(f"  - {w}")
    print(f"written      {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
