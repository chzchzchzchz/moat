#!/usr/bin/env python3
"""
End-to-end GSM8K accuracy for the native Metal engine.

What this is for: the project's central claim is that running N reasoning
channels in parallel on idle GPU lanes buys accuracy. Nothing in this repository
has measured that on this engine. The one artifact that reports it,
antigravity_benchmark_results.json, covers about five problems and states a
"20.0" accuracy lift with no sample count, no interval and no hardware. The
positive result quoted elsewhere (68.8% -> 74.2%, n=449) came from HuggingFace
running Qwen2.5-Math-1.5B, not from this engine at all.

So this measures the engine itself, and reports enough to judge the number:
every problem's record, both conditions' Wilson intervals, an exact paired test
on the same problems, and, when the result is null, how many problems would
actually have been needed. It will not print a lift that it cannot distinguish
from chance.

Both conditions come out of one generation call, so this costs the same as
running best-of-N alone, and the comparison is properly paired:

  baseline   channel 0 on its own   — one sample, what you get without the idea
  candidate  majority vote over N   — self-consistency across all N channels

Requires Apple Silicon, the built engine dylib, and model weights. Nothing here
runs on a CI runner; it is meant to be run on a device and to leave behind an
artifact that can be checked.

    python3 tools/benchmark_quality.py \
        --model-dir models/tinyllama \
        --dataset gsm8k_test_set.jsonl \
        --limit 200 --channels 8 \
        --out quality_gsm8k.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
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
)

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Let's think step by step, then give the final number after ####.\n"
    "Answer:"
)


def hardware_profile() -> dict:
    """Record the machine, so a number is never again quoted without its context."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if sys.platform == "darwin":
        for key, cmd in [("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                         ("memory_bytes", ["sysctl", "-n", "hw.memsize"])]:
            try:
                info[key] = subprocess.check_output(cmd, text=True).strip()
            except Exception as exc:                      # noqa: BLE001
                info[key] = f"unavailable: {exc}"
    return info


def load_problems(path: Path, limit: int) -> list:
    """Read GSM8K jsonl. A problem with no parseable gold answer is dropped here
    rather than counted as a failure of the model."""
    problems, skipped = [], 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if len(problems) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gold = extract_gold_answer(record.get("answer", ""))
            if gold is None:
                skipped += 1
                continue
            problems.append({"index": index, "question": record["question"], "gold": gold})
    if skipped:
        print(f"skipped {skipped} problem(s) with no parseable gold answer", file=sys.stderr)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True,
                        help="directory with the safetensors weights and tokenizer.json")
    parser.add_argument("--dataset", default=str(REPO.parent / "gsm8k_test_set.jsonl"))
    parser.add_argument("--limit", type=int, default=200,
                        help="problems to run; see the power note this prints on a null result")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--int4", action="store_true",
                        help="load weights as INT4 super-blocks (sets ANTIGRAVITY_INT4)")
    parser.add_argument("--out", default="quality_gsm8k.json")
    args = parser.parse_args()

    if args.channels < 2:
        print("--channels must be at least 2: with one channel the candidate and the "
              "baseline are the same run, and the comparison is vacuous", file=sys.stderr)
        return 2

    if args.int4:
        import os
        os.environ["ANTIGRAVITY_INT4"] = "1"

    model_dir = Path(args.model_dir).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_file():
        print(f"dataset not found: {dataset}", file=sys.stderr)
        return 2

    problems = load_problems(dataset, args.limit)
    if not problems:
        print(f"no usable problems in {dataset}", file=sys.stderr)
        return 2

    # Imported late so --help works on a machine with no engine.
    from native_bridge import NativeMetalEngine   # noqa: PLC0415
    from tokenizer import LlamaTokenizer          # noqa: PLC0415

    tokenizer = LlamaTokenizer(str(model_dir / "tokenizer.json"))
    engine = NativeMetalEngine(n_channels=args.channels)
    if not engine.load_weights(str(model_dir)):
        print(f"engine failed to load weights from {model_dir}", file=sys.stderr)
        return 1

    records, errors = [], []
    baseline_correct, candidate_correct = [], []
    started = time.time()

    for position, problem in enumerate(problems):
        prompt_ids = tokenizer.encode(PROMPT_TEMPLATE.format(question=problem["question"]))
        try:
            channel_tokens, logprobs, ttft_ms, total_ms = engine.generate(
                prompt_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:                          # noqa: BLE001
            # Record and keep going: one bad problem should not discard the run,
            # but the run must not be reported as clean either.
            errors.append({"problem_index": problem["index"], "error": repr(exc)})
            continue

        texts = [tokenizer.decode(tokens) for tokens in channel_tokens]
        answers = [extract_model_answer(text) for text in texts]
        selected = majority_vote(answers, scores=logprobs)

        base_ok = answers_match(answers[0] if answers else None, problem["gold"])
        cand_ok = answers_match(selected, problem["gold"])
        baseline_correct.append(base_ok)
        candidate_correct.append(cand_ok)

        records.append({
            "problem_index": problem["index"],
            "gold": problem["gold"],
            "channel_answers": answers,
            "channel_logprobs": list(logprobs),
            "channel_token_counts": [len(t) for t in channel_tokens],
            "baseline_answer": answers[0] if answers else None,
            "selected_answer": selected,
            "baseline_correct": base_ok,
            "candidate_correct": cand_ok,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "channel_texts": texts,
        })

        if (position + 1) % 10 == 0:
            done = len(baseline_correct)
            print(f"  {position + 1}/{len(problems)}  "
                  f"baseline {sum(baseline_correct)}/{done}  "
                  f"best-of-{args.channels} {sum(candidate_correct)}/{done}", flush=True)

    engine.destroy()

    if not records:
        print(f"every problem failed ({len(errors)} error(s)); nothing to report",
              file=sys.stderr)
        for err in errors[:5]:
            print(f"  {err['error']}", file=sys.stderr)
        return 1

    comparison = compare_conditions(baseline_correct, candidate_correct)

    result = {
        "benchmark": "gsm8k",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_profile(),
        "config": {
            "model_dir": str(model_dir),
            "dataset": str(dataset),
            "channels": args.channels,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "int4_weights": bool(args.int4),
            "baseline": "channel 0 alone",
            "candidate": f"majority vote over {args.channels} channels, "
                         f"ties broken by cumulative logprob",
        },
        "elapsed_seconds": time.time() - started,
        "comparison": comparison,
        "errors": errors,
        "records": records,
    }
    if not comparison["significant"]:
        # A null result is only meaningful next to the sample size it would have taken.
        result["power_note"] = {
            "problems_run": comparison["n_problems"],
            "problems_needed_for_5_point_effect": min_detectable_problems(0.05),
            "problems_needed_for_10_point_effect": min_detectable_problems(0.10),
            "comment": "at 80% power and alpha 0.05; a run smaller than this cannot "
                       "distinguish an effect of that size from chance either way",
        }

    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    base, cand = comparison["baseline"], comparison["candidate"]
    print()
    print(f"problems graded      {comparison['n_problems']}")
    print(f"baseline (1 channel) {base['correct']}/{comparison['n_problems']} = "
          f"{base['accuracy'] * 100:.1f}%  "
          f"[{base['ci95'][0] * 100:.1f}, {base['ci95'][1] * 100:.1f}]")
    print(f"best-of-{args.channels:<12} {cand['correct']}/{comparison['n_problems']} = "
          f"{cand['accuracy'] * 100:.1f}%  "
          f"[{cand['ci95'][0] * 100:.1f}, {cand['ci95'][1] * 100:.1f}]")
    print(f"verdict              {comparison['verdict']}")
    if "power_note" in result:
        note = result["power_note"]
        print(f"power                detecting a 5-point effect needs about "
              f"{note['problems_needed_for_5_point_effect']} problems; "
              f"this run had {note['problems_run']}")
    if errors:
        print(f"errors               {len(errors)} problem(s) failed and were excluded")
    print(f"written              {args.out}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
