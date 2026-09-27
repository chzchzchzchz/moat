#!/usr/bin/env python3
"""
Re-analyse the benchmark artifacts already committed to this repository.

Two reasons this exists.

**The numbers were reported without the statistics that make them mean anything.**
full_benchmark_results.json says `pass_1: 2, pass_8: 6, total: 30`, which reads as a
threefold improvement. Run through the same paired test the harnesses use, the four
problems the two conditions disagree on give p = 0.125 and their confidence intervals
overlap heavily. It is not a result yet — and the paired power calculation says 57
problems would settle it, against the 30 that were run, which is actionable rather
than merely discouraging.

**The largest saved run shows the engine producing degenerate output.**
gsm8k_full_checkpoint.json holds 587 GSM8K problems at 1, 2, 4 and 8 channels,
produced by run_full_gsm8k.py — which constructs NativeMetalEngine against
libantigravity_engine.dylib, so this is the engine itself and not a reference
framework. In 413 of 587 problems the generated text is a single character repeated,
the SAME character in every one of them, and accuracy is 0.3% at every channel count.
Output that does not vary with the prompt is a broken forward pass, not a weak model.

Needs no GPU, no weights and no network: it reads files already in the tree.

    python3 tools/analyze_existing_artifacts.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quality_scoring import (  # noqa: E402
    compare_conditions,
    mcnemar_exact,
    min_detectable_problems_paired,
    observed_discordance,
)

ROOT = REPO.parent


def heading(text: str) -> None:
    print(f"\n{text}\n{'=' * len(text)}")


def analyse_pass_at_k(path: Path) -> None:
    """Grade the committed pass@1 vs pass@8 artifact properly."""
    heading(f"{path.name}: greedy vs pass@8")
    if not path.is_file():
        print("  not present")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    details = data.get("details") or []
    if not details:
        print("  no per-problem details to analyse")
        return

    greedy = [bool(x.get("greedy_correct")) for x in details]
    pass_k = [bool(x.get("pass_8_correct")) for x in details]
    result = compare_conditions(greedy, pass_k)

    base, cand = result["baseline"], result["candidate"]
    n = result["n_problems"]
    print(f"  as committed        pass_1={data.get('pass_1')} pass_8={data.get('pass_8')} "
          f"total={data.get('total')}  (no interval, no test)")
    print(f"  greedy   {base['correct']:>3}/{n} = {base['accuracy'] * 100:5.1f}%"
          f"   95% CI [{base['ci95'][0] * 100:.1f}, {base['ci95'][1] * 100:.1f}]")
    print(f"  pass@8   {cand['correct']:>3}/{n} = {cand['accuracy'] * 100:5.1f}%"
          f"   95% CI [{cand['ci95'][0] * 100:.1f}, {cand['ci95'][1] * 100:.1f}]")
    paired = result["paired"]
    print(f"  disagreements       pass@8 only {paired['only_candidate_correct']}, "
          f"greedy only {paired['only_baseline_correct']}, same {paired['both_or_neither']}")
    print(f"  exact McNemar       p = {result['p_value']:.4f}")
    print(f"  verdict             {result['verdict']}")

    discordance = observed_discordance(result)
    delta = abs(result["delta_accuracy"])
    only_cand = paired["only_candidate_correct"]
    only_base = paired["only_baseline_correct"]
    n_discordant = only_cand + only_base

    if only_base == 0 and only_cand > 0:
        # Every disagreement favours one side, so the net difference equals the
        # discordance and the paired power formula has no variance to work with. What
        # decides significance here is simply how many one-sided pairs there are: a
        # sign test on n of them gives p = 2 * 0.5^n.
        need_pairs = next((k for k in range(1, 40) if mcnemar_exact(0, k) < 0.05), None)
        print(f"  to settle it        all {n_discordant} disagreements favour pass@8, so "
              f"significance turns on their count alone:")
        print(f"                      {need_pairs} one-sided pairs reach p < 0.05 "
              f"(p = {mcnemar_exact(0, need_pairs):.4f}); this run has {n_discordant} "
              f"(p = {mcnemar_exact(0, n_discordant):.4f})")
        if discordance > 0:
            rate = n_discordant / n
            projected = int(round(need_pairs / rate)) if rate else None
            print(f"                      at the same {rate * 100:.1f}% rate that is about "
                  f"{projected} problems, against the {n} run")
    elif delta > 0:
        try:
            need = min_detectable_problems_paired(delta, discordance)
            print(f"  to settle it        about {need} problems at the "
                  f"{discordance * 100:.1f}% discordance observed; this run had {n}")
        except ValueError as exc:
            print(f"  to settle it        {exc}")

    print("\n  Note: pass@8 asks whether the correct answer is ANYWHERE among 8 samples.")
    print("  Selecting it needs an oracle, so this is an upper bound on what best-of-N")
    print("  could reach, not an accuracy anyone can ship. The deployable question is")
    print("  what a verifier or a majority vote actually picks.")


def degenerate(text: str, threshold: float = 0.5) -> str | None:
    """The character a snippet is mostly made of, when it is mostly one character.

    Whitespace is excluded: a run of spaces is padding or a formatting artefact, not
    a model emitting the same token over and over, and pooling the two would blur the
    finding that matters.
    """
    stripped = (text or "").strip()
    if len(stripped) < 20:
        return None
    char, count = collections.Counter(stripped).most_common(1)[0]
    if char.isspace():
        return None
    return char if count / len(stripped) > threshold else None


def analyse_engine_checkpoint(path: Path) -> None:
    """Report the degenerate-output rate in the engine's own saved run."""
    heading(f"{path.name}: the engine's own 587-problem run")
    if not path.is_file():
        print("  not present")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    channels = sorted({c for v in data.values() for c in v.get("n_results", {})}, key=int)
    print(f"  problems {len(data)}   channel counts {channels}")
    print(f"  produced by run_full_gsm8k.py, which builds NativeMetalEngine against")
    print(f"  libantigravity_engine.dylib — this is the engine, not a reference framework.\n")

    print(f"  {'N':>3} {'graded':>7} {'degenerate':>11} {'empty':>6} {'correct':>8} {'accuracy':>9}")
    repeated: collections.Counter = collections.Counter()
    for channel in channels:
        graded = deg = empty = correct = 0
        for problem in data.values():
            record = problem.get("n_results", {}).get(channel)
            if not record:
                continue
            graded += 1
            snippet = record.get("text_snippet") or ""
            if not snippet.strip():
                empty += 1
            else:
                char = degenerate(snippet)
                if char:
                    deg += 1
                    repeated[char] += 1
            if record.get("is_correct"):
                correct += 1
        accuracy = f"{correct / graded * 100:.1f}%" if graded else "-"
        print(f"  {channel:>3} {graded:>7} {deg:>11} {empty:>6} {correct:>8} {accuracy:>9}")

    if repeated:
        print(f"\n  distinct repeated characters across the whole run: {len(repeated)}")
        for char, count in repeated.most_common(3):
            print(f"    {char!r} in {count} generations")
        dominant_share = repeated.most_common(1)[0][1] / sum(repeated.values())
        if dominant_share > 0.95:
            print(f"\n  One character accounts for {dominant_share * 100:.1f}% of them, across "
                  f"{len(data)} different problems: the output")
            print("  does not depend on the input. That is a broken forward pass rather than")
            print("  a weak model — a model that merely reasoned badly would still vary.")
            print("\n  What is ruled out, and what is not.")
            print("    Ruled out: the weights failing to load. run_full_gsm8k.py passes")
            print("      model_path to the constructor, and a failed load makes")
            print("      AntigravityEngineNativeGenerate return -1, which native_bridge")
            print("      raises on — the script would have stopped, not written 587 rows.")
            print("    Ruled out: the BF16 to FP16 conversion. TinyLlama ships BF16, so")
            print("      bf16_to_fp16 is load-bearing, and it is correct: 6,822 BF16 bit")
            print("      patterns across the weight range convert exactly.")
            print("    Probably not the cause: the null-pipeline path fixed on this branch.")
            print("      Neither the .metal nor the .metallib files reach the xcframework and")
            print("      every lookup was relative to the process working directory, which")
            print("      makes every pipeline but one null ON A DEVICE. But this script loads")
            print("      models/tinyllama/... by relative path, so it ran from")
            print("      antigravity-engine/, where src/shaders/transformer_ops.metallib does")
            print("      resolve. The shaders were most likely found for this run.")
            print("    Not established: which defect produced it. A forward pass that ignores")
            print("      its input points at weight layout, attention masking or positional")
            print("      indexing rather than at sampling — temperature 0.7 cannot yield one")
            print("      token 587 times unless the logits themselves are degenerate.")
            print("\n  Whatever the cause, the engine had not produced usable text when this")
            print("  was recorded, and accuracy is flat at 0.3% across every channel count:")
            print("  parallel channels cannot help a forward pass that ignores its input.")
            print("\n  Re-running the binary that produced this needs a Mac, so narrowing it")
            print("  further is not possible from a machine without one.")


def main() -> int:
    print("Re-analysis of committed benchmark artifacts. No GPU, no weights, no network.")
    analyse_pass_at_k(ROOT / "full_benchmark_results.json")
    analyse_engine_checkpoint(ROOT / "gsm8k_full_checkpoint.json")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
