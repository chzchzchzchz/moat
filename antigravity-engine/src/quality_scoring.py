"""
Scoring and statistics for end-to-end quality benchmarks.

Kept free of the engine, torch and Metal on purpose: this is the code that
decides what number gets claimed, so it has to be testable on any machine.

The repository already contains antigravity_benchmark_results.json, which
reports "accuracy_lift_pct: 20.0" for 8 channels. That run covered about five
problems. One problem either way moves such a number by 20 points, so it is not
evidence of anything, and nothing in the file says so. Everything here exists to
make that failure mode impossible to repeat: an accuracy is never reported
without its interval, and a difference is never called a lift unless a paired
test on the same problems says it can be distinguished from chance.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

# A number, optionally signed, with thousands separators and a decimal part.
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def parse_number(text: str) -> Optional[float]:
    """Parse one numeric token, tolerating $ and thousands separators."""
    cleaned = text.replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_gold_answer(answer_field: str) -> Optional[float]:
    """GSM8K puts the gold answer after '####' on the final line."""
    marker = answer_field.rfind("####")
    if marker < 0:
        return None
    match = _NUMBER.search(answer_field[marker + 4:])
    return parse_number(match.group(0)) if match else None


def extract_model_answer(text: str) -> Optional[float]:
    """The model's final answer: after '####' when it follows the format, else the
    last number it wrote. Returns None when it produced no number at all, which is
    scored as wrong rather than quietly skipped."""
    marker = text.rfind("####")
    if marker >= 0:
        match = _NUMBER.search(text[marker + 4:])
        if match:
            return parse_number(match.group(0))
    matches = _NUMBER.findall(text)
    return parse_number(matches[-1]) if matches else None


def answers_match(a: Optional[float], b: Optional[float], tol: float = 1e-4) -> bool:
    """No answer is never a match, including against another no-answer."""
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def majority_vote(answers: Sequence[Optional[float]],
                  scores: Optional[Sequence[float]] = None) -> Optional[float]:
    """Self-consistency selection over N channels.

    Channels that produced no number do not vote — they are not evidence for any
    answer. Ties break on summed score (cumulative logprob) when scores are given,
    and otherwise on the first answer reached, so the result never depends on
    dict ordering.
    """
    usable = [(i, a) for i, a in enumerate(answers) if a is not None]
    if not usable:
        return None

    # Group on a rounded key so 18.0 and 18.000001 are one candidate.
    def key(value: float) -> float:
        return round(value, 6)

    counts: Counter = Counter(key(a) for _, a in usable)
    best_count = max(counts.values())
    tied = [value for value, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]

    if scores is not None:
        totals = {value: sum(scores[i] for i, a in usable if key(a) == value)
                  for value in tied}
        return max(tied, key=lambda v: totals[v])

    for _, a in usable:
        if key(a) in tied:
            return key(a)
    return tied[0]


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the textbook normal interval because these runs are small
    and accuracies land near 0 and 1, where the normal interval runs outside
    [0, 1] and badly under-covers.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes.

    The two conditions are graded on the same problems, so the problems both get
    right and both get wrong carry no information about which is better. Only the
    disagreements do: under the null they split 50/50, which is a binomial test
    on only_a out of only_a + only_b.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0

    def binom_pmf(k: int) -> float:
        return math.comb(n, k) * (0.5 ** n)

    observed = binom_pmf(min(only_a, only_b))
    # Sum every outcome no more likely than the observed one. The distribution is
    # symmetric, so a small tolerance keeps floating point from dropping the
    # mirror-image term.
    return min(1.0, sum(binom_pmf(k) for k in range(n + 1)
                        if binom_pmf(k) <= observed * (1.0 + 1e-9)))


def compare_conditions(baseline: Sequence[bool], candidate: Sequence[bool],
                       alpha: float = 0.05) -> Dict:
    """Compare two graded conditions over the same problems, in order.

    Returns the accuracies with intervals, the paired disagreement counts, the
    exact p-value, and a verdict that refuses to call a difference a lift when
    the test cannot distinguish it from chance.
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"paired comparison needs equal lengths, "
                         f"got {len(baseline)} and {len(candidate)}")
    n = len(baseline)
    if n == 0:
        raise ValueError("nothing to compare: zero problems were graded")

    base_correct = sum(1 for x in baseline if x)
    cand_correct = sum(1 for x in candidate if x)
    only_candidate = sum(1 for b, c in zip(baseline, candidate) if c and not b)
    only_baseline = sum(1 for b, c in zip(baseline, candidate) if b and not c)

    p_value = mcnemar_exact(only_baseline, only_candidate)
    delta = (cand_correct - base_correct) / n
    significant = p_value < alpha

    if significant:
        verdict = (f"candidate {'beats' if delta > 0 else 'loses to'} baseline: "
                   f"{delta * 100:+.1f} points over {n} problems, p = {p_value:.4f}")
    else:
        verdict = (f"no difference distinguishable from chance: {delta * 100:+.1f} "
                   f"points over {n} problems, p = {p_value:.4f} "
                   f"(the {only_baseline + only_candidate} problems the two "
                   f"conditions disagree on are consistent with a coin flip)")

    return {
        "n_problems": n,
        "baseline": {
            "correct": base_correct,
            "accuracy": base_correct / n,
            "ci95": wilson_interval(base_correct, n),
        },
        "candidate": {
            "correct": cand_correct,
            "accuracy": cand_correct / n,
            "ci95": wilson_interval(cand_correct, n),
        },
        "paired": {
            "only_candidate_correct": only_candidate,
            "only_baseline_correct": only_baseline,
            "both_or_neither": n - only_candidate - only_baseline,
        },
        "delta_accuracy": delta,
        "p_value": p_value,
        "alpha": alpha,
        "significant": significant,
        "verdict": verdict,
    }


def min_detectable_problems(delta: float, base_rate: float = 0.5,
                            alpha: float = 0.05, power: float = 0.8) -> int:
    """Roughly how many problems are needed to detect `delta` at this power.

    Printed alongside a null result so "we saw nothing" can be read as either
    "there is nothing" or "this run was far too small to tell", which is the
    distinction the existing five-problem artifact leaves out.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    z_alpha, z_beta = 1.959963985, 0.8416212336
    variance = 2 * base_rate * (1 - base_rate)
    return max(1, math.ceil(variance * (z_alpha + z_beta) ** 2 / (delta ** 2)))
