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


def extract_model_answer(text: str, truncated: bool = False) -> Optional[float]:
    """The model's final answer, or None when it did not produce one.

    Order: the number after '####' when the model followed the format; otherwise the
    last number in the text, which is the standard GSM8K convention.

    `truncated` says the generation hit its token limit rather than stopping on its
    own, and it changes the answer. The last-number fallback assumes the text ran to
    a conclusion; on a cut-off generation the last number is whatever intermediate
    step it happened to reach, and returning that turns "no answer" into a confident
    wrong one. Measured on real output rather than supposed: of eight samples for one
    GSM8K problem, two were cut off mid-sentence and the fallback read 9 and 10 out
    of their working, while the single sample that finished said 18, the correct
    answer. The majority vote then chose 9. A no-answer does not vote; a fabricated
    one does, so this silently corrupted the selection and understated accuracy.

    A '####' marker is still honoured when truncated, because the model did emit its
    answer before running out of room.
    """
    marker = text.rfind("####")
    if marker >= 0:
        match = _NUMBER.search(text[marker + 4:])
        if match:
            return parse_number(match.group(0))

    if truncated:
        return None

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
    """Problems needed to detect `delta` if the two conditions were INDEPENDENT.

    This is the two-sample formula, n = 2p(1-p)(z_a + z_b)^2 / delta^2, and it is
    the wrong test for this design: both conditions are graded on the same problems,
    so the comparison is paired and McNemar's test needs substantially fewer. Kept
    because it is a valid conservative upper bound, and because a caller that does
    not yet know its discordance rate has nothing better to use.

    Prefer min_detectable_problems_paired() once a run has produced a discordance.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    z_alpha, z_beta = 1.959963985, 0.8416212336
    variance = 2 * base_rate * (1 - base_rate)
    return max(1, math.ceil(variance * (z_alpha + z_beta) ** 2 / (delta ** 2)))


def min_detectable_problems_paired(delta: float, discordance: float,
                                   alpha: float = 0.05, power: float = 0.8) -> int:
    """Problems needed for McNemar's test to detect a net difference of `delta`.

    This is the test compare_conditions() actually runs, so this is the number a null
    result should be read against. It needs `discordance`: the fraction of problems
    the two conditions answer differently. That is what McNemar's power turns on and
    what the two-sample formula ignores — a method that changes many answers needs
    more problems for the same net shift than one that changes few, because the extra
    disagreements are noise the test has to see past.

        n = (z_a * sqrt(p_d) + z_b * sqrt(p_d - delta^2))^2 / delta^2

    Checked against the two-sample bound: for delta = 0.10 at 20% discordance this
    gives 155 where the two-sample formula demands 393, and a simulated run with that
    discordance at n = 100 already reaches p = 0.041. Over-stating the requirement
    four-fold is not harmless — it invites the conclusion that a feasible run is
    pointless.

    Raises ValueError when the discordance is too small to carry the claimed
    difference: the net shift cannot exceed the disagreements that produce it.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must be in (0, 1]")
    if discordance <= delta:
        raise ValueError(
            f"a net difference of {delta:.3f} cannot arise from a discordance of "
            f"{discordance:.3f}: every net gain is a disagreement, so discordance "
            f"must exceed it")

    z_alpha, z_beta = 1.959963985, 0.8416212336
    numerator = (z_alpha * math.sqrt(discordance)
                 + z_beta * math.sqrt(discordance - delta * delta)) ** 2
    return max(1, math.ceil(numerator / (delta * delta)))


def observed_discordance(comparison: Dict) -> float:
    """The discordance a completed comparison actually showed.

    Feeding this back into min_detectable_problems_paired() answers the question a
    null result raises — how much more data would settle this — using the run's own
    behaviour rather than an assumed rate.
    """
    paired = comparison["paired"]
    n = comparison["n_problems"]
    if n == 0:
        return 0.0
    return (paired["only_candidate_correct"] + paired["only_baseline_correct"]) / n
