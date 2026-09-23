"""
Tests for src/quality_scoring.py — the code that decides what accuracy number
gets claimed, and whether a difference may be called a lift.

Worth testing carefully because the failure mode is silent: a scoring bug does
not raise, it just reports a different number, and a benchmark result is exactly
the thing nobody can sanity-check by eye. The repo already contains
antigravity_benchmark_results.json claiming a 20-point lift measured over about
five problems, which is what these tests exist to stop recurring.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quality_scoring import (  # noqa: E402
    answers_match,
    compare_conditions,
    extract_gold_answer,
    extract_model_answer,
    majority_vote,
    mcnemar_exact,
    min_detectable_problems,
    parse_number,
    wilson_interval,
)


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------

def test_gold_answer_comes_from_the_hash_marker():
    answer = ("She sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
              "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18")
    assert extract_gold_answer(answer) == 18.0


def test_gold_answer_ignores_the_working_above_it():
    # Every intermediate number appears before ####; only the last one counts.
    assert extract_gold_answer("70000-130000=<<200000-130000=70000>>70,000\n#### 70000") == 70000.0


def test_gold_answer_is_none_without_a_marker():
    assert extract_gold_answer("the answer is 42") is None


def test_model_answer_prefers_the_hash_marker_over_the_last_number():
    text = "Step 1: 5 + 5 = 10.\n#### 10\nThanks for reading, see you in 2026."
    assert extract_model_answer(text) == 10.0


def test_model_answer_falls_back_to_the_last_number():
    assert extract_model_answer("First 3, then 4, so the total is 7.") == 7.0


def test_model_answer_is_none_when_no_number_was_produced():
    # A channel that rambled without answering must score wrong, not be skipped.
    assert extract_model_answer("I am not sure how to solve this.") is None


@pytest.mark.parametrize("text,expected", [
    ("$1,234", 1234.0),
    ("-5", -5.0),
    ("3.5", 3.5),
    ("1,000,000", 1000000.0),
    ("42.", 42.0),
])
def test_parse_number_handles_currency_separators_and_signs(text, expected):
    assert parse_number(text) == expected


def test_no_answer_never_matches_even_another_no_answer():
    assert not answers_match(None, None)
    assert not answers_match(None, 1.0)
    assert not answers_match(1.0, None)


def test_answers_match_tolerates_float_representation():
    assert answers_match(18.0, 18.000001)
    assert not answers_match(18.0, 18.1)


# --------------------------------------------------------------------------
# Majority vote
# --------------------------------------------------------------------------

def test_majority_vote_picks_the_most_common_answer():
    assert majority_vote([7.0, 18.0, 18.0, 5.0, 18.0]) == 18.0


def test_channels_without_an_answer_do_not_vote():
    # None is not evidence for anything; two real votes must beat three silences.
    assert majority_vote([None, None, None, 18.0, 18.0]) == 18.0


def test_majority_vote_is_none_when_nothing_answered():
    assert majority_vote([None, None]) is None


def test_ties_break_on_cumulative_logprob_when_scores_are_given():
    # Two answers, two votes each; the higher summed score wins.
    answers = [3.0, 3.0, 9.0, 9.0]
    assert majority_vote(answers, scores=[-1.0, -1.0, -0.1, -0.1]) == 9.0
    assert majority_vote(answers, scores=[-0.1, -0.1, -1.0, -1.0]) == 3.0


def test_tie_without_scores_is_deterministic():
    # Must not depend on dict iteration order; same input, same answer, always.
    answers = [5.0, 9.0]
    assert majority_vote(answers) == majority_vote(answers) == 5.0


def test_near_identical_floats_count_as_one_candidate():
    assert majority_vote([18.0, 18.0000001, 3.0]) == 18.0


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high


def test_wilson_interval_stays_inside_zero_and_one_at_the_edges():
    # Where the textbook normal interval goes negative or above 1.
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-3))
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and low > 0.6


def test_wilson_interval_narrows_as_n_grows():
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0]) / 5


def test_wilson_interval_of_nothing_is_degenerate_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_five_problem_run_cannot_support_a_twenty_point_claim():
    # The scenario in antigravity_benchmark_results.json: 1/5 vs 0/5 reported as
    # a 20-point lift. The interval alone shows the claim is unsupportable.
    low, high = wilson_interval(1, 5)
    assert high - low > 0.5


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

def test_no_disagreements_means_no_evidence():
    assert mcnemar_exact(0, 0) == 1.0


def test_symmetric_disagreement_is_not_significant():
    assert mcnemar_exact(5, 5) == 1.0


def test_p_value_is_symmetric_in_its_arguments():
    assert mcnemar_exact(2, 9) == pytest.approx(mcnemar_exact(9, 2))


def test_one_sided_disagreement_becomes_significant_only_with_enough_of_them():
    # 5 of 5 one way is the sign-test floor at 0.0625 — just short of 0.05.
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(0, 6) == pytest.approx(0.03125)
    assert mcnemar_exact(0, 5) > 0.05
    assert mcnemar_exact(0, 6) < 0.05


def test_p_value_is_a_probability():
    for a in range(0, 12):
        for b in range(0, 12):
            assert 0.0 <= mcnemar_exact(a, b) <= 1.0


# --------------------------------------------------------------------------
# compare_conditions — the verdict
# --------------------------------------------------------------------------

def test_a_small_run_refuses_to_claim_a_lift():
    # 1/5 vs 0/5: the exact shape of the existing artifact's 20-point claim.
    result = compare_conditions([False] * 5, [True] + [False] * 4)
    assert result["delta_accuracy"] == pytest.approx(0.2)
    assert not result["significant"]
    assert "distinguishable from chance" in result["verdict"]


def test_a_large_consistent_improvement_is_reported_as_one():
    baseline = [False] * 100
    candidate = [True] * 30 + [False] * 70
    result = compare_conditions(baseline, candidate)
    assert result["significant"]
    assert result["p_value"] < 0.001
    assert "beats" in result["verdict"]


def test_a_real_regression_is_named_as_one():
    baseline = [True] * 30 + [False] * 70
    candidate = [False] * 100
    result = compare_conditions(baseline, candidate)
    assert result["significant"]
    assert result["delta_accuracy"] < 0
    assert "loses to" in result["verdict"]


def test_equal_accuracy_from_different_problems_is_still_no_difference():
    # Both score 50/100, disagreeing on 100 problems. Same accuracy, no lift.
    baseline = [True, False] * 50
    candidate = [False, True] * 50
    result = compare_conditions(baseline, candidate)
    assert result["baseline"]["accuracy"] == result["candidate"]["accuracy"]
    assert not result["significant"]
    assert result["paired"]["only_candidate_correct"] == 50
    assert result["paired"]["only_baseline_correct"] == 50


def test_paired_counts_partition_the_problems():
    baseline = [True, True, False, False, True]
    candidate = [True, False, True, False, True]
    result = compare_conditions(baseline, candidate)
    paired = result["paired"]
    assert (paired["only_candidate_correct"] + paired["only_baseline_correct"]
            + paired["both_or_neither"]) == result["n_problems"] == 5


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="equal lengths"):
        compare_conditions([True, False], [True])


def test_an_empty_run_is_refused_rather_than_scored_as_zero():
    with pytest.raises(ValueError, match="zero problems"):
        compare_conditions([], [])


# --------------------------------------------------------------------------
# Power
# --------------------------------------------------------------------------

def test_detecting_a_small_effect_needs_many_more_problems():
    assert min_detectable_problems(0.20) < min_detectable_problems(0.05)
    # A 5-point effect at 80% power needs samples in the high hundreds.
    assert min_detectable_problems(0.05) > 500


def test_five_problems_is_nowhere_near_enough_for_a_twenty_point_effect():
    assert min_detectable_problems(0.20) > 5


def test_a_nonpositive_effect_size_is_refused():
    with pytest.raises(ValueError):
        min_detectable_problems(0.0)


def test_returns_a_whole_number_of_problems():
    n = min_detectable_problems(0.1)
    assert isinstance(n, int) and n == math.ceil(n)
