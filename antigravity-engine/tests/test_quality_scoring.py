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
    min_detectable_problems_paired,
    observed_discordance,
    parse_number,
    sanity_checks,
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


def test_a_truncated_generation_with_no_marker_yields_no_answer():
    """The bug this was written for, in the shape it actually occurred.

    Real output from Qwen2.5-0.5B on the Janet's-ducks problem, cut off at the token
    limit mid-sentence. The last number is 9, an intermediate step; the model never
    stated an answer.
    """
    cut_off = ("The remaining eggs sold are: 16 - 7 = 9. "
               "Finally, we calculate the revenue from selling these remaining eggs at the farmers'")
    assert extract_model_answer(cut_off, truncated=True) is None
    # Without the truncation flag the old behaviour stands, which is what made two
    # cut-off samples out-vote the one that finished and said 18.
    assert extract_model_answer(cut_off, truncated=False) == 9.0


def test_a_truncated_generation_that_did_state_its_answer_is_still_read():
    # The marker means the answer was emitted before the budget ran out.
    assert extract_model_answer("working... #### 18\nand then it was cut o",
                                truncated=True) == 18.0


def test_a_complete_generation_still_uses_the_last_number():
    # Real completed output: no #### marker, answer stated in prose.
    done = ("The number of eggs left for selling is 16 - 7 = 9. Since each egg is sold "
            "for $2, she earns 9 x 2 = $18 every day. Final Answer: Janet makes $18.")
    assert extract_model_answer(done, truncated=False) == 18.0


def test_truncation_only_suppresses_the_fallback_not_a_real_answer():
    # A truncated sample with no numbers at all was already None; stays None.
    assert extract_model_answer("Let me think about this problem care", truncated=True) is None


def test_truncated_samples_stop_corrupting_the_majority_vote():
    """End to end on the real case: three samples, two cut off, one complete.

    Before: the two cut-off samples contributed 9 and 10 from their working, so the
    vote landed on an intermediate value. After: they abstain and the one sample that
    actually answered decides it.
    """
    texts = [
        "eggs sold are 16 - 7 = 9. Finally we calculate the revenue at the farmers'",
        "the number of eggs sold is 16 - 6 = 10. Therefore the total revenue is 10",
        "she earns 9 x 2 = $18 every day. Final Answer: Janet makes $18 every day.",
    ]
    truncated = [True, True, False]

    fixed = majority_vote([extract_model_answer(t, truncated=c)
                           for t, c in zip(texts, truncated)])
    assert fixed == 18.0

    unfixed = majority_vote([extract_model_answer(t, truncated=False) for t in texts])
    assert unfixed != 18.0


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


def test_the_paired_test_needs_far_fewer_problems_than_two_independent_groups():
    """The reason the paired calculation exists.

    compare_conditions() runs McNemar on the same problems, which is much more
    sensitive than comparing two independent groups. Using the two-sample formula to
    describe it overstated the requirement roughly 2.5x, and an inflated figure
    invites the wrong conclusion — that a feasible run is not worth doing.
    """
    two_sample = min_detectable_problems(0.10)
    paired = min_detectable_problems_paired(0.10, discordance=0.20)
    assert paired < two_sample / 2
    assert (two_sample, paired) == (393, 155)


def test_more_disagreement_needs_more_problems_for_the_same_net_shift():
    # This is what the two-sample formula cannot express: the extra disagreements are
    # noise the test must see past, so a noisier method needs a bigger sample.
    low = min_detectable_problems_paired(0.10, discordance=0.12)
    high = min_detectable_problems_paired(0.10, discordance=0.50)
    assert low < high


def test_an_effect_larger_than_the_disagreement_is_refused():
    # Every net gain is a disagreement, so a 20-point shift cannot come out of 10%
    # discordance. Returning a number here would be nonsense.
    with pytest.raises(ValueError, match="cannot arise from a discordance"):
        min_detectable_problems_paired(0.20, discordance=0.10)
    # Equality is refused too: it would require every disagreement to favour one side.
    with pytest.raises(ValueError, match="must exceed it"):
        min_detectable_problems_paired(0.20, discordance=0.20)


@pytest.mark.parametrize("discordance", [0.0, -0.1, 1.5])
def test_an_impossible_discordance_is_refused(discordance):
    with pytest.raises(ValueError, match="discordance must be in"):
        min_detectable_problems_paired(0.05, discordance)


def test_observed_discordance_reads_the_comparison_back():
    # 8 problems, candidate wins 2, baseline wins 1, 5 agree -> 3/8 discordant.
    baseline  = [True, True, False, False, False, True, True, False]
    candidate = [True, False, True, True, False, True, True, False]
    result = compare_conditions(baseline, candidate)
    assert result["paired"]["only_candidate_correct"] == 2
    assert result["paired"]["only_baseline_correct"] == 1
    assert observed_discordance(result) == pytest.approx(3 / 8)


def test_observed_discordance_is_zero_when_the_conditions_never_differ():
    result = compare_conditions([True, False, True], [True, False, True])
    assert observed_discordance(result) == 0.0


def test_the_paired_figure_beats_the_two_sample_one_across_the_range():
    # Sanity across plausible effect sizes, so the relationship is not an artefact of
    # the one case the first test pins.
    for effect in (0.05, 0.10, 0.15):
        assert min_detectable_problems_paired(effect, 0.20) < min_detectable_problems(effect)


def test_a_nonpositive_effect_size_is_refused():
    with pytest.raises(ValueError):
        min_detectable_problems(0.0)


def test_returns_a_whole_number_of_problems():
    n = min_detectable_problems(0.1)
    assert isinstance(n, int) and n == math.ceil(n)


# --------------------------------------------------------------------------
# sanity_checks — telling a broken measurement from a null one
#
# These exist because a real run reported 18.0% for both conditions with zero
# disagreements across 50 problems, and nothing objected. The verdict read "no
# difference distinguishable from chance", which is what a working harness says
# when the method does nothing — so the failure was invisible. The cause was
# truncation judged from generate()'s padded sequence length: every sample in a
# batch shares the padded length, so all 8 were marked truncated whenever one hit
# the cap, and every answer on 35 of 50 problems was discarded.
# --------------------------------------------------------------------------

def _records(truncated_counts, no_answer_counts=None):
    no_answer_counts = no_answer_counts or [0] * len(truncated_counts)
    return [{"n_samples_truncated": t, "n_samples_with_no_answer": a}
            for t, a in zip(truncated_counts, no_answer_counts)]


def _comparison(only_cand=0, only_base=0, n=50, both=0):
    baseline = [False] * n
    candidate = [False] * n
    for i in range(n - both, n):
        baseline[i] = candidate[i] = True
    for i in range(only_cand):
        candidate[i] = True
    for i in range(only_cand, only_cand + only_base):
        baseline[i] = True
    return compare_conditions(baseline, candidate)


def test_all_or_nothing_truncation_is_flagged_as_batch_level():
    # The observed pattern: 0 or 8, never in between, over 50 problems.
    counts = [8 if i % 3 else 0 for i in range(50)]
    warnings = sanity_checks(_records(counts), 8, _comparison(only_cand=5, only_base=3))
    assert any("batch-level signal" in w for w in warnings)


def test_truncation_that_varies_within_a_batch_is_not_flagged():
    # What correct per-sample detection looks like: intermediate counts appear.
    counts = [0, 3, 8, 1, 5, 8, 0, 2, 6, 4] * 5
    warnings = sanity_checks(_records(counts), 8, _comparison(only_cand=5, only_base=3))
    assert not any("batch-level signal" in w for w in warnings)


def test_no_truncation_anywhere_is_not_flagged():
    # All-zero is uniform but benign: nothing was cut off. Flagging it would cry wolf
    # on every short-output run.
    warnings = sanity_checks(_records([0] * 50), 8, _comparison(only_cand=5, only_base=3))
    assert not any("batch-level signal" in w for w in warnings)


def test_zero_disagreement_over_many_problems_is_flagged():
    counts = [0, 2, 5] * 17
    warnings = sanity_checks(_records(counts[:50]), 8, _comparison(0, 0, n=50, both=9))
    assert any("agree on all" in w for w in warnings)


def test_both_conditions_at_zero_correct_is_not_flagged_as_broken():
    """A model below the floor agrees everywhere. TinyLlama-1.1B scored 0/40 on GSM8K
    with varied samples; flagging that told the user to distrust a working engine."""
    counts = [0, 2, 5] * 17
    warnings = sanity_checks(_records(counts[:50]), 8, _comparison(0, 0, n=50))
    assert not any("agree on all" in w for w in warnings)


def test_zero_disagreement_on_a_tiny_run_is_not_flagged():
    # With few problems, agreeing everywhere is ordinary luck rather than a signal.
    warnings = sanity_checks(_records([0, 1, 2]), 8, _comparison(0, 0, n=3))
    assert not any("agree on all" in w for w in warnings)


def test_a_high_no_answer_rate_is_flagged():
    # The observed run: 267 of 400 samples, 66.8%.
    warnings = sanity_checks(_records([0] * 50, [6] * 50), 8,
                             _comparison(only_cand=5, only_base=3))
    assert any("yielded no" in w for w in warnings)


def test_a_low_no_answer_rate_is_not_flagged():
    warnings = sanity_checks(_records([0] * 50, [1] * 50), 8,
                             _comparison(only_cand=5, only_base=3))
    assert not any("yielded no" in w for w in warnings)


def test_a_healthy_run_produces_no_warnings():
    counts = [0, 1, 3, 0, 2, 8, 1, 0, 4, 2] * 5
    warnings = sanity_checks(_records(counts, [0, 1, 0, 2, 0, 1, 0, 0, 1, 0] * 5), 8,
                             _comparison(only_cand=8, only_base=3))
    assert warnings == []


def test_no_records_is_reported_rather_than_passing_silently():
    assert sanity_checks([], 8, _comparison(0, 0, n=1)) == ["no problems were graded"]


def test_every_warning_names_an_observation_not_a_diagnosis():
    # Each message must cite what was seen, so a reader can check it rather than
    # trust a guessed cause.
    counts = [8 if i % 2 else 0 for i in range(50)]
    warnings = sanity_checks(_records(counts, [7] * 50), 8, _comparison(0, 0, n=50, both=9))
    assert len(warnings) == 3
    assert all(any(ch.isdigit() for ch in w) for w in warnings)


# --------------------------------------------------------------------------
# The guards against the run that actually broke
#
# Everything above constructs inputs designed to trip sanity_checks(). This tests
# it against the real artifact from the first 50-problem reference run, kept as a
# fixture. The difference matters: data built to fail a check proves the check
# reacts to what I imagined, not to what went wrong.
# --------------------------------------------------------------------------

import json  # noqa: E402


def _broken_run():
    path = Path(__file__).resolve().parent / "fixtures" / "broken_reference_run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_real_broken_run_is_caught():
    data = _broken_run()
    warnings = sanity_checks(data["records"], data["config"]["samples"], data["comparison"])

    # All three patterns were present in that run, and all three must be reported: any
    # one of them alone is enough to make the 18.0% figure meaningless.
    assert len(warnings) == 3, warnings
    assert any("batch-level signal" in w for w in warnings)
    assert any("agree on all" in w for w in warnings)
    assert any("yielded no" in w for w in warnings)


def test_the_broken_run_looked_like_a_clean_null_result():
    """Why it needed a guard rather than a better verdict.

    The comparison itself is unremarkable: equal accuracy, no disagreements, a verdict
    saying the difference cannot be told from chance. That is exactly what a correct
    harness reports when a method does nothing, which is why nothing objected.
    """
    comparison = _broken_run()["comparison"]
    assert comparison["baseline"]["accuracy"] == comparison["candidate"]["accuracy"]
    assert not comparison["significant"]
    assert comparison["p_value"] == 1.0
    assert "distinguishable from chance" in comparison["verdict"]


def test_the_fixture_shows_the_all_or_nothing_truncation_signature():
    # 0 or 8, never in between: the fingerprint of a batch-wide signal reported per
    # sample. Pinned so the fixture cannot be quietly replaced with something milder.
    counts = {r["n_samples_truncated"] for r in _broken_run()["records"]}
    assert counts == {0, 8}
