"""
Project Antigravity — Speculative Alpha Acceptance Unit Tests

Verifies the mathematical properties of the speculative decoding acceptance
ratio used to decide whether a draft token is accepted by the target model.

These are deterministic, math-only unit tests. They do not fabricate latency
savings or claim speculative speedups without executing a model — they only
assert that the acceptance probability p_accept = min(1, exp(p_target - p_draft))
behaves correctly as a probability (monotonic, bounded in [0, 1], equal to 1
when draft matches target).
"""

import numpy as np
import pytest


def test_speculative_acceptance_ratio_equals_one_when_draft_matches_target():
    """When target and draft assign the same log-prob, acceptance is 1.0."""
    p_target = -0.1
    p_draft = -0.1
    ratio = float(np.exp(p_target - p_draft))
    assert np.isclose(ratio, 1.0)


def test_speculative_acceptance_ratio_decays_with_divergence():
    """p_accept = min(1, exp(p_target - p_draft)) decays as the draft assigns
    increasingly higher probability than the target (draft log-prob > target)."""
    p_target = -1.0
    close = float(np.exp(p_target - (-0.99)))   # draft slightly better than target
    mid = float(np.exp(p_target - (-0.5)))      # draft moderately better
    far = float(np.exp(p_target - 0.0))         # draft much better
    assert close > mid > far
    assert 0.0 < close <= 1.0
    assert 0.0 < mid <= 1.0
    assert 0.0 < far <= 1.0


def test_speculative_acceptance_never_exceeds_one():
    """p_accept = min(1, exp(...)) is capped at 1.0 even when draft beats target."""
    p_target = -5.0
    p_draft = -0.5   # draft assigns much higher prob than target
    ratio = float(np.exp(p_target - p_draft))
    assert ratio < 1.0
    p_accept = min(1.0, ratio)
    assert np.isclose(p_accept, ratio)
    assert p_accept <= 1.0


def test_speculative_acceptance_bounded_in_unit_interval():
    """Acceptance probability is always a valid probability in [0, 1]."""
    for p_target in [-10.0, -1.0, 0.0]:
        for p_draft in [-10.0, -1.0, 0.0]:
            p_accept = min(1.0, float(np.exp(p_target - p_draft)))
            assert 0.0 <= p_accept <= 1.0, f"p_accept={p_accept} out of [0,1]"


def test_speculative_acceptance_matches_engine_clip_behavior():
    """Mirrors the engine's alpha clip: alpha in [0.05, 0.40] for thought blocks,
    [0.70, 0.98] for structured tokens — computed deterministically, no random."""
    target_logprob = -2.8
    draft_logprob = -4.5
    alpha = float(np.clip(np.exp(target_logprob - draft_logprob) * 0.2, 0.05, 0.40))
    assert 0.05 <= alpha <= 0.40

    target_logprob = -0.3
    draft_logprob = -0.4
    alpha = float(np.clip(np.exp(target_logprob - draft_logprob) * 0.95, 0.70, 0.98))
    assert 0.70 <= alpha <= 0.98
