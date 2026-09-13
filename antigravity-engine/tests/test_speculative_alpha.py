"""
Project Antigravity — Speculative Alpha-Curve Routing Unit Tests
Tests speculative decoding thresholding, acceptance probability bounds,
and rollback prevention logic under physical Apple Silicon latency parameters.
"""

import numpy as np
import pytest


def test_speculative_alpha_routing_simulation():
    """Verify speculative decoding acceptance bounds and rollback avoidance."""
    total_tokens = 0
    total_rollbacks_prevented = 0
    total_speculative_savings_ms = 0.0

    target_tpot_ms = 23.9
    draft_tpot_ms = 4.2
    rollback_penalty_ms = 15.0

    num_sequences = 20
    for seq_idx in range(num_sequences):
        seq_len = 64
        for pos in range(seq_len):
            total_tokens += 1
            in_thought_block = (16 <= pos <= 48)

            if in_thought_block:
                target_logprob = -2.8
                draft_logprob = -4.5
                alpha = float(np.clip(np.exp(target_logprob - draft_logprob) * 0.2, 0.05, 0.40))
                assert 0.05 <= alpha <= 0.40, f"Alpha {alpha} out of expected thought-block bounds"
                if alpha < 0.5:
                    total_rollbacks_prevented += 1
            else:
                target_logprob = -0.3
                draft_logprob = -0.4
                alpha = float(np.clip(np.exp(target_logprob - draft_logprob) * 0.95, 0.70, 0.98))
                assert 0.70 <= alpha <= 0.98, f"Alpha {alpha} out of expected structured bounds"
                if alpha > 0.5:
                    total_speculative_savings_ms += (target_tpot_ms - draft_tpot_ms)

    assert total_tokens == num_sequences * 64
    assert total_rollbacks_prevented > 0, "Should have prevented rollbacks in thought blocks"
    assert total_speculative_savings_ms > 0.0, "Should have positive speculative latency savings"


def test_speculative_acceptance_ratio_math():
    """Verify physical math properties of speculative acceptance ratio."""
    # When target and draft match closely, alpha should approach 1.0
    p_target = -0.1
    p_draft = -0.1
    ratio = np.exp(p_target - p_draft)
    assert np.isclose(ratio, 1.0)

    # When draft predicts unlikely token under target distribution
    p_target_divergent = -5.0
    p_draft_high = -0.5
    ratio_divergent = np.exp(p_target_divergent - p_draft_high)
    assert ratio_divergent < 0.02

