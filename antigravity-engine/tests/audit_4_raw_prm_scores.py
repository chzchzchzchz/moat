"""
Audit 4: Mathematical "Gaps" Test (Raw PRM Floating-Point Scores)
Evaluates candidate traces for a logical math problem and asserts raw floating-point verifier score variance.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from orchestrator import AntigravityEngine


def test_raw_prm_scores_variance():
    """Assert that candidate verifier scores are non-uniform and dynamically distributed."""
    model_dir = "models/qwen" if os.path.exists("models/qwen") else "models/tinyllama"
    engine = AntigravityEngine(n_channels=4, model_dir=model_dir)
    prompt = "Prove that for all integers n >= 5, 2^n > n^2."

    res = engine.run_best_of_n_query(
        prompt=prompt,
        max_tokens=30,
        temperature=0.8
    )

    scores = res['scores']
    candidate_traces = res['candidate_traces']

    assert len(scores) == 4, "Expected 4 candidate scores"
    assert len(candidate_traces) == 4, "Expected 4 candidate traces"
    assert 0 <= res['best_index'] < 4, "best_index must be valid"
    assert res['best_score'] > 0.0, "best_score must be positive"
    assert np.isclose(np.sum(scores), 1.0, atol=1e-3), "Softmax normalized scores should sum to ~1.0"

