import sys
import os
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from genprm_verifier import GenPRMVerifier


def test_extract_python_code():
    verifier = GenPRMVerifier()
    trace = """
    Let's write a quick script to test:
    ```python
    x = 2 ** 5
    print(x)
    ```
    Therefore the answer is 32.
    """
    codes = verifier.extract_python_code(trace)
    assert len(codes) == 1
    assert "2 ** 5" in codes[0]


def test_extract_stated_answer():
    verifier = GenPRMVerifier()
    assert verifier.extract_stated_answer("The result is #### 42") == "42"
    assert verifier.extract_stated_answer("Thus \\boxed{100}") == "100"
    assert verifier.extract_stated_answer("So x equals 3.14") == "3.14"


def test_execute_code_safely():
    verifier = GenPRMVerifier()
    success, stdout, err = verifier.execute_code_safely("print(2 + 2)")
    assert success is True
    assert stdout == "4"
    assert err is None

    # Test error handling
    success_err, stdout_err, err_msg = verifier.execute_code_safely("print(1 / 0)")
    assert success_err is False
    assert "ZeroDivisionError" in err_msg


def test_verify_trace_consistent_code():
    verifier = GenPRMVerifier()
    trace = """
    To solve 2^5:
    ```python
    print(2 ** 5)
    ```
    #### 32
    """
    res = verifier.verify_trace_with_code(trace)
    assert res['has_code'] is True
    assert res['code_success'] is True
    assert res['code_answer'] == "32"
    assert res['stated_answer'] == "32"
    assert res['is_consistent'] is True
    assert res['reward_modifier'] == 1.5


def test_verify_trace_contradictory_code():
    verifier = GenPRMVerifier()
    trace = """
    To solve 2^5:
    ```python
    print(2 ** 5)
    ```
    #### 999
    """
    res = verifier.verify_trace_with_code(trace)
    assert res['has_code'] is True
    assert res['code_success'] is True
    assert res['code_answer'] == "32"
    assert res['stated_answer'] == "999"
    assert res['is_consistent'] is False
    assert res['reward_modifier'] == -1.0
    assert "contradicts" in res['feedback_prompt']


def test_score_candidates_genprm():
    verifier = GenPRMVerifier()
    candidates = [
        "Incorrect math ```python\nprint(10)\n``` #### 99",
        "Correct math ```python\nprint(32)\n``` #### 32",
    ]
    base_scores = np.array([0.5, 0.5], dtype=np.float32)
    adj_scores, reports = verifier.score_candidates_genprm(candidates, base_scores)

    assert adj_scores[1] > adj_scores[0]
    assert reports[1]['is_consistent'] is True
