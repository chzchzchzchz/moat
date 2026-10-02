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


def test_execute_code_unsandboxed():
    verifier = GenPRMVerifier(enable_code_execution=True)
    success, stdout, err = verifier.execute_code_unsandboxed("print(2 + 2)")
    assert success is True
    assert stdout == "4"
    assert err is None

    # Test error handling
    success_err, stdout_err, err_msg = verifier.execute_code_unsandboxed("print(1 / 0)")
    assert success_err is False
    assert "ZeroDivisionError" in err_msg


def test_verify_trace_consistent_code():
    # These assert on execution results, so they opt in explicitly.
    verifier = GenPRMVerifier(enable_code_execution=True)
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
    # These assert on execution results, so they opt in explicitly.
    verifier = GenPRMVerifier(enable_code_execution=True)
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
    # These assert on execution results, so they opt in explicitly.
    verifier = GenPRMVerifier(enable_code_execution=True)
    candidates = [
        "Incorrect math ```python\nprint(10)\n``` #### 99",
        "Correct math ```python\nprint(32)\n``` #### 32",
    ]
    base_scores = np.array([0.5, 0.5], dtype=np.float32)
    adj_scores, reports = verifier.score_candidates_genprm(candidates, base_scores)

    assert adj_scores[1] > adj_scores[0]
    assert reports[1]['is_consistent'] is True


def test_code_execution_is_off_by_default():
    """The default must not run model-generated code, and must not score on it."""
    verifier = GenPRMVerifier()
    assert verifier.enable_code_execution is False

    success, stdout, err = verifier.execute_code_unsandboxed("print('should not run')")
    assert success is False
    assert stdout == ""
    assert "disabled" in err

    trace = """
    ```python
    print(32)
    ```
    #### 32
    """
    res = verifier.verify_trace_with_code(trace)
    assert res['code_execution_disabled'] is True
    # Neutral: a trace must not be rewarded or punished for code that never ran.
    assert res['reward_modifier'] == 0.0
    assert res['is_consistent'] is False


def test_opted_in_execution_is_confined():
    """Opted-in execution still applies the limits the class documents."""
    verifier = GenPRMVerifier(enable_code_execution=True, max_memory_mb=64)

    # Environment is scrubbed to PATH rather than inherited wholesale.
    ok, out, _ = verifier.execute_code_unsandboxed("import os; print(len(os.environ))")
    assert ok is True
    assert int(out) <= 3, f"expected a scrubbed environment, saw {out} variables"

    # Working directory holds only the candidate file.
    ok, out, _ = verifier.execute_code_unsandboxed("import os; print(len(os.listdir('.')))")
    assert ok is True
    assert int(out) == 1

    # max_memory_mb is enforced, not merely stored.
    ok, _, _ = verifier.execute_code_unsandboxed("x = bytearray(400 * 1024 * 1024); print('ok')")
    assert ok is False, "400MB allocation should fail under a 64MB cap"
