"""
Escape tests for src/vericoding_shell.py's restricted expression evaluation.

These payloads are not hypothetical: each one was verified to succeed against
the previous eval()/exec() implementation, which guarded only __builtins__.
The math path returned the process working directory and the exec path returned
the process uid. A builtins whitelist does not contain Python, because attribute
traversal from any reachable object reaches the interpreter without using one.

Run: pytest tests/test_restricted_eval.py
"""

import pytest

from vericoding_shell import iOSCodeExecutor, safe_eval, UnsafeExpressionError


# Reaches os through warnings.catch_warnings._module without any builtin.
_ESCAPE = (
    "[c for c in ().__class__.__base__.__subclasses__() "
    "if c.__name__=='catch_warnings'][0]()._module"
    ".__builtins__['__import__']('os').getcwd()"
)


@pytest.fixture
def executor():
    return iOSCodeExecutor()


def test_interpreter_escape_is_rejected(executor):
    ok, out, err = executor.execute_math_expression(_ESCAPE)
    assert ok is False
    assert out == ""
    assert "rejected" in err


@pytest.mark.parametrize("payload", [
    "().__class__",                      # dunder attribute traversal
    "[].__class__.__mro__[1]",           # subscript into the type hierarchy
    "__import__('os').getcwd()",         # undefined name
    "[c for c in (1, 2)]",               # comprehension
    "(lambda: 1)()",                     # lambda
    "open('/etc/passwd').read()",        # undefined name reaching the filesystem
])
def test_escape_shapes_are_rejected(executor, payload):
    ok, _, err = executor.execute_math_expression(payload)
    assert ok is False, f"{payload!r} should not evaluate"
    assert "rejected" in err


@pytest.mark.parametrize("expr,expected", [
    ("2 + 3 * 4", "14"),
    ("sqrt(16)", "4.0"),
    ("max(3, 7)", "7"),
    ("math.floor(3.9)", "3"),
    ("pow(2, 10)", "1024"),
    ("2 if 1 < 2 else 3", "2"),
])
def test_arithmetic_still_evaluates(executor, expr, expected):
    ok, out, err = executor.execute_math_expression(expr)
    assert ok is True, err
    assert out == expected


def test_in_process_exec_refuses_by_default(executor):
    ok, out, err = executor.execute_simple_program("print(1)")
    assert ok is False
    assert "disabled" in err


def test_in_process_exec_available_when_opted_in():
    executor = iOSCodeExecutor(allow_unrestricted_exec=True)
    ok, out, err = executor.execute_simple_program("print(6 * 7)")
    assert ok is True, err
    assert out == "42"


def test_safe_eval_requires_declared_names():
    with pytest.raises(UnsafeExpressionError):
        safe_eval("undeclared + 1", {})
    assert safe_eval("a + 1", {"a": 41}) == 42
