"""
Project Antigravity — Neurosymbolic Verification & Self-Healing.

Implements in-process verification of model-generated code/math claims:

  * ASTVerifier — Python AST parsing: validates that generated code blocks
    parse cleanly and use only safe, whitelisted operations.
  * JSCVerifier — JavaScript syntax validation via subprocess (Node.js) or
    a pure-Python fallback regex check.
  * Z3Verifier — SMT-based math verification using z3-solver when available,
    else a rational-arithmetic fallback.
  * SelfHealingLoop — on verification failure, appends a feedback prompt and
    re-dispatches to the reasoner (up to max_retries).

Designed for phone-level use: all verification is in-process, no network,
sub-second micro-bursts.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
import os
import time
import threading
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field


# Whitelisted Python AST node types — safe, side-effect-free constructs only
SAFE_NODES = {
    ast.Module, ast.Expr, ast.Call, ast.Name, ast.Constant,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript,
    ast.Index, ast.Slice, ast.Attribute, ast.Load, ast.Store,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
}

# Whitelisted builtin names — no open(), os.system, eval, exec, __import__, etc.
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "chr", "dict", "enumerate", "filter",
    "float", "int", "isinstance", "len", "list", "map", "max", "min",
    "ord", "pow", "print", "range", "round", "set", "sorted", "str",
    "sum", "tuple", "type", "zip",
}


@dataclass
class VerificationResult:
    passed: bool
    detail: str
    verifier: str
    latency_ms: float
    feedback: Optional[str] = None


@dataclass
class HealingAttempt:
    attempt: int
    trace: str
    result: VerificationResult
    feedback_prompt: Optional[str]


class ASTVerifier:
    """
    In-process Python AST verifier.

    Parses extracted Python code blocks and rejects anything using unsafe
    constructs (open(), os.system, eval, exec, __import__, attribute access
    on dunder names, etc.). Returns a VerificationResult with a feedback
    prompt for self-healing on failure.
    """

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def verify(self, code: str) -> VerificationResult:
        t0 = time.monotonic()
        feedback: Optional[str] = None
        passed = True
        detail = "AST parse clean"

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            passed = False
            detail = f"Syntax error: {exc.msg} at line {exc.lineno}"
            feedback = (
                f"Your Python code has a syntax error: {detail}\n"
                "Fix the syntax and re-output a complete, valid Python program."
            )
            return VerificationResult(False, detail, "ast",
                                      (time.monotonic() - t0) * 1000.0, feedback)

        for node in ast.walk(tree):
            node_type = type(node)
            if node_type not in SAFE_NODES:
                passed = False
                detail = f"Unsafe AST node: {node_type.__name__}"
                feedback = (
                    f"Your code uses an unsafe construct ({node_type.__name__}). "
                    "Rewrite using only safe builtins and expressions "
                    "(no open, os, subprocess, eval, exec, __import__, dunder access)."
                )
                return VerificationResult(False, detail, "ast",
                                          (time.monotonic() - t0) * 1000.0, feedback)

            if isinstance(node, ast.Name):
                if node.id.startswith("__"):
                    passed = False
                    detail = f"Dunder name access: {node.id}"
                    feedback = "Dunder attribute access is not allowed."
                    return VerificationResult(False, detail, "ast",
                                              (time.monotonic() - t0) * 1000.0, feedback)
                if node.id not in SAFE_BUILTINS and not self._is_local_var(node, tree):
                    passed = False
                    detail = f"Unsafe name: {node.id}"
                    feedback = (
                        f"Name '{node.id}' is not a safe builtin or local variable. "
                        "Use only safe builtins."
                    )
                    return VerificationResult(False, detail, "ast",
                                              (time.monotonic() - t0) * 1000.0, feedback)

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    if func.id not in SAFE_BUILTINS:
                        passed = False
                        detail = f"Unsafe builtin call: {func.id}"
                        feedback = (
                            f"Calling '{func.id}' is not allowed. "
                            "Use only safe builtins."
                        )
                        return VerificationResult(False, detail, "ast",
                                                  (time.monotonic() - t0) * 1000.0, feedback)
                elif isinstance(func, ast.Attribute):
                    attr = func.attr
                    if attr.startswith("__"):
                        passed = False
                        detail = f"Dunder attribute: {attr}"
                        feedback = "Dunder attribute access is not allowed."
                        return VerificationResult(False, detail, "ast",
                                                  (time.monotonic() - t0) * 1000.0, feedback)

        latency_ms = (time.monotonic() - t0) * 1000.0
        return VerificationResult(passed, detail, "ast", latency_ms, feedback)

    def _is_local_var(self, name_node: ast.Name, tree: ast.Module) -> bool:
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    defined.add(arg.arg)
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
        return name_node.id in defined


class JSCVerifier:
    """
    JavaScript verification.

    Tries Node.js first (if available) for real syntax validation via
    `node --check`. Falls back to a pure-Python structural check.
    """

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self._node_available: Optional[bool] = None

    def _has_node(self) -> bool:
        if self._node_available is None:
            try:
                subprocess.run(
                    ["node", "--version"],
                    capture_output=True, timeout=3.0,
                )
                self._node_available = True
            except Exception:
                self._node_available = False
        return self._node_available

    def verify(self, code: str) -> VerificationResult:
        t0 = time.monotonic()
        if self._has_node():
            return self._verify_with_node(code, t0)
        return self._verify_python_fallback(code, t0)

    def _verify_with_node(self, code: str, t0: float) -> VerificationResult:
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name
        try:
            res = subprocess.run(
                ["node", "--check", tmp_path],
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            if res.returncode == 0:
                return VerificationResult(True, "JS syntax clean", "jsc", latency_ms)
            feedback = (
                f"JavaScript syntax error: {res.stderr.strip()}\n"
                "Fix the syntax and re-output valid JavaScript."
            )
            return VerificationResult(False, res.stderr.strip(), "jsc", latency_ms, feedback)
        except Exception:
            return self._verify_python_fallback(code, t0)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _verify_python_fallback(self, code: str, t0: float) -> VerificationResult:
        latency_ms = (time.monotonic() - t0) * 1000.0
        if code.count("{") != code.count("}"):
            return VerificationResult(False, "Unbalanced braces", "jsc-fallback", latency_ms,
                                      "Fix unbalanced braces in the JavaScript code.")
        if code.count("(") != code.count(")"):
            return VerificationResult(False, "Unbalanced parens", "jsc-fallback", latency_ms,
                                      "Fix unbalanced parentheses in the JavaScript code.")
        for kw in ["eval(", "Function(", "setTimeout(", "setInterval(", "require("]:
            if kw in code:
                return VerificationResult(False, f"Unsafe JS construct: {kw}", "jsc-fallback", latency_ms,
                                          f"'{kw}' is not allowed in verified JavaScript.")
        return VerificationResult(True, "JS structural check passed (Node unavailable)", "jsc-fallback", latency_ms)


class Z3Verifier:
    """
    SMT-based math verification using z3-solver when available, else a
    rational-arithmetic fallback that evaluates simple arithmetic expressions.

    Verifies that a stated numeric answer matches an independently computed
    value — catches math hallucinations.
    """

    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s
        self._z3_available: Optional[bool] = None

    def _has_z3(self) -> bool:
        if self._z3_available is None:
            try:
                import z3  # noqa: F401
                self._z3_available = True
            except Exception:
                self._z3_available = False
        return self._z3_available

    def verify(self, stated_answer: str, expression: Optional[str] = None) -> VerificationResult:
        t0 = time.monotonic()
        if expression is None:
            return VerificationResult(True, "No expression to verify; accepted on face value", "z3", 0.0)

        if self._has_z3():
            return self._verify_with_z3(stated_answer, expression, t0)
        return self._verify_python_fallback(stated_answer, expression, t0)

    def _verify_with_z3(self, stated_answer: str, expression: str, t0: float) -> VerificationResult:
        try:
            import z3
        except Exception:
            return self._verify_python_fallback(stated_answer, expression, t0)

        try:
            solver = z3.Solver()
            solver.set("timeout", int(self.timeout_s * 1000))
            z3_vars = {"Int": z3.Int, "Real": z3.Real, "z3": z3}
            expr = eval(expression, {"__builtins__": {}}, z3_vars)
            solver.add(expr == z3.IntVal(int(stated_answer)) if stated_answer.lstrip("-").isdigit()
                       else expr == z3.RealVal(float(stated_answer)))
            result = solver.check()
            latency_ms = (time.monotonic() - t0) * 1000.0
            if result == z3.sat:
                return VerificationResult(True, "Z3: expression equals stated answer", "z3", latency_ms)
            return VerificationResult(False, "Z3: expression contradicts stated answer", "z3", latency_ms,
                                      "The math expression does not evaluate to the stated answer. Re-check your steps.")
        except Exception:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._verify_python_fallback(stated_answer, expression, t0)

    def _verify_python_fallback(self, stated_answer: str, expression: str, t0: float) -> VerificationResult:
        latency_ms = (time.monotonic() - t0) * 1000.0
        try:
            allowed = {"abs": abs, "pow": pow, "max": max, "min": min, "round": round}
            result = eval(expression, {"__builtins__": {}}, allowed)
            expected = float(result)
            stated = float(stated_answer)
            if abs(expected - stated) < 1e-6:
                return VerificationResult(True, f"Arithmetic check passed: {expected}", "z3-fallback", latency_ms)
            return VerificationResult(False, f"Arithmetic mismatch: computed {expected}, stated {stated}",
                                      "z3-fallback", latency_ms,
                                      f"The expression evaluates to {expected}, not {stated}. Re-check your math.")
        except Exception as exc:
            return VerificationResult(False, f"Cannot evaluate expression: {exc}", "z3-fallback", latency_ms,
                                      "Could not verify the math expression automatically.")


class SelfHealingLoop:
    """
    Self-healing dispatch loop.

    On verification failure, appends a feedback prompt and re-dispatches to
    the reasoner (up to max_retries). Returns the first verified trace or
    the best-effort trace after all retries are exhausted.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._attempts: List[HealingAttempt] = []

    def run(self, dispatch: Callable[[str], str],
            verifiers: List[Any],
            initial_prompt: str,
            context: Optional[Dict[str, Any]] = None) -> Tuple[str, List[HealingAttempt]]:
        self._attempts = []
        current_prompt = initial_prompt
        best_trace = ""
        best_result: Optional[VerificationResult] = None

        for attempt in range(self.max_retries + 1):
            trace = dispatch(current_prompt)
            results = []
            all_passed = True
            for v in verifiers:
                if isinstance(v, Z3Verifier):
                    expr = context.get("expression") if context else None
                    stated = self._extract_answer(trace)
                    res = v.verify(stated, expr)
                else:
                    code = self._extract_code(trace)
                    res = v.verify(code) if code else VerificationResult(True, "No code to verify", "ast", 0.0)
                results.append(res)
                if not res.passed:
                    all_passed = False

            attempt_record = HealingAttempt(
                attempt=attempt,
                trace=trace,
                result=best_result or results[-1] if results else VerificationResult(False, "no verifiers", "none", 0.0),
                feedback_prompt=None,
            )
            self._attempts.append(attempt_record)

            if all_passed:
                best_trace = trace
                best_result = results[0] if results else None
                return trace, self._attempts

            feedback_parts = []
            for res in results:
                if not res.passed and res.feedback:
                    feedback_parts.append(res.feedback)
            feedback = "\n".join(feedback_parts)

            current_prompt = (
                f"{initial_prompt}\n\n"
                f"[SELF-HEALING FEEDBACK from attempt {attempt + 1}]\n"
                f"{feedback}\n"
                "Please correct your reasoning and output a revised, verified answer."
            )
            best_trace = trace
            best_result = results[0] if results else None

        return best_trace, self._attempts

    @staticmethod
    def _extract_answer(trace: str) -> Optional[str]:
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", trace)
        if m:
            return m.group(1)
        m = re.search(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", trace)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _extract_code(trace: str) -> str:
        m = re.search(r"```python\s*(.*?)\s*```", trace, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"```\s*(.*?)\s*```", trace, re.DOTALL)
        if m:
            return m.group(1)
        return ""


__all__ = [
    "VerificationResult", "HealingAttempt",
    "ASTVerifier", "JSCVerifier", "Z3Verifier", "SelfHealingLoop",
]
