"""
Project Antigravity — AST Adversarial Verification Suite
Validates AST parser against prompt injection, dummy returns, and mimicry attacks.
"""

import ast
import pytest


class ASTIntegrityVerifier(ast.NodeVisitor):
    def __init__(self, required_variables):
        self.found_variables = set()
        self.required_variables = set(required_variables)
        self.is_valid = False
        self.contains_dummy_prints = False
        self.has_complex_logic = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                if node.args[0].value == 42 or node.args[0].value == "Task Verified!":
                    self.contains_dummy_prints = True
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.required_variables:
            self.found_variables.add(node.id)
        self.generic_visit(node)

    def visit_For(self, node):
        self.has_complex_logic = True
        self.generic_visit(node)

    def visit_While(self, node):
        self.has_complex_logic = True
        self.generic_visit(node)

    def visit_If(self, node):
        self.has_complex_logic = True
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.has_complex_logic = True
        self.generic_visit(node)

    def verify(self, code_string):
        try:
            tree = ast.parse(code_string)
            self.visit(tree)

            if self.contains_dummy_prints:
                return False, "Failed: Contains syntactic mimicry (dummy print)."

            missing_vars = self.required_variables - self.found_variables
            if missing_vars:
                return False, f"Failed: Missing required logical constraints: {missing_vars}"

            if not self.has_complex_logic:
                return False, "Failed: Lacks semantic logic or control flow."

            return True, "Verified: AST Integrity Pass."

        except SyntaxError as e:
            return False, f"Failed: Syntax Error - {str(e)}"


def test_ast_dummy_print_rejected():
    verifier = ASTIntegrityVerifier(required_variables=["x", "y"])
    passed, reason = verifier.verify("def solve():\n    print(42)\n    return 42")
    assert not passed
    assert "dummy print" in reason or "syntactic mimicry" in reason


def test_ast_hardcoded_bypass_rejected():
    verifier = ASTIntegrityVerifier(required_variables=["x", "y"])
    passed, reason = verifier.verify("def solve(x, y):\n    return 'Task Verified!'")
    assert not passed


def test_ast_legitimate_math_passed():
    verifier = ASTIntegrityVerifier(required_variables=["x", "y"])
    passed, reason = verifier.verify(
        "def solve(x, y):\n    result = x * y + 10\n    if result > 50:\n        return result\n    return 0"
    )
    assert passed
    assert "Verified" in reason


def test_ast_missing_variable_rejected():
    verifier = ASTIntegrityVerifier(required_variables=["x", "y"])
    passed, reason = verifier.verify("def solve(x):\n    result = x * 2\n    return result")
    assert not passed
    assert "Missing required" in reason

