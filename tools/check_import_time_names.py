#!/usr/bin/env python3
"""
Find names used at import time before they are defined.

`python -m compileall` only checks that a file parses. A module whose top level
reads a name above the line that binds it parses perfectly and then raises
NameError the moment anything imports it — the file is simply dead, and nothing
in CI notices until someone tries to run it.

That is not hypothetical here. Replacing one developer's absolute paths with a
MOAT_ROOT block put the definition *below* the sys.path.insert() that used it in
four scripts at once, and compileall passed all four.

Scope is deliberately narrow, so this reports facts rather than guesses: only
module-level statements (never inside a def or class, where execution is
deferred), and only names the same module binds at its top level. A name that
comes from a star-import or is never bound here is not reported, because this
cannot know when it becomes available.

    python3 tools/check_import_time_names.py [path ...]

Exits non-zero if any use-before-definition is found.
"""

import ast
import sys
from pathlib import Path


def bindings(tree: ast.Module) -> dict:
    """Module-level name -> the first line number that binds it."""
    out: dict = {}
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    out.setdefault(alias.asname or alias.name.split(".")[0], sub.lineno)
            elif isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    if alias.name != "*":
                        out.setdefault(alias.asname or alias.name, sub.lineno)
            elif isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            out.setdefault(name.id, sub.lineno)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.setdefault(sub.name, sub.lineno)
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                for name in ast.walk(sub.target):
                    if isinstance(name, ast.Name):
                        out.setdefault(name.id, sub.lineno)
    return out


def check(path: Path) -> list:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []          # compileall's job, not this one

    bound = bindings(tree)
    found, seen = [], set()

    for node in tree.body:
        # A def or class body does not run at import time.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                first = bound.get(sub.id)
                if first is not None and sub.lineno < first and (sub.id, first) not in seen:
                    seen.add((sub.id, first))
                    found.append((sub.lineno, sub.id, first))
    return found


def main(argv) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path(".")]
    files = []
    for root in roots:
        files.extend([root] if root.is_file() else sorted(root.rglob("*.py")))

    failures = 0
    for path in files:
        if ".git" in path.parts or "node_modules" in path.parts:
            continue
        for line, name, defined in check(path):
            print(f"{path}:{line}: '{name}' is read at import time but only bound "
                  f"at line {defined} — importing this module raises NameError")
            failures += 1

    print(f"checked {len(files)} file(s): "
          + ("no use-before-definition found" if not failures
             else f"{failures} use(s) before definition"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
