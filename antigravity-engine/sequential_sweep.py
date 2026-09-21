import os
import re

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


repo_root = os.path.join(MOAT_ROOT, "antigravity-engine")
allowed_exts = {".py", ".cpp", ".mm", ".h", ".metal", ".swift", ".kt", ".md", ".c", ".txt", ".sh"}
exclude_dirs = {".git", "venv", "build", ".build", "mcache", "build_xcf", "src/shaders"}

regex = re.compile(r'(stub|placeholder|dummy|mock|fake|TODO|FIXME|hardcode)', re.IGNORECASE)

files_to_check = []
for root, dirs, files in os.walk(repo_root):
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    for file in files:
        if any(file.endswith(ext) for ext in allowed_exts):
            files_to_check.append(os.path.join(root, file))

files_to_check.sort()
total = len(files_to_check)
issues_found = []

for i, filepath in enumerate(files_to_check):
    rel_path = os.path.relpath(filepath, repo_root)
    #print(f"[{i+1}/{total}] Checking {rel_path}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if regex.search(line):
                    issues_found.append(f"{rel_path}:{line_num} -> {line.strip()}")
    except Exception as e:
        pass

print(f"\n--- SWEEP COMPLETE: {total} FILES CHECKED ---")
for issue in issues_found:
    print(issue)
