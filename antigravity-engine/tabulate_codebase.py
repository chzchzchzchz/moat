import os

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


repo_root = os.path.join(MOAT_ROOT, "antigravity-engine")
# Output location. The default used to be a scratch directory belonging to an external
# tool on one developer's machine, so the script wrote nowhere on any other checkout.
artifact_path = os.environ.get(
    "CODEBASE_TABULATION_PATH",
    os.path.join(MOAT_ROOT, "codebase_tabulation.md"),
)

allowed_exts = {".py", ".cpp", ".mm", ".h", ".metal", ".swift", ".kt", ".md"}
exclude_dirs = {".git", "venv", "build", ".build", "mcache", "build_xcf", "src/shaders"}

def analyze_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        docstring = ""
        # Super simple docstring extraction
        if lines and len(lines) > 0:
            # Check python/swift/c++ comment headers
            for i in range(min(15, len(lines))):
                line = lines[i].strip()
                if line.startswith('\"\"\"') or line.startswith('//') or line.startswith('/*') or line.startswith('*'):
                    docstring += line.replace('\"\"\"', '').replace('//', '').replace('/*', '').replace('*/', '').replace('*', '').strip() + " "
                
        lines_count = len(lines)
        size_bytes = os.path.getsize(filepath)
        
        return {
            "path": os.path.relpath(filepath, repo_root),
            "lines": lines_count,
            "size": size_bytes,
            "purpose": docstring[:200].strip() if docstring else "No module docstring provided."
        }
    except Exception as e:
        return None

results = []
for root, dirs, files in os.walk(repo_root):
    # filter dirs
    dirs[:] = [d for d in dirs if d not in exclude_dirs]
    
    for file in files:
        if any(file.endswith(ext) for ext in allowed_exts):
            full_path = os.path.join(root, file)
            res = analyze_file(full_path)
            if res:
                results.append(res)

results.sort(key=lambda x: x['path'])

os.makedirs(os.path.dirname(os.path.abspath(artifact_path)), exist_ok=True)
with open(artifact_path, "w", encoding="utf-8") as f:
    f.write("# Project Antigravity Codebase Tabulation\n\n")
    f.write("A comprehensive file-by-file audit of the entire Antigravity repository, mapping the Metal/Vulkan engines, Python Verifiers, and Native SDKs.\n\n")
    f.write("| File Path | Lines | Size (Bytes) | Module Purpose / Docstring Extract |\n")
    f.write("|---|---|---|---|\n")
    for r in results:
        f.write(f"| `{r['path']}` | {r['lines']} | {r['size']} | {r['purpose']} |\n")

print(f"Tabulated {len(results)} files to {artifact_path}")
