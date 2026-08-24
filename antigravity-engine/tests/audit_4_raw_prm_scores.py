"""
Audit 4: Mathematical "Gaps" Test (Raw PRM Floating-Point Scores)
Evaluates 8 candidate traces for a logical math problem and prints raw floating-point verifier scores.
"""

import os
import sys
import numpy as np

sys.path.insert(0, 'antigravity-engine/src')

from orchestrator import AntigravityEngine

print("=================================================================")
print("  AUDIT 4: MATHEMATICAL GAPS TEST (Raw PRM Floating-Point Scores)")
print("=================================================================")

engine = AntigravityEngine(n_channels=8, model_dir="models/tinyllama")
prompt = "Prove that for all integers n >= 5, 2^n > n^2."

res = engine.run_best_of_n_query(
    prompt=prompt,
    max_tokens=40,
    temperature=0.8
)

print(f"\nPrompt: '{prompt}'")
print(f"Generation Mode: {res['generation_mode']}")
print(f"Best Candidate Index: {res['best_index']}")
print(f"Best Candidate Score: {res['best_score']:.6f}")

print("\n--- RAW FLOATING-POINT VERIFIER CONFIDENCE SCORES Across N=8 Channels ---")
scores = res['scores']
traces = res['candidate_traces']

for idx, (score, trace) in enumerate(zip(scores, traces)):
    marker = "🏆 BEST" if idx == res['best_index'] else "  "
    print(f"  Channel {idx}: Raw Score = {score:.6f} {marker}")
    print(f"             Trace snippet: {trace[:80]}...")

print("\nStatistical Analysis of Scores:")
print(f"  • Min Score:  {np.min(scores):.6f}")
print(f"  • Max Score:  {np.max(scores):.6f}")
print(f"  • Mean Score: {np.mean(scores):.6f}")
print(f"  • Std Dev:    {np.std(scores):.6f}")
print(f"  • Unique Scores Count: {len(set(scores))}")

print("\n================== AUDIT 4 VERDICT ==================")
if np.std(scores) > 0 and len(set(scores)) > 1:
    print("🏆 SUCCESS: Raw verifier confidence scores vary continuously")
    print("   and dynamically across reasoning candidate traces!")
else:
    print("❌ FAILED: Scores are uniform or hardcoded.")
print("=====================================================")
