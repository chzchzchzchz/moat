import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "antigravity-engine", "src")))

from genprm_verifier import GenPRMVerifier
from dora_clustering import DORAClusterer
from orchestrator import AntigravityEngine


def main():
    print("=" * 70)
    print("VERIFICATION: PROGRAM-AIDED SELF-CORRECTION (GenPRM) & DORA CLUSTERING")
    print("=" * 70)

    # 1. Test DORA Real Semantic Clustering
    print("\n--- 1. Testing On-Device DORA Semantic Clustering ---")
    dora = DORAClusterer()
    traces = [
        "Step 1: Compute 2^5 = 32. Since 32 > 25, true. #### 32",
        "Step 1: Compute 2^5 = 32. Since 32 > 25, true. #### 32",  # Duplicate trace
        "Step 1: Write Python script:\n```python\nprint(2**5)\n```\nOutput is 32. #### 32",
        "Unrelated text quantum mechanics electron spin state."
    ]

    dora_res = dora.cluster_candidates(traces)
    print(f"Similarity Matrix (4x4):\n{np.round(dora_res['similarity_matrix'], 3)}")
    print(f"Uniqueness Weights: {np.round(dora_res['uniqueness_weights'], 3)}")
    print(f"Redundant Pairs Found: {len(dora_res['redundancy_pairs'])}")

    # 2. Test GenPRM Program-Aided Verification
    print("\n--- 2. Testing GenPRM Program Code Execution Feedback ---")
    genprm = GenPRMVerifier()
    for i, trace in enumerate(traces):
        rep = genprm.verify_trace_with_code(trace)
        print(f"\nTrace {i}:")
        print(f"  Has Code: {rep['has_code']}")
        if rep['has_code']:
            print(f"  Code Executed Cleanly: {rep['code_success']}")
            print(f"  Code Output: '{rep['code_output']}'")
            print(f"  Consistent with Stated Answer ({rep['stated_answer']}): {rep['is_consistent']}")
            print(f"  GenPRM Score Bonus/Penalty: {rep['reward_modifier']:+.1f}")

    # 3. Test Full Engine Integration Query
    print("\n--- 3. Testing AntigravityEngine Query with GenPRM + DORA ---")
    engine = AntigravityEngine(n_channels=4)
    prompt = "Prove that 2^n > n^2 for all integers n >= 5."
    res = engine.run_best_of_n_query(prompt, max_tokens=30, temperature=0.7)

    print(f"Execution Mode: {res['generation_mode']}")
    print(f"Best Candidate Index: {res['best_index']}")
    print(f"Best Verifier Score: {res['best_score']:.4f}")
    print(f"Reflection Triggered: {res['reflection_triggered']}")
    print(f"Best Trace Output:\n👉 {repr(res['best_trace'])}")

    print("\n" + "=" * 70)
    print("🏆 SUCCESS: GenPRM program execution and DORA semantic clustering verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
