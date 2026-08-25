import os
import sys

# Ensure src module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "antigravity-engine/src")))

from orchestrator import AntigravityEngine

def main():
    print("=" * 60)
    print("AUDIT 4: PARALLEL CHANNEL DECODING PROOF (N=8)")
    print("=" * 60)
    
    prompt = "Prove that 2^n > n^2 for all integers n >= 5."
    print(f"Prompt: '{prompt}'")
    
    engine = AntigravityEngine(n_channels=8)
    
    # Run generating and verification
    print("\nRunning test-time search (N=8 parallel paths)...")
    res = engine.run_best_of_n_query(prompt, max_tokens=20, temperature=0.7)
    
    print(f"\nExecution Mode: {res['generation_mode']}")
    print(f"Total Time: {res['latency_ms']:.2f} ms")
    
    # Sort candidates by PRM score
    scores = res['scores']
    candidates = res['candidate_traces']
    
    scored_channels = [(i, scores[i], candidates[i]) for i in range(8)]
    scored_channels.sort(key=lambda x: x[1], reverse=True)
    
    best_ch, best_score, best_text = scored_channels[0]
    worst_ch, worst_score, worst_text = scored_channels[-1]
    
    print("\n" + "=" * 60)
    print("THE PHYSICAL TRUTH: HIGH VS LOW SCORING CHANNELS")
    print("=" * 60)
    
    print(f"\n✅ HIGH-SCORING CHANNEL (Rank 1)")
    print(f"Channel ID: {best_ch}")
    print(f"PRM Score: {best_score:.4f}")
    print(f"Decoded Output:\n👉 {repr(best_text)}")
    
    print(f"\n❌ LOW-SCORING CHANNEL (Rank 8)")
    print(f"Channel ID: {worst_ch}")
    print(f"PRM Score: {worst_score:.4f}")
    print(f"Decoded Output:\n👉 {repr(worst_text)}")
    
    print("\n" + "=" * 60)
    print("If the engine is mathematically aligned, the high-scoring text")
    print("will be coherent English math proof steps, and the low-scoring")
    print("text will be noticeably lower quality or hallucinated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
