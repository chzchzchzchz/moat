"""
Project Antigravity — Minimal "Hello World" Integration Example

The 30-Minute Integration Metric:
Demonstrates how a developer can boot the engine and run parallel offline
Best-of-N reasoning in 5 lines of code.

Usage:
    python3 antigravity-engine/examples/hello_world.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from orchestrator import AntigravityEngine

# 1. Initialize engine with strict 4GB memory footprint
engine = AntigravityEngine(n_channels=8)

# 2. Run offline Best-of-N reasoning query
print("Running local offline Best-of-N reasoning query...")
result = engine.run_best_of_n_query(
    prompt="Prove that 2^n > n^2 for all integers n >= 5.",
    max_tokens=40
)

# 3. Print output and verified metrics
print("\n" + "=" * 60)
print("  ANTIGRAVITY HELLO WORLD RESPONSE")
print("=" * 60)
print(f"Top Output Trace:\n{result['best_trace']}")
print(f"\nVerifier Score:      {result['best_score']:.4f}")
print(f"Evaluated Traces:    {result['candidates_evaluated']} parallel channels")
print(f"Token Savings:       {result['token_savings_pct']:.1f}%")
print(f"Total Latency:       {result['latency_ms']:.2f} ms")
print("=" * 60)
