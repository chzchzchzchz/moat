import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from orchestrator import AntigravityEngine

print("Testing Orchestrator with Native Engine binding...")
orch = AntigravityEngine(
    model_dir="models/qwen" if os.path.exists("models/qwen") else "models/tinyllama",
    n_channels=1
)
print("Orchestrator initialized. Native engine available:", orch.native_engine is not None)
print("Active generation mode:", orch._generation_mode)
