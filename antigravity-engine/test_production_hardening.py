import sys
sys.path.insert(0, 'src')
from orchestrator import AntigravityEngine
from model_loader import AntigravityConfig

print("Testing Orchestrator with Native Engine binding...")
config = AntigravityConfig(
    reasoner_model_path="models/qwen3.5/model.safetensors",
    verifier_model_path="models/qwen3.5/model.safetensors",
    n_channels=1,
    max_new_tokens=10
)
orch = AntigravityEngine(config)
print("Orchestrator initialized. Native engine available:", orch.native_engine is not None)
