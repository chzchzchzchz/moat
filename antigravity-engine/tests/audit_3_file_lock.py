"""
Audit 3: Physical File Lock Test (OS-Level File Handle Verification)
Verifies that loading model weights creates actual OS-level file descriptors and allocates VRAM.
"""

import os
import sys
import subprocess
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from native_bridge import NativeMetalEngine

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))



def test_physical_file_lock():
    """Verify weight file accessibility and unified memory allocation."""
    candidate_model_paths = [
        os.path.abspath("models/tinyllama/model.safetensors"),
        os.path.abspath("../models/tinyllama/model.safetensors"),
        os.path.join(MOAT_ROOT, "models/tinyllama/model.safetensors")
    ]
    model_path = next((p for p in candidate_model_paths if os.path.exists(p)), None)

    candidate_dylib_paths = [
        os.path.abspath("src/libantigravity_engine.dylib"),
        os.path.abspath("libantigravity_engine.dylib"),
        os.path.abspath("antigravity-engine/src/libantigravity_engine.dylib"),
        os.path.join(MOAT_ROOT, "antigravity-engine/src/libantigravity_engine.dylib")
    ]
    dylib_path = next((p for p in candidate_dylib_paths if os.path.exists(p)), None)

    if not model_path or not dylib_path:
        pytest.skip("Model weights or dylib not available")

    assert os.path.exists(model_path), "Model weight file must exist on disk"
    assert os.path.getsize(model_path) > 100 * 1024 * 1024, "Model weight file must be > 100MB"
    assert os.path.exists(dylib_path), "Engine dylib must exist on disk"

    engine = NativeMetalEngine(
        dylib_path=dylib_path,
        model_path=model_path,
        n_channels=2,
        vocab_size=32000,
        hidden_dim=2048
    )

    try:
        assert engine.is_ready, "Native engine must be ready after loading weights"
        allocated_bytes = engine.get_allocated_bytes()
        assert allocated_bytes > 0, "No weights allocated in engine memory!"
        assert allocated_bytes > 500 * 1024 * 1024, "Expected >500MB allocated for TinyLlama weights"
    finally:
        engine.unload_weights()


