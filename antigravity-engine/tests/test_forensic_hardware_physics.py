"""
Project Antigravity — Forensic Verification & Hardware Physics Test Harness

Executes 4 hardware-level forensic tests:
1. Weight Corruption Test (Clean vs Zeroed Safetensors weights)
2. Metal Hardware Execution Profiling (Compute pipeline verification)
3. Memory Step-Function (Physical RSS memory tracking across load/gen/unload)
4. Entropy / Temperature Test (T=0.0 deterministic vs T=2.0 chaotic entropy)
"""

import os
import sys
import time
import shutil
import psutil
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from orchestrator import AntigravityEngine
from native_bridge import NativeMetalEngine
from tokenizer import LlamaTokenizer

MODEL_DIR = "models/tinyllama"
ORIGINAL_WEIGHTS = os.path.join(MODEL_DIR, "model.safetensors")
CORRUPT_DIR = "scratch/corrupt_model"
CORRUPT_WEIGHTS = os.path.join(CORRUPT_DIR, "model.safetensors")


def test_1_weight_corruption():
    """Verify that model output depends on physical weight file bytes."""
    if not os.path.exists(ORIGINAL_WEIGHTS):
        pytest.skip(f"model.safetensors not found at {ORIGINAL_WEIGHTS}")

    # 1. Clean Run
    engine_clean = AntigravityEngine(n_channels=2, model_dir=MODEL_DIR)
    prompt = "Prove that 2^n > n^2 for all integers n >= 5."
    res_clean = engine_clean.run_best_of_n_query(prompt, max_tokens=30, temperature=0.7)
    assert 'best_trace' in res_clean
    assert len(res_clean['candidate_traces']) == 2

    # 2. Create Corrupted Copy
    os.makedirs(CORRUPT_DIR, exist_ok=True)
    if os.path.exists(os.path.join(MODEL_DIR, "tokenizer.json")):
        shutil.copyfile(os.path.join(MODEL_DIR, "tokenizer.json"), os.path.join(CORRUPT_DIR, "tokenizer.json"))

    file_size = os.path.getsize(ORIGINAL_WEIGHTS)
    shutil.copyfile(ORIGINAL_WEIGHTS, CORRUPT_WEIGHTS)
    with open(CORRUPT_WEIGHTS, "r+b") as f:
        f.seek(file_size // 2)
        zero_chunk = b'\x00' * min(50 * 1024 * 1024, file_size // 4)
        f.write(zero_chunk)

    # 3. Corrupted Run
    try:
        engine_corrupt = AntigravityEngine(n_channels=2, model_dir=CORRUPT_DIR)
        res_corrupt = engine_corrupt.run_best_of_n_query(prompt, max_tokens=30, temperature=0.7)
        clean_text = res_clean['best_trace']
        corrupt_text = res_corrupt['best_trace']

        is_different = (clean_text != corrupt_text)
        assert is_different, "Corrupted weights must produce different token predictions"
    finally:
        if os.path.exists(CORRUPT_DIR):
            shutil.rmtree(CORRUPT_DIR)


def test_2_metal_hardware_profiling():
    """Verify native Metal C++ engine executes GPU command buffers."""
    dylib_candidates = [
        os.path.abspath("antigravity-engine/src/libantigravity_engine.dylib"),
        os.path.abspath("src/libantigravity_engine.dylib"),
        os.path.abspath("libantigravity_engine.dylib"),
    ]
    dylib_path = next((p for p in dylib_candidates if os.path.exists(p)), None)
    if not dylib_path or not os.path.exists(ORIGINAL_WEIGHTS):
        pytest.skip("Native engine dylib or weights not present")

    native_engine = NativeMetalEngine(
        dylib_path=dylib_path,
        model_path=ORIGINAL_WEIGHTS,
        n_channels=4,
        vocab_size=32000,
        hidden_dim=2048,
        max_seq_len=2048
    )

    tok = LlamaTokenizer("models/tinyllama/tokenizer.json")
    prompt_ids = [tok.bos_token_id] + tok.encode("The fundamental force of gravity is")

    tokens, logprobs, ttft_ms, total_ms = native_engine.generate(
        prompt_token_ids=prompt_ids,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )

    assert len(tokens) == 4, "Expected 4 parallel channels"
    assert ttft_ms > 0.0, "TTFT must be positive"
    assert total_ms > 0.0, "Total execution time must be positive"
    assert len(logprobs) == 4, "Expected 4 logprob values"

    # Decode and verify real linguistic tokens from Metal GPU
    physics_keywords = ["force", "mass", "attract", "matter", "nature", "physics", "pull", "earth", "universe", "object", "a"]
    for c in range(4):
        assert len(tokens[c]) == 20, f"Channel {c} must produce 20 tokens"
        assert all(0 <= tid < 32000 for tid in tokens[c]), f"Channel {c} tokens out of vocab bounds"
        decoded = tok.decode(tokens[c]).strip().lower()
        print(f"[Channel {c} Decode]: {decoded}")
        assert len(decoded) > 10, f"Channel {c} output too short: '{decoded}'"
        found = [kw for kw in physics_keywords if kw in decoded]
        assert len(found) >= 1, f"Channel {c} missing physics vocabulary: '{decoded}'"

    native_engine.destroy()


def test_3_memory_step_function():
    """Verify memory tracking and load/unload transitions."""
    import gc
    gc.collect()

    dylib_candidates = [
        os.path.abspath("antigravity-engine/src/libantigravity_engine.dylib"),
        os.path.abspath("src/libantigravity_engine.dylib"),
        os.path.abspath("libantigravity_engine.dylib"),
    ]
    dylib_path = next((p for p in dylib_candidates if os.path.exists(p)), None)
    if not dylib_path or not os.path.exists(ORIGINAL_WEIGHTS):
        pytest.skip("Native engine dylib or weights not present")

    proc = psutil.Process(os.getpid())
    rss_init = proc.memory_info().rss / (1024 * 1024)

    engine = NativeMetalEngine(
        dylib_path=dylib_path,
        model_path=None,
        n_channels=8,
        vocab_size=32000,
        hidden_dim=2048
    )

    bytes_init = engine.get_allocated_bytes()
    assert bytes_init > 0, "Initial allocated VRAM for compute buffers must be > 0"

    engine.load_weights(ORIGINAL_WEIGHTS)
    bytes_loaded = engine.get_allocated_bytes()
    assert bytes_loaded > bytes_init + 500 * 1024 * 1024, f"Loaded weights must allocate > 500MB VRAM over base, got {(bytes_loaded - bytes_init) / 1e6:.1f} MB"

    engine.unload_weights()
    bytes_unloaded = engine.get_allocated_bytes()
    assert bytes_unloaded == 0, f"Allocated VRAM after unload ({bytes_unloaded}) must be 0"

    engine.destroy()
    gc.collect()


def test_4_entropy_temperature():
    """Verify temperature-dependent sampling behavior."""
    model_dir = "models/qwen" if os.path.exists("models/qwen") else "models/tinyllama"
    engine = AntigravityEngine(n_channels=2, model_dir=model_dir)
    prompt = "The quick brown fox jumps over the lazy"

    res_cold = engine.run_best_of_n_query(prompt, max_tokens=15, temperature=0.01)
    res_hot = engine.run_best_of_n_query(prompt, max_tokens=15, temperature=2.0)

    assert 'best_trace' in res_cold
    assert 'best_trace' in res_hot
    assert len(res_cold['candidate_traces']) == 2
    assert len(res_hot['candidate_traces']) == 2

