# Antigravity Engine

Antigravity Engine is a bare-metal C++ transformer inference engine for Apple Silicon. It runs TinyLlama 1.1B (and experimentally Qwen) entirely on-device using Metal GPU compute shaders. No cloud. No Python runtime required for inference.

## Table of Contents
1. [Architecture](#architecture)
2. [Performance](#real-performance-numbers-honest)
3. [Features](#what-works)
4. [Known Limitations](#whats-work-in-progress--known-limitations)
5. [How to Build & Test](#how-to-build--test)
6. [Proof It Works](#proof-it-works)
7. [Project Structure](#project-structure)

## Architecture
- **Core Engine**: C++/Objective-C++ (`src/transformer_engine.mm`) — 22-layer transformer with RMSNorm, Grouped Query Attention, SwiGLU MLP, RoPE, paged KV cache
- **Metal Shaders**: `src/shaders/transformer_ops.metal` (attention, RMSNorm, RoPE, softmax, embedding) + `src/shaders/batched_gemm.metal` (SIMD group 8x8 tiled GEMM)
- **Swift SDK**: `Sources/AntigravityEngine/` wraps the C++ engine as an SPM package via `frameworks/AntigravityEngine.xcframework` (macOS arm64, iOS arm64, iOS Simulator arm64)
- **Python SDK**: `src/native_bridge.py` bridges via ctypes to `libantigravity_engine.dylib`
- **Quantization**: Custom INT4 symmetric quantization with 256-element superblocks (144 bytes each). Also supports GGUF Q4_K_M format loading.
- **Decoding**: 4-8 channel parallel Best-of-N decoding with candidate verification
- **Encryption**: AES-256-GCM column-level encryption for clinical data (Keychain-backed key)

## Real Performance Numbers (HONEST)
- **Native C++ Metal**: ~5.3 tokens/sec single channel (TinyLlama 1.1B on M-series)
- **PyTorch MPS fallback**: ~27 tokens/sec
- **Memory**: 3,037 MB VRAM for TinyLlama 1.1B FP16
- **4-channel parallel decode**: ~27.6 tokens/sec aggregate across channels

## What Works
- ✅ Metal GPU transformer forward pass (real inference)
- ✅ Safetensors model loading via mmap
- ✅ 4-channel parallel Best-of-N generation
- ✅ Swift SDK (SPM xcframework for macOS + iOS)
- ✅ Python SDK (platform wheel with bundled dylib)
- ✅ BPE tokenizer with merge rules
- ✅ AES-256-GCM clinical data encryption
- ✅ Ed25519 license signature verification (Monocypher)
- ✅ Basic speculative decoding (draft + verify)
- ✅ INT4 quantization pipeline
- ✅ GGUF Q4_K_M weight loading
- ✅ OpenAI-compatible REST API server

## What's Work in Progress / Known Limitations
- ⚠️ MCTS is sequential best-of-N beam search, not true Monte Carlo tree search
- ⚠️ Process Reward Model uses token-hash heuristic, not a trained neural verifier
- ⚠️ Speculative decoding uses greedy sampling only (ignores temperature/top_p)
- ⚠️ Softmax kernel has known issue with sequences >32 tokens in batched mode
- ⚠️ Android support is planned but not implemented
- ⚠️ Vulkan backend is planned but not functional
- ⚠️ Network egress monitoring is informational only, not enforced
- ⚠️ Adaptive reflection appends text but doesn't re-run inference

## How to Build & Test

```bash
# Prerequisites: macOS with Apple Silicon, Xcode Command Line Tools

# 1. Download model weights (~2.2GB)
python3 download_model.py  # or manually: huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama

# 2. Build & test Swift SDK
swift build
swift test  # Runs 26 tests against Metal GPU

# 3. Run C++ integration test (requires model weights)
./bin/test_cpp_sdk  # Loads 3GB weights, generates tokens, verifies output

# 4. Run Python tests
pip install dist/antigravity_engine-2.5.0-py3-none-macosx_12_0_arm64.whl
python3 -m pytest tests/ -v  # ~150 tests

# 5. Start OpenAI-compatible server
python3 run_server.py --port 8080
curl http://localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Proof It Works
- The C++ `test_cpp_sdk` loads real 3GB weights and verifies exact token IDs: `token 21737 = " feelings"`, `token 310 = " of"`, `token 14610 = " sadness"`
- 152 Python tests pass covering quantization, attention, batching, Metal GPU, clinical storage
- 26 Swift tests pass including real Metal GPU inference
- Real VRAM allocation: 3,037 MB loaded, 0 bytes after unload

## Project Structure
```text
src/                          # C++ engine core
  transformer_engine.mm       # Metal GPU transformer (22 layers)
  antigravity_c_api.cpp       # C API bridge
  shaders/                    # Metal compute shaders
  native_bridge.py            # Python ctypes bridge
  orchestrator.py             # Multi-path inference orchestrator
Sources/AntigravityEngine/    # Swift SDK
frameworks/                   # Pre-built xcframework
tests/                        # Test suites (Python, Swift, C++)
examples/TherapistAgent/      # Clinical documentation demo app
scripts/                      # Build scripts
```
