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

## Performance

**Throughput is currently unknown.** Three committed artifacts disagree about
single-channel decode — `benchmark_metrics.json` says 5.13763 tok/s,
`benchmark_real_weights.json` says 5.28812, and `benchmark_results_v2.json` says
855.62 — and the "~27 tok/s" and "~27.6 tok/s" figures this section used to give
appear in no artifact at all. None of the files records the chip, the OS, the
model, the date or the sample count, so none can be reproduced or compared.

The 30,327.6 tok/s in `metal_hardware_proof.md` measures something else again:
`src/metal_runner.cpp` times one 8x2048x2048 GEMM and divides by the batch size.
A TinyLlama decode step is 154 GEMMs plus attention, norms, RoPE and sampling,
which is why that number and the 5 tok/s ones differ by four orders of magnitude.

`tools/benchmark_throughput.py` replaces them: native engine against PyTorch MPS
in one run on one machine, writing an artifact that carries the hardware
profile, every sample, the weight footprint and the fraction of memory bandwidth
reached, and exiting non-zero if anything failed.

Measured and reproducible:
- **Memory**: 3,037 MB VRAM for TinyLlama 1.1B in FP16, 0 MB after unload

INT4 super-block weights are on the inference path behind `ANTIGRAVITY_INT4=1`
and should cut the projection weights to about a quarter of that. Not yet
measured on hardware, so no figure is quoted.

**Accuracy from parallel channels has not been demonstrated on this engine.**
The only artifact that measures it covers about five problems per row, where one
problem is worth 20 points.

`scripts/run_quality_benchmark.sh` measures it properly, in one command on any
Apple Silicon Mac: it checks the GPU, verifies the INT4 kernels against a host
reference, builds the dylib, fetches weights, and runs GSM8K reporting Wilson
intervals and an exact paired test.

It cannot run in CI. `MTLCreateSystemDefaultDevice()` returns nil on GitHub's
`macos-14` runners — they are arm64 VMs with no GPU — which the `INT4 kernels on a
real GPU (macOS)` job reports rather than assumes. CI therefore compiles every
shader with `-Werror` and checks the super-block layout, the packer against
`src/dequant.py`, and the GEMV arithmetic against a host reference; it has never
executed a kernel.

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
- 153 Python tests pass covering quantization, attention, batching, Metal GPU, clinical storage
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
