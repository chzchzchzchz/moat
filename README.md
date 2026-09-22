# Project Antigravity

Bare-metal C++/Metal GPU transformer inference engine with Swift and Python SDKs designed for Apple Silicon (macOS and iOS).

Antigravity executes local, private large language model inference on-device using custom Apple Metal compute shaders. It runs TinyLlama 1.1B (and experimentally Qwen) entirely within edge memory constraints without relying on external cloud APIs or heavyweight runtime frameworks.

---

## Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [How It Works](#how-it-works)
   - [Core C++/Metal Engine](#1-core-cmetal-gpu-engine)
   - [Unified Memory Architecture (UMA)](#2-zero-copy-unified-memory)
   - [Parallel Best-of-N Search](#3-parallel-best-of-n-search)
   - [Swift & Python SDKs](#4-swift-and-python-sdks)
   - [Security & Encryption](#5-security-and-data-protection)
3. [Empirical Performance (Verified)](#empirical-performance-verified)
4. [Proof That It Works](#proof-that-it-works)
5. [Repository Structure](#repository-structure)
6. [Quickstart & Build Instructions](#quickstart--build-instructions)
7. [Current Status & Known Limitations](#current-status--known-limitations)

---

## What This Project Is

Antigravity is an on-device inference stack engineered specifically for Apple Silicon hardware:

- **Zero Cloud Dependence**: Full transformer forward pass, KV cache management, and token sampling run locally on the device GPU.
- **Edge Budget Compliant**: Designed to operate within iOS physical RAM limitations (~3.5 GB Jetsam memory boundary on iPhone 15 Pro) and Apple M-series Macs.
- **Multi-Platform Integration**: Includes an Objective-C++/C API, a native Swift Package (`AntigravityEngine.xcframework`) for iOS/macOS applications, and a Python `ctypes` bridge with an OpenAI-compatible REST server.
- **Encrypted Local Persistence**: AES-256-GCM column-level database encryption using keys derived from Apple Keychain Data Protection.

---

## How It Works

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                              │
│   SwiftUI (TherapistAgent, Private Journal)   │   Python / REST Server │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                          SDK Interfaces                                │
│   Swift SDK (Package.swift)       │   Python SDK (native_bridge.py)    │
│   - AntigravityTokenizer (BPE)    │   - ModelWeightLoader (Safetensors)│
│   - ClinicalDatabase (AES-256-GCM)│   - ZeroEgressNetworkAuditor       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                       C API (antigravity_c_api.cpp)                     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│                  Metal Transformer Core Engine                         │
│                    (src/transformer_engine.mm)                         │
│                                                                        │
│   [RMSNorm] ──► [QKV Proj] ──► [RoPE] ──► [Attention] ──► [SwiGLU]   │
│                                                                        │
│   Metal Shaders:                                                       │
│   - transformer_ops.metal: Attention, RMSNorm, RoPE, Softmax, Embed    │
│   - batched_gemm.metal:    8x8 SIMD-group tiled matrix multiplication   │
│   - Memory: MTLResourceStorageModeShared (Zero-Copy UMA)               │
└────────────────────────────────────────────────────────────────────────┘
```

### 1. Core C++/Metal GPU Engine
The inference pipeline is implemented in `antigravity-engine/src/transformer_engine.mm`. For TinyLlama 1.1B, the engine executes a complete 22-layer transformer:
- **RMSNorm**: Parallel reductions across hidden dimension H = 2048.
- **Attention**: Grouped Query Attention (32 query heads, 4 key/value heads) with causal masking and paged KV cache buffers.
- **RoPE**: Rotary Position Embeddings computed dynamically in half-precision Metal kernels.
- **SwiGLU MLP**: Gated linear units with SiLU activations projecting from 2048 to 5632 dimensions and back.
- **Tiled GEMM**: SIMD-group matrix multiplication using Metal threadgroup memory tiles (`batched_gemm.metal`).

### 2. Zero-Copy Unified Memory
On Apple Silicon, CPU and GPU share high-bandwidth physical memory. Antigravity loads weights from `.safetensors` files using `mmap()` directly into `MTLResourceStorageModeShared` buffers, avoiding redundant CPU-to-GPU data copies during model loading.

### 3. Parallel Best-of-N Search
The engine supports multi-channel parallel decoding (e.g., N=4 channels). During generation:
- The prefill pass processes prompt tokens across active channels.
- Autoregressive decode steps run batched GEMM projections across all N channels in a single Metal dispatch.
- Log-probabilities are accumulated at each step to score and select winning candidate completions.

### 4. Swift and Python SDKs
- **Swift SDK** (`antigravity-engine/Sources/AntigravityEngine/`): Exposes an async Swift interface, bundled as `AntigravityEngine.xcframework`. Includes a byte-level BPE tokenizer matching HuggingFace tokenization rules, clinical SOAP note synthesis templates, and CoreText PDF export.
- **Python SDK** (`antigravity-engine/src/native_bridge.py`): Interfaces directly with `libantigravity_engine.dylib` via `ctypes`. Provides PyTorch MPS fallbacks, INT4/Q4_K_M dequantization utilities, and an OpenAI-compatible local HTTP server.

### 5. Security and Data Protection
- **Zero Network Egress**: Inference runs locally with no external socket connections. The included `ZeroEgressNetworkAuditor` monitors OS socket counters to verify no outbound data leaves during processing.
- **Model-Generated Code Execution (off by default)**: `GenPRMVerifier` can score a reasoning trace by running the Python the model emitted and comparing its output to the stated answer. Because the model's output is shaped by whatever text reaches the prompt, enabling this turns a prompt injection into code execution on the host, with filesystem and network access — which would also break the zero-egress property above. It is therefore disabled unless you pass `enable_code_execution=True`. When enabled, the subprocess is limited by a wall-clock timeout, `RLIMIT_AS` at `max_memory_mb` (POSIX), a scrubbed environment, an empty working directory and `python -I`. That is confinement, not a sandbox: there is no syscall filter, namespace, or network restriction.
- **Encrypted Persistence**: Clinical notes and patient identifiers are encrypted at rest using AES-256-GCM via Apple CryptoKit. Keys are retrieved from the macOS/iOS Keychain (`kSecClassGenericPassword`), and decryption operations authenticate tag integrity before releasing plaintext.

---

## Empirical Performance (Verified)

Performance measured on Apple Silicon M-series hardware with TinyLlama 1.1B:

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Single-Channel Native Metal** | **~5.3 tokens/sec** | End-to-end custom C++ Metal compute shaders |
| **4-Channel Parallel Decode** | **~27.6 aggregate tokens/sec** | 4 parallel rollout channels executed simultaneously |
| **PyTorch MPS Fallback** | **~27.0 tokens/sec** | Single-channel reference via PyTorch MPS |
| **Model VRAM Footprint** | **3,037 MB** | TinyLlama 1.1B in FP16 / BF16 representation |
| **Post-Unload Memory** | **0 MB** | Full VRAM deallocation verified via Metal allocator |

> **Note on Benchmarking**: Early iterations of this codebase contained mock loops that reported unverified throughputs (e.g., 243 tok/s). The figures above represent actual measured hardware execution with real weights generating verified text.

---

## Proof That It Works

The engine is verified through three independent test suites across C++, Swift, and Python:

### 1. C++ Native Metal Test (`./bin/test_cpp_sdk`)
- **Status**: **10/10 tests passed (100%)**
- Loads 3,037 MB of real TinyLlama weights into Metal VRAM.
- Executes 4-channel parallel rollouts on the GPU.
- Verifies exact generated vocabulary tokens (e.g., `token 21737 = " feelings"`, `token 310 = " of"`, `token 14610 = " sadness"`).
- Confirms candidate verification and clean VRAM deallocation to 0 bytes.

### 2. Swift SDK Test Suite (`swift test`)
- **Status**: **26/26 tests passed (100%)**
- Validates Swift / C++ bridge integration.
- Tests real Metal GPU allocation and execution.
- Generates structured SOAP clinical notes from live model completions.
- Verifies BPE tokenizer merge rules against the HuggingFace vocabulary.
- Tests AES-256-GCM encryption roundtrips and Keychain error handling.

### 3. Python Test Suite (`pytest tests/`)
- **Status**: **153 passed, 1 skipped (100% runnable passing)**
- Verifies GGUF and Safetensors model weight parsing.
- Validates INT4 256-element superblock dequantization (144 bytes per block).
- Tests attention kernel operations, memory budget limits, and orchestrator routing.
- Confirms zero-egress network isolation and clinical memory store persistence.

---

## Repository Structure

```text
moat/
├── README.md                               # Project documentation and architecture guide
└── antigravity-engine/                     # Core engine workspace
    ├── bin/
    │   └── test_cpp_sdk                    # Native C++ verification binary
    ├── distribution/
    │   └── Package.swift                   # Binary SPM distribution specification
    ├── examples/
    │   └── TherapistAgent/                 # Clinical documentation demo & memory store
    ├── frameworks/
    │   └── AntigravityEngine.xcframework   # Pre-built universal Apple Silicon framework
    ├── models/
    │   └── tinyllama/                      # Model weights (gitignored, download locally)
    ├── scripts/
    │   ├── build_metal_lib.sh              # Compile .metal shaders to .metallib
    │   ├── build_xcframework.sh            # Package static libs into xcframework
    │   └── build_python_wheel.sh           # Build platform-specific Python wheel
    ├── Sources/
    │   └── AntigravityEngine/              # Swift SDK source code
    │       ├── AntigravityEngine.swift     # High-level engine interface
    │       ├── ClinicalDatabase.swift      # AES-256-GCM encrypted persistence
    │       ├── PDFExportService.swift      # CoreText PDF rendering
    │       ├── SOAPTemplate.swift          # Clinical documentation parser
    │       ├── SpeakerDiarizer.swift       # Audio feature extraction & diarization
    │       ├── Tokenizer.swift             # BPE tokenizer with GPT-2 byte mapping
    │       └── WeightManager.swift         # On-device weight downloader and cache
    ├── src/
    │   ├── antigravity_c_api.cpp           # C interface bridge
    │   ├── model_loader.py                 # Weight parsing and dequantization
    │   ├── native_bridge.py                # Python ctypes dylib binding
    │   ├── orchestrator.py                 # Multi-backend inference router
    │   ├── transformer_engine.h            # C++ engine header
    │   ├── transformer_engine.mm           # Metal GPU 22-layer transformer engine
    │   └── shaders/
    │       ├── batched_gemm.metal          # SIMD-group matrix multiplication kernel
    │       └── transformer_ops.metal       # Attention, RMSNorm, RoPE, Softmax shaders
    └── tests/
        ├── AntigravityEngineTests/         # Swift XCTest suite (26 tests)
        ├── test_cpp_sdk.cpp                # C++ integration test source
        ├── test_model_loader.py            # Python model loader unit tests
        └── test_therapist_sentinel.py      # Clinical pipeline and security tests
```

---

## Quickstart & Build Instructions

### Prerequisites
- macOS Sonoma (14.0+) or later on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.10+ (for Python SDK and pytest)

### 1. Download Model Weights
Download the TinyLlama 1.1B Chat Safetensors model (~2.2 GB):
```bash
# Using huggingface-cli
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0   --local-dir antigravity-engine/models/tinyllama   --include "*.safetensors" "config.json" "tokenizer.json"
```

### 2. Build & Test Swift SDK
```bash
cd antigravity-engine
swift build
swift test
```

### 3. Run Native C++ Verification
```bash
# Compile test runner
clang -O3 -c src/monocypher.c -Isrc -o /tmp/monocypher.o
clang -O3 -c src/monocypher-ed25519.c -Isrc -o /tmp/monocypher_ed.o
clang++ -O3 -std=c++17 -Wall -x objective-c++ -fobjc-arc -I./src \
  tests/test_cpp_sdk.cpp src/transformer_engine.mm src/antigravity_c_api.cpp \
  src/antigravity_engine_c.cpp src/gguf_reader.cpp src/config_parser.cpp src/license_verifier.cpp \
  -x none /tmp/monocypher.o /tmp/monocypher_ed.o \
  -framework Metal -framework Foundation \
  -o bin/test_cpp_sdk

# Execute test suite
./bin/test_cpp_sdk
```

### 4. Run Python Tests
```bash
cd antigravity-engine
pytest tests/ -v
```

### 5. Launch Local OpenAI-Compatible Server
```bash
cd antigravity-engine
python3 run_server.py --port 8080 --model-path models/tinyllama

# Send completion request
curl http://localhost:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "tinyllama-1.1b",
    "messages": [{"role": "user", "content": "What are symptoms of anxiety?"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

---

## Current Status & Known Limitations

To maintain full transparency, here is the current engineering status of all system components:

- **What Is Real & Functional**:
  - 22-layer transformer forward pass on Apple Silicon GPU via Metal compute shaders.
  - Safetensors memory-mapped model loading.
  - Multi-channel parallel Best-of-N candidate decoding.
  - Swift SPM package with iOS/macOS `.xcframework` binary target.
  - Python ctypes native bridge and OpenAI-compatible API server.
  - Standard BPE tokenizer with merge table matching HuggingFace.
  - AES-256-GCM column encryption backed by Keychain Data Protection.

- **Work in Progress & Roadmapped**:
  - **Trained Verifier**: The current Process Reward Model (PRM) uses token-frequency and logprob heuristics — concretely `logprob / len^0.6 + unique_token_ratio * 3.0 + log1p(len) * 0.5`, hand-tuned rather than learned. Training an on-device neural verifier is in progress.
  - **Tree Search**: Despite the name, `generateMCTS` / `AntigravityEngineNativeMCTSGenerate` is **not** Monte Carlo Tree Search. It is a chunk-wise greedy hill climb: each round generates N continuations, scores them with the PRM heuristic, appends the single best one and moves on. There is no tree, no visit counts, no UCT selection and no backpropagation, so it cannot recover from an early wrong turn. The name is retained because it is part of the published ABI. Real UCT expansion and backpropagation are planned.
  - **Speculative Sampling**: `AntigravityEngineNativeGenerateSpeculative` accepts `temperature` and `top_p` but **ignores them** — both draft and target decode greedily, because the acceptance test is an exact match against the target's greedy pick, which is only distribution-correct for greedy. Proper stochastic speculative sampling needs a probability-ratio accept/reject step and is roadmapped. Output is also single-channel regardless of `n_channels`.
  - **Cross-Platform**: Windows ARM64 and Vulkan shader backends are experimental drafts and not yet functional for production use.

- **Notes on Verification**:
  - The C++, Swift and Python suite results quoted above were measured on Apple Silicon. They require the Apple toolchain and real model weights, so they cannot be reproduced on Linux or in a GPU-less CI runner.
  - `tests/test_bridge_contract.cpp` is the exception: it links the pure-C++ bridge against stub implementations of the Metal C API and runs anywhere, with no GPU and no weights.
  - Set `ANTIGRAVITY_MODEL_DIR` to point the engine, the PRM weight loader and the test clients at a model directory outside the working tree.
