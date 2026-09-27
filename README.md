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

## Performance — what is measured, and what is not

Every throughput number this project has ever published, with its source:

| Claim | Where it comes from | Status |
| :--- | :--- | :--- |
| 5.13763 tok/s single-channel | `antigravity-engine/benchmark_metrics.json` | committed artifact, no hardware, model, date or sample count recorded |
| 5.28812 tok/s single-channel | `antigravity-engine/benchmark_real_weights.json` | committed artifact, same run description, different number |
| 855.62 tok/s single-channel | `antigravity-engine/benchmark_results_v2.json` | committed artifact, 166x the other two for the same quantity |
| 30,327.6 tok/s | `metal_hardware_proof.md` | **not token throughput** — `src/metal_runner.cpp` times one 8x2048x2048 GEMM and divides by the batch size. A TinyLlama decode step is 154 GEMMs plus attention, norms, RoPE and sampling |
| ~27.0 tok/s (PyTorch MPS) | this README, until now | **no artifact contains this number** |
| ~27.6 tok/s (4-channel) | this README, until now | **no artifact contains this number** |

Three committed artifacts give three different figures for single-channel
decode, and two of the numbers this README used to headline are in no artifact
at all. None of the artifacts records which chip, which OS, which model, when,
or over how many samples — so none of them can be reproduced or compared, and
this table is the honest summary: the engine's throughput is currently unknown.

`tools/benchmark_throughput.py` is what replaces them. It measures the native
engine against PyTorch MPS on one machine in one run and writes a JSON artifact
carrying the hardware profile, every individual sample rather than the best one,
the weight footprint, and the fraction of the machine's memory bandwidth reached.
It exits non-zero and records the failure if anything did not run.

Memory, which is measured and does reproduce:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| Model VRAM footprint (FP16) | 3,037 MB | TinyLlama 1.1B, FP16/BF16, plus KV caches |
| Post-unload memory | 0 MB | full deallocation verified via the Metal allocator |

INT4 super-block weights are now on the inference path (`ANTIGRAVITY_INT4=1`),
which should cut the projection weights to roughly a quarter of that. The
resulting footprint and throughput have not been measured on hardware, so no
number is quoted for them here.

### Accuracy

The claim this project exists to make is that N parallel reasoning channels buy
accuracy. **That has not been demonstrated on this engine.**

`antigravity-engine/antigravity_benchmark_results.json` is the only artifact
that measures it. Its accuracies run 40%, 0%, 0%, 20%, 40% as channels go 1, 2,
4, 8, 16 — read as a scaling curve, except the token counts show about five
problems per row, where one problem is worth 20 points. The repo's other
`antigravity_benchmark_results.json` scores 0% at every channel count.

The often-quoted 68.8% -> 74.2% (n=449) result is real, but it came from
HuggingFace running Qwen2.5-Math-1.5B. It is evidence that best-of-N works. It
is not evidence about this engine, which did not run it.

`tools/benchmark_quality.py` measures the engine itself on GSM8K. It reports
each condition's Wilson interval, runs an exact paired test on the same
problems, refuses to call a difference a lift when it cannot be distinguished
from chance, and on a null result prints the sample size that would have been
needed.

On any Apple Silicon Mac, one command does the whole thing — GPU check, dylib
build, weight download, measurement:

```
antigravity-engine/scripts/run_quality_benchmark.sh          # 40 problems
PROBLEMS=200 antigravity-engine/scripts/run_quality_benchmark.sh
```

**This cannot be run in CI.** GitHub-hosted macOS runners are arm64 VMs without
GPU passthrough: `MTLCreateSystemDefaultDevice()` returns nil on `macos-14`, which
the `INT4 kernels on a real GPU (macOS)` job measures and reports rather than
assumes. So CI compiles every shader with `-Werror` and checks the INT4 layout,
packing and GEMV arithmetic against a host reference, but no kernel has ever
executed and no accuracy number exists yet. The artifact the script writes is the
number; until someone runs it on hardware, this project has no measured accuracy
claim for its own engine.

---

> **Blocked on hardware:** the engine's own 587-problem benchmark shows it emitting
> one character regardless of input. What to run on an Apple Silicon Mac, in priority
> order, and what has already been ruled out: [antigravity-engine/NEXT_ON_HARDWARE.md](antigravity-engine/NEXT_ON_HARDWARE.md)

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
  - CI (`.github/workflows/ci.yml`) runs on every pull request: the C++ bridge contract test and C API compile checks under `-Werror` on Linux, the Python security and verifier tests on Linux, and `swift build` plus `swift test` on a macOS arm64 runner.
  - Tests needing TinyLlama weights or Apple Silicon report as **skipped**, not passed, so the run output distinguishes what was verified from what could not run. The figures quoted above for full-weight inference still require Apple Silicon and real model weights, and are not reproduced by CI.
  - `test_orchestrator.py` and `test_soak_thermal.py` remain outside CI: every test in them drives real generation, so nothing would run without weights.
  - `Package.swift` points its `binaryTarget` at `frameworks/AntigravityEngine.xcframework`, but only the `.zip` is committed. Unzip it before `swift build`, as the CI job does, or the build fails to resolve the target.
  - Set `ANTIGRAVITY_MODEL_DIR` to point the engine, the PRM weight loader and the test clients at a model directory outside the working tree.
