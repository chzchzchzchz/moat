# Antigravity Engine: High-Performance B2B Edge AI SDK
### Enterprise Developer Guide & Architectural Specification (v2.5.0)

---

## 1. Executive Summary

Antigravity Engine is an ultra-low-latency, zero-cloud C++/Metal inference and speculative reasoning engine designed for Apple Silicon (macOS and iOS). It executes parallel channel rollouts, dynamic architecture execution, and process-reward verifier scoring entirely within unified Apple Silicon memory (VRAM).

### Key Architectural Attributes
- **Zero Cloud Egress**: Strict offline execution with hardware-backed air-gap guarantees.
- **Dynamic Format Parsing**: Native runtime parsing of both GGUF v2/v3 metadata and Safetensors headers without hardcoded model dimensions.
- **Offline Cryptographic Licensing**: Monocypher RFC 8032 SHA-512 Ed25519 offline token verification with hardware binding.
- **Universal Multi-Slice Distribution**: Pre-compiled and codesigned `AntigravityEngine.xcframework` for macOS arm64, iOS arm64 device, and iOS Simulator arm64.
- **Python Native Wheel**: Pip-installable wheel bundling precompiled Metal dynamic libraries and shaders.

---

## 2. Distribution Packages & Checksums

| Package Type | File Path | SPM / SHA256 Checksum |
| :--- | :--- | :--- |
| **XCFramework Archive** | `frameworks/AntigravityEngine.xcframework.zip` | `e0c871d7ed7382946cba5992bc430038c165c8de37e90dc4c56d7f7e7a3c0a9c` |
| **Python Wheel** | `dist/antigravity_engine-2.5.0-py3-none-any.whl` | `abb03f4c60c1285327f8d110b7f5727a0bbcad13adc19f7558c59e0323fda345` |

---

## 3. Swift Package Manager (SPM) Integration

### Binary Target Declaration
Add the binary target to your `Package.swift`:

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "EnterpriseClientApp",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "EnterpriseClientApp", targets: ["EnterpriseClientApp"])
    ],
    targets: [
        .binaryTarget(
            name: "AntigravityEngine",
            url: "https://github.com/chzchzchzchz/moat/releases/download/v1.0.0/AntigravityEngine.xcframework.zip",
            checksum: "e0c871d7ed7382946cba5992bc430038c165c8de37e90dc4c56d7f7e7a3c0a9c"
        ),
        .target(
            name: "EnterpriseClientApp",
            dependencies: ["AntigravityEngine"]
        )
    ]
)
```

---

## 4. Swift SDK Quickstart

```swift
import AntigravityEngine

// 1. Initialize Engine with strict memory bounds
let engine = try AntigravityEngine(config: .strict4GBFootprint)

// 2. Set Enterprise License Key
try engine.setLicenseKey("YOUR_ED25519_LICENSE_TOKEN")

// 3. Load Model (Safetensors or GGUF)
let modelURL = Bundle.main.url(forResource: "model", withExtension: "safetensors")!
try await engine.loadModel(at: modelURL)

// 4. Generate Text with Multi-Channel Verifier Consensus
let result = try await engine.generateText(
    prompt: "Summarize the primary risk factors for cardiovascular disease:",
    maxTokens: 150,
    temperature: 0.3,
    topP: 0.90
)

print("Output: \(result.bestTraceText)")
print("TTFT: \(result.timeToFirstTokenMs) ms | Throughput: \(result.throughputTokensPerSec) tok/s")
```

---

## 5. Python SDK Quickstart

```bash
pip install antigravity_engine-2.5.0-py3-none-any.whl
```

```python
from antigravity_engine import NativeMetalEngine, EngineConfig

# 1. Initialize Native Metal Engine
config = EngineConfig(n_channels=4, vocab_size=32000, hidden_dim=2048, use_metal_gpu=True)
engine = NativeMetalEngine(config=config)

# 2. Set Enterprise License
engine.set_license_key("YOUR_ED25519_LICENSE_TOKEN")

# 3. Load Weights into Unified Metal VRAM
engine.load_weights("models/tinyllama/model.safetensors")
print(f"Loaded VRAM: {engine.get_allocated_bytes() / 1024 / 1024:.1f} MB")

# 4. Execute 4-Channel Speculative Rollout
tokens = [1, 15043, 29892, 590]  # BOS, prompt
result = engine.generate(tokens, max_new_tokens=32, temperature=0.7, top_p=0.9)
print(f"TTFT: {result['ttft_ms']:.2f}ms | Total: {result['total_ms']:.2f}ms")
```

---

## 6. Offline License Key Management

Licenses are cryptographically generated using `tools/license_keygen.py` via Ed25519 signatures:

```bash
# Generate keypair (first-time setup)
python3 tools/license_keygen.py init

# Issue 1-Year Commercial License for 8 Parallel Channels
python3 tools/license_keygen.py generate \
  --licensee "Acme Health Systems" \
  --features clinical,sdk,speculative \
  --max-channels 8 \
  --expiry 2027-12-31

# Verify an existing key
python3 tools/license_keygen.py verify <LICENSE_TOKEN>
```
