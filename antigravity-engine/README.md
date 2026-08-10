# Project Antigravity — Edge Reasoning Engine for Apple Silicon

**Commercial On-Device Reasoning & Verification SDK for iOS and macOS**  
*Powered by Bare-Metal Metal C++ Compute Shaders & Apple Neural Engine (ANE)*

---

## Highlights

- **Native C++ Metal Compute Core**: Executes $N=8$ parallel reasoning rollouts using Apple `simdgroup_matrix` SIMD instructions.
- **Strict 4.5 GB RAM Footprint**: Native weight swapping flushes the 1.1B Reasoner from Metal VRAM before scoring with the 1.5B PRM Verifier.
- **App-Side Weight Manager (`WeightManager.swift`)**: Streaming download and local caching of `.safetensors` model weights directly into sandboxed iOS storage.
- **CoreML Vision Multimodal Encoder (`VisionEncoder.swift`)**: Pre-processes visual problem inputs into 256 embedding vectors using Apple Neural Engine (ANE) acceleration.
- **Universal XCFramework Packaging (`AntigravityEngine.xcframework`)**: Ready for distribution via Swift Package Manager (SPM) or binary embedding on iOS devices (`ios-arm64`), iOS Simulator, and macOS (`arm64`).
- **Developer Kit Demo App (`Examples/AntigravityDemo`)**: SwiftUI single-page demonstration app showing visual camera input, real-time step-level reflection, and candidate verification scores.

---

## Directory Layout

```
antigravity-engine/
├── Package.swift                             # Swift Package Manager (SPM) manifest
├── DOCUMENTATION.md                          # Full developer integration guide
├── Sources/
│   ├── AntigravityEngine/
│   │   ├── AntigravityEngine.swift           # Public Swift SDK wrapper
│   │   ├── WeightManager.swift               # App-side streaming download & caching manager
│   │   ├── VisionEncoder.swift               # CoreML ANE visual patch encoder
│   │   ├── AgentController.swift             # Step-level reflection controller (tau=0.75)
│   │   └── Resources/                        # CoreML assets (SigLIPVisionEncoder.mlpackage)
│   └── CAntigravityEngine/include/           # Public C API headers & module.modulemap
├── frameworks/
│   └── AntigravityEngine.xcframework/        # Pre-built universal Xcode framework
├── Examples/
│   └── AntigravityDemo/                      # Developer Kit SwiftUI Demo App
│       ├── AntigravityDemoApp.swift
│       ├── ContentView.swift
│       └── DemoViewModel.swift
├── scripts/
│   ├── build_xcframework.sh                  # XCFramework build script
│   ├── convert_vision_model.py               # CoreML visual encoder exporter
│   └── download_prm_model.py                 # PRM verifier download script
├── src/
│   ├── transformer_engine.h / .mm            # Objective-C++ Metal decode engine (1,188 lines)
│   ├── shaders/transformer_ops.metal         # 9 Metal compute kernels
│   ├── shaders/batched_gemm.metal            # Bare-metal SIMD group GEMM shader
│   ├── native_bridge.py                      # Python ctypes FFI bridge
│   ├── orchestrator.py                       # Engine orchestrator
│   └── tokenizer.py                          # HuggingFace LlamaTokenizer wrapper
└── tests/                                    # 115 passing unit & integration tests
```

---

## Quick Start (Swift)

```swift
import AntigravityEngine

// 1. Initialize Engine
let engine = try AntigravityEngine(config: .strict4GBFootprint)

// 2. Download and load model weights into Metal VRAM
try await WeightManager.shared.prepareAndLoad(modelType: .reasoner1B, into: engine)

// 3. Execute N=8 parallel reasoning channels
let result = try await engine.reason(promptTokens: [1, 10, 50, 100], maxTokens: 50)

print("Verified Output: \(result.bestTraceText)")
print("Latency: \(result.totalLatencyMs) ms | Throughput: \(result.throughputTokensPerSec) tok/s")
```

See [DOCUMENTATION.md](file:///Users/MohssineChazi2/moat/antigravity-engine/DOCUMENTATION.md) for full integration details and Xcode code signing instructions.
