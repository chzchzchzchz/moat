# Project Antigravity — Developer Integration Guide & SDK Documentation

**Target Hardware**: Apple Silicon edge hardware (iOS 16+, macOS 13+, M1–M4, A17 Pro, A18 Pro)  
**SDK Package**: `AntigravityEngine` (Swift Package Manager / `.xcframework`)

---

## 1. App-Side Weight Downloading & Local Caching (`WeightManager`)

To keep initial App Store download sizes lightweight, model weights (`model.safetensors` ~1.1 GB for TinyLlama-1.1B, Skywork-1.5B PRM ~2.88 GB) are downloaded asynchronously on first app launch and cached in sandboxed local directories (`Application Support/AntigravityEngine/Models`).

### Swift API Usage

```swift
import AntigravityEngine

// Initialize engine and weight manager
let engine = try AntigravityEngine(config: .strict4GBFootprint)
let weightManager = WeightManager.shared

// Check if weights are already cached locally
if !weightManager.isModelDownloaded(type: .reasoner1B) {
    // Stream download with live progress tracking
    let stream = weightManager.downloadModel(type: .reasoner1B)
    for await progress in stream {
        print("Progress: \(Int(progress.fractionCompleted * 100))% - Speed: \(progress.speedBytesPerSec / 1024 / 1024) MB/s")
    }
}

// Pass sandboxed local file URL directly to C++ Safetensors weight parser
let localURL = weightManager.localURL(for: .reasoner1B)
try engine.loadModel(at: localURL.path)
```

---

## 2. Compiling & Bundling CoreML Vision Assets (`VisionEncoder`)

For multimodal inference (processing visual problem inputs like math equations or logic diagrams), the visual encoder is compiled from PyTorch to Apple CoreML format (`.mlpackage` / `.mlmodelc`) for Apple Neural Engine (ANE) hardware acceleration.

### Exporting PyTorch Vision Model to CoreML

Run the included Python exporter script:

```bash
python3 scripts/convert_vision_model.py --output-dir Sources/AntigravityEngine/Resources
```

This exports `SigLIPVisionEncoder.mlpackage` configured with `Float16` compute precision targeted for ANE execution.

### Swift Multimodal Inference

```swift
import AntigravityEngine
import CoreGraphics

let visionEncoder = VisionEncoder(
    modelURL: Bundle.main.url(forResource: "SigLIPVisionEncoder", withExtension: "mlmodelc")
)

// Encode input problem photo into 256 embedding vectors (2048-dim)
if let cgImage = UIImage(named: "math_problem")?.cgImage {
    let patchEmbeddings = try visionEncoder.encode(image: cgImage)
    
    // Execute multimodal decode pass on Metal GPU
    let result = try await engine.reasonMultimodal(
        textTokens: [1, 10, 100, 200],
        imageEmbeddings: patchEmbeddings,
        patchCount: visionEncoder.patchCount,
        maxTokens: 60
    )
    
    print("Verified Output: \(result.bestTraceText)")
}
```

---

## 3. Physical Device Code Signing & Provisioning (iOS)

When deploying `AntigravityEngine.xcframework` onto physical iPhone devices (iOS arm64):

### Xcode Project Configuration

1. **Import Framework**: Drag `frameworks/AntigravityEngine.xcframework` into your Xcode target under **Target Settings -> General -> Frameworks, Libraries, and Embedded Content**.
2. **Embed & Sign**: Select **Embed & Sign** for `AntigravityEngine.xcframework`.
3. **Code Signing**:
   - Navigate to **Signing & Capabilities**.
   - Check **Automatically manage signing**.
   - Select your valid **Apple Developer Team**.
4. **Build Settings**:
   - `ENABLE_BITCODE = NO` (Default for Xcode 14+)
   - `STRIP_STYLE = non-global`
   - `OTHER_LDFLAGS = -framework Metal -framework CoreML -framework Accelerate`

---

## 4. Step-Level Reflection & Tool-Calling Agent (`AgentController`)

For tool-calling agents, `AgentController` implements step-level reflection gated by the PRM verifier score threshold ($\tau = 0.75$). If initial greedy decode yields a confidence score below $0.75$, the controller automatically triggers $N=8$ parallel reasoning rollouts to refine tool arguments.

```swift
import AntigravityEngine

let agentController = AgentController(
    engine: engine,
    reflectionThreshold: 0.75
)

let result = try await agentController.executeStep(
    promptTokens: [1, 50, 120],
    maxTokens: 50
)

print("Tool Call: \(result.bestTraceText)")
print("Reflection Triggered: \(result.reflectionTriggered)")
```

---

## 5. Running Developer Kit SwiftUI Demo App

Open `Examples/AntigravityDemo/AntigravityDemoApp.swift` in Xcode or import `Examples/AntigravityDemo/ContentView.swift` into any iOS app project for a full interactive demonstration view.
