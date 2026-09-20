# TherapistAgent — On-Device Clinical Documentation

A production-grade native macOS/iOS application that generates HIPAA-compliant SOAP clinical notes from psychotherapy session transcripts using **real on-device AI inference on Apple Silicon Metal GPU**. 

**Zero cloud. Zero third-party telemetry. Zero network egress.**

---

## 🏗 System Architecture & Security Highlights

1. **Hardware-Accelerated Inference**: Executes 22-layer transformer forward passes directly in unified Apple Silicon GPU memory via `AntigravityEngine.xcframework` (compiled Metal C++).
2. **Zero-Trust Column-Level Encryption**: Patient records and clinical documents in `ClinicalDatabase.swift` are encrypted using **AES-256-GCM** (Apple CryptoKit) with symmetric keys stored via Keychain Data Protection (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`). Raw SQLite disk files contain zero plaintext PHI.
3. **Real BPE Tokenization**: `AntigravityTokenizer` natively parses HuggingFace `tokenizer.json`, loading full vocabulary (32,000 tokens) and all 61,249 BPE merge rules.
4. **Offline ASR & Diarization**: On-device speech recognition via Apple `Speech` framework (`requiresOnDeviceRecognition = true`) paired with Accelerate `vDSP` spectral feature diarization.
5. **No Fake Fallbacks**: If model weights are missing, the UI presents an honest download banner rather than simulating output.

---

## 📋 Prerequisites

- **Hardware**: Mac with Apple Silicon (M1/M2/M3/M4) or iPhone (A17 Pro+)
- **OS**: macOS 14.0+ (Sonoma/Sequoia) or iOS 17.0+
- **Developer Tools**: Xcode 15.0+ with Swift 5.9+
- **Model Weights**: TinyLlama 1.1B (~2.2 GB) or Qwen 2.5 1.5B (~3 GB) Safetensors

---

## 🚀 Quick Start

### 1. Download Model Weights

Weights can be downloaded automatically inside the app via the **Download TinyLlama 1.1B Model** button, or manually using the CLI:

```bash
pip install huggingface_hub

# Download TinyLlama 1.1B Chat to standard local repository path
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --local-dir ~/moat/models/tinyllama \
    --include "model.safetensors" "tokenizer.json" "tokenizer_config.json"
```

### 2. Build the Package

From the repository root:

```bash
cd antigravity-engine
swift build
```

### 3. Run the Full Test Suite

Execute the 26 native tests (including live GPU forward passes, token-for-token verification, and zero-trust disk checks):

```bash
swift test
```

Expected output:
```text
Test Suite 'All tests' passed at 2026-09-06.
Executed 26 tests, with 0 failures (0 unexpected) in 34.804 seconds.
```

---

## 📱 Standalone Xcode Application Integration

1. Open Xcode and create a new **macOS App** or **iOS App** with SwiftUI.
2. In Project Settings under **Package Dependencies**, click `+` and choose **Add Local...**, selecting `path/to/antigravity-engine`.
3. Add the `AntigravityEngine` library to your target's **Frameworks, Libraries, and Embedded Content**.
4. Link system frameworks:
   - `Metal.framework`
   - `Accelerate.framework`
   - `Speech.framework`
   - `AVFoundation.framework`
   - `Security.framework`
   - `CryptoKit.framework`
   - `libc++.tbd`
5. In your `Info.plist`, declare the following privacy permissions:
   - `NSSpeechRecognitionUsageDescription`: *"On-device clinical transcription with zero network egress."*
   - `NSMicrophoneUsageDescription`: *"On-device audio capture for therapy documentation."*
6. Copy the files in `examples/TherapistAgent/` (`TherapistApp.swift`, `ContentView.swift`, `TherapistViewModel.swift`, `AudioWaveformView.swift`) into your app target.
7. Select your target device (My Mac or iOS Device) and press **Cmd+R** to build and run.

---

## 📂 Source File Catalog

| File | Lines | Purpose |
|------|:-----:|---------|
| `TherapistApp.swift` | 11 | SwiftUI App entrypoint |
| `ContentView.swift` | 310 | Complete clinical user interface (waveform visualizer, template selector, model status card, export buttons) |
| `TherapistViewModel.swift` | 320 | State management, live ASR streaming, candidate evaluation, zero-trust SQLite persistence, and model downloading |
| `AudioWaveformView.swift` | 55 | Real-time audio waveform visualization using 24 dynamic frequency bars |
| `clinical_memory_engine.py` | 510 | Python counterpart for 8-channel clinical rollouts with verified logprobs |

---

## 🔒 Zero-Trust Verification

To verify that sensitive patient information is never exposed in plaintext on the physical drive:

```bash
swift test --filter testClinicalDatabaseZeroTrustColumnLevelEncryption
```

The test injects a patient record and assertively inspects raw binary sectors of the `.sqlite` file on disk:
- `rawDiskBytes` contains **zero plaintext occurrences** of patient names, dates of birth, clinical dialogues, or diagnoses.
- Queries authenticated through the Keychain decrypt seamlessly into typed Swift structs.
