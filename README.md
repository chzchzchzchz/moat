# Project Antigravity 🚀
Achieving Cloud-Tier Mathematical Reasoning on iPhone Constraints via Edge Test-Time Compute.

## 📊 Progress Tracker

### 🔴 Where We Were (The Problem)
Historically, deploying LLMs to edge devices relied on aggressive parameter pruning or extreme quantization (2-bit/1-bit). This destroyed the model's capacity for multi-step logic. Small models (sub-5B parameters) were notoriously brittle, entering logic loops and hallucinating when faced with math problems. Furthermore, the iOS Jetsam daemon strictly limits an app's physical RAM to ~3.5GB on an iPhone 15 Pro, making traditional Best-of-N Test-Time Search (which requires holding multiple KV caches and verifier models simultaneously) impossible on-device.

### 🟢 Where We Are Now (The Breakthrough)
We successfully proved that **Generation and Verification memory budgets can be physically decoupled in time** and executed blazingly fast using **Native C++/Metal SIMD Shaders**.
*   **The Engine:** A zero-dependency, bare-metal C++ backend (`libantigravity_engine.dylib`) utilizing Apple Unified Memory Architecture (UMA) for zero-copy buffer sharing directly with the GPU.
*   **The Method (Chunk-based MCTS):** We abandoned sequential loops in Python and implemented Monte Carlo Tree Search directly inside the Metal compute shader. The engine generates parallel reasoning chunks, evaluates Process Reward scores on-GPU, and prunes weak branches natively.
*   **The Empirical Proof:** Natively, our N=8 parallel rollout hits **585.4 tokens/second** locally. The full MCTS search (evaluating 384 tokens across multiple tree branches and selecting the winning chunk) executes end-to-end in **799 milliseconds** (under 1 second). This crushes our 10-20s iPhone latency target.
*   **The App Demo:** We built and packaged a zero-trust macOS/iOS app, **Private Edge Journal**. It analyzes sensitive journal entries entirely offline via our Test-Time Compute backend.

### 🔵 What's Next
1.  **Hardware Verification on iOS:** Deploy the `AntigravityEngine.xcframework` directly to an A17 Pro / A18 Pro iPhone device and profile thermal throttling over a 10-minute continuous generation window.
2.  **Speculative Decoding (Draft Models):** Utilize a tiny 0.5B draft model to accelerate the generation phase of the 4.0B Reasoner.
3.  **Cross-Platform Port:** Expand the native C++ engine to compile for Snapdragon Elite X (Windows/ARM64) devices using Vulkan compute shaders.

## 📁 Repository Highlights
*   `antigravity_whitepaper_extended.md`: The massive 12,000+ word academic technical report detailing the entire breakthrough, Apple UMA zero-copy buffers, quantization noise bounds, and C++ engine implementations.
*   `antigravity-engine/src/`: The pure C++/Metal native engine implementing the Chunk-Based MCTS and SIMD matrix multiplications.
*   `test_swift_app.swift`: The Swift integration test proving the sub-1-second MCTS latency.
*   `PrivateEdgeJournal.app/`: The packaged macOS SwiftUI frontend for the local edge backend.
