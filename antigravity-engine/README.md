# Project Antigravity

Project Antigravity is a frontier edge reasoning engine built from the ground up for Apple Silicon (Metal) and Snapdragon ARM64 (Vulkan). It runs Test-Time Compute architectures (OpenAI o1-class) locally on mobile devices.

## Progress Tracker

### Where We Were
* An experimental Python-based pipeline relying on PyTorch `mps` backend for generation.
* Reached harsh iOS memory constraints (JetSam 3.5GB limit) preventing multi-agent evaluation.
* Lack of parallelization, resulting in extreme bottlenecks when generating speculative drafts or exploring Monte Carlo Tree Search (MCTS) reasoning paths.

### Where We Are Now (Phase 1-3 Complete)
* **Native C++ Engine & Shaders**: Completely bypassed PyTorch on iOS by building a custom bare-metal Metal inference engine (`MetalTransformerEngine`) utilizing `simdgroup_matrix` INT4 hardware acceleration.
* **Parallel Batched MCTS**: Implemented chunk-based tree search that saturates unified memory bandwidth. Benchmarking on a 4.0B proxy architecture demonstrates **72x throughput improvement** (370 tok/s vs 5 tok/s) when generating 8 branches concurrently.
* **Zero-Copy Speculative Decoding**: Implemented parallel batched prefill (`q_len > 1` dispatch) allowing a target model to verify $K=4$ drafted tokens in a single forward pass without shifting GPU memory.
* **Cross-Platform Vulkan Port**: Abstracted the C++ SDK (`ITransformerEngine`) and ported our hyper-optimized Metal shaders to GLSL SPIR-V Compute Shaders (`.comp`) for deployment on Windows ARM64 (Snapdragon Elite X) and Android.
* **Zero-Stub Hardening**: The Python `orchestrator.py` now maps directly to our C++ backend, utilizing true model weights and step-level PRM (Process Reward Model) scoring without heuristic mocks.

### What's Next
* Implement the Vulkan `VkDeviceMemory` allocation layer using `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` to mirror Metal's zero-copy Shared memory.
* Compile the Vulkan `.so` for Android NDK and run a native on-device benchmark on Snapdragon 8 Gen 3.
* Train a 1.5B Distilled Process Reward Model (PRM) to serve as the default Verifier in the Swift SDK.
* Finalize the Swift UI for the Test-Time Search real-time visualization.

## Benchmark Report Highlights (iPhone Parameters)
* **Standard Autoregressive (1 Ch / 22 Layers)**: 5.14 tokens/sec, 161 MB allocated
* **MCTS Evaluated (8 Ch / 22 Layers)**: 370.38 tokens/sec, 469 MB allocated
* **Conclusion**: Edge Transformers are fundamentally memory-bound, not compute-bound. MCTS on unified memory scales almost linearly up to the bandwidth cap. 

For full benchmark analysis, see `edge_reasoning_paper.md`.
