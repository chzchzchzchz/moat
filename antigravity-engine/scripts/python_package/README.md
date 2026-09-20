# Antigravity Engine Python SDK

High-performance, bare-metal C++/Metal LLM inference engine for Apple Silicon (macOS & iOS).

## Features
- Full multi-layer Transformer forward pass directly in Apple Silicon Metal GPU
- Zero-copy shared memory architecture (eliminates PyTorch MPS CPU/GPU copies)
- Parallel Best-of-N candidate rollouts ($N=8$ channels)
- Offline Ed25519 cryptographic licensing
- Fully offline local execution (zero cloud network egress)
