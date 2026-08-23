# Phase 3: Project Antigravity 2.0 (The Trifecta)

To achieve absolute dominance on Edge hardware, we will pursue all three architectural breakthroughs simultaneously. They interlock to form a complete, zero-dependency engine.

## 1. Native C++/Metal Engine (Zero-Python)
Currently, our backend relies on the `mlx_lm` Python wrapper. While fast, it carries a heavy Python runtime overhead and cannot be bundled cleanly into a native iOS App Store release. 
**Plan:** We will integrate the existing `deltanet_forward.metal` (Linear Attention) and our `batched_gemm.metal` into `transformer_engine.mm`. We will rip out the MLX Python dependency entirely, resulting in a pure Swift/Objective-C++ stack.

## 2. Speculative Decoding (Draft-Verifier Pipeline)
To maximize Tokens-Per-Second (TPS) on the 4B Reasoner, we will load `Qwen3.5-0.5B` into a secondary, isolated memory buffer. 
**Plan:** The 0.5B model will run ahead, drafting 5-10 tokens instantaneously. The 4B Reasoner will perform a single batched forward pass to verify those tokens. If accepted, we effectively get 5-10 tokens for the cost of 1 memory read.

## 3. MCTS with KV-Cache Tree Pruning (Test-Time Compute)
Our current Test-Time Search is "Best-of-N" (4 independent parallel paths). While batched generation made this extremely fast, it is still mathematically wasteful.
**Plan:** We will implement Monte Carlo Tree Search. Instead of generating 4 independent paths from scratch, the Reasoner will generate a shared logic prefix (e.g., "Step 1: 5 * 10 = 50"), branch into multiple hypotheses, score the intermediate logic, and aggressively prune bad branches *before* finishing them.

---

### Sequence of Attack
I will begin with **Speculative Decoding** in our Python prototype to mathematically prove the TPS gains, then transition it into the **C++/Metal Engine** alongside the **MCTS** logic. 

If this plan aligns with your vision for "all 3", say the word and I will execute immediately.
