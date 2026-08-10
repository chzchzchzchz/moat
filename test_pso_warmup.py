"""
Empirical test for Metal PSO JIT Warmup & Host Sync behavior.
Compares Run 1 (Cold / Un-warmed), Run 2 (Warmed), and Run 3 (Hot) across batch sizes N.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'antigravity-engine', 'src'))

import torch
from batch_generator import BatchedRolloutCoordinator

def test_warmup_impact():
    print("=" * 75)
    print("EMPIRICAL TEST: Metal PSO (Pipeline State Object) Cold vs Warm Execution")
    print("=" * 75)

    hidden_dim = 2048
    vocab_size = 32000
    weights = np.random.randn(hidden_dim, vocab_size).astype(np.float16)

    for N in [1, 8, 16]:
        coordinator = BatchedRolloutCoordinator(n_channels=N, hidden_dim=hidden_dim, vocab_size=vocab_size)
        activations = np.random.randn(N, hidden_dim).astype(np.float16)

        # Run 1: COLD (includes Metal PSO creation & initial GPU stream allocations)
        t0 = time.perf_counter()
        _ = coordinator.generate(weights=weights, activations=activations, max_steps=50, temperature=0.7, eos_token_id=-1)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        cold_time = t1 - t0

        # Run 2: WARM (PSO cached, GPU allocations active)
        t0 = time.perf_counter()
        _ = coordinator.generate(weights=weights, activations=activations, max_steps=50, temperature=0.7, eos_token_id=-1)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        warm_time = t1 - t0

        # Run 3: HOT
        t0 = time.perf_counter()
        _ = coordinator.generate(weights=weights, activations=activations, max_steps=50, temperature=0.7, eos_token_id=-1)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        hot_time = t1 - t0

        print(f"Batch N={N:2d} | Run 1 (Cold): {cold_time*1000:6.2f} ms | Run 2 (Warm): {warm_time*1000:6.2f} ms | Run 3 (Hot): {hot_time*1000:6.2f} ms | Speedup (Cold vs Hot): {cold_time/hot_time:4.2f}x")


    print("=" * 75)

if __name__ == "__main__":
    test_warmup_impact()
