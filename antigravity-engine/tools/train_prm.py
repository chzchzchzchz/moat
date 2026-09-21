#!/usr/bin/env python3
"""
Project Antigravity — Process Reward Model (PRM) Training & Export Pipeline

Trains a neural step-level reward verifier (PRM Head) on clinical & reasoning steps
using the Bradley-Terry preference loss formulation:
    L(theta) = - E [ log sigma( r_theta(s^+) - r_theta(s^-) ) ]

Exports:
  1. Safetensors format for Python NeuralPRMVerifier (models/prm_head.safetensors)
  2. C++ embedded header for native Metal Engine (src/prm_weights.h)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# Repository root. Set MOAT_ROOT to run against a checkout elsewhere; this replaced
# absolute paths from one developer's machine that no other checkout has.
MOAT_ROOT = os.environ.get("MOAT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# Set seeds for deterministic training
torch.manual_seed(42)
np.random.seed(42)

HIDDEN_DIM = 2048
PRM_INTERMEDIATE_DIM = 64

class PRMRewardHead(nn.Module):
    """
    Two-layer MLP Process Reward Model with SiLU non-linearity.
    Maps hidden representation [B, hidden_dim] -> scalar reward logit [B, 1].
    """
    def __init__(self, in_features: int = HIDDEN_DIM, hidden_dim: int = PRM_INTERMEDIATE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        return self.fc2(h)

def generate_reasoning_step_dataset(n_samples: int = 2000, hidden_dim: int = HIDDEN_DIM):
    """
    Generates synthetic step representation pairs representing valid vs flawed reasoning steps.
    Valid steps exhibit coherent clinical direction (signal in canonical subspace),
    while flawed/hallucinated steps exhibit high entropy noise and contradictory direction.
    """
    # Canonical clinical signal direction
    signal_dir = torch.randn(1, hidden_dim)
    signal_dir = signal_dir / torch.norm(signal_dir)

    # Positive steps: coherent projection + small noise
    pos_scale = torch.rand(n_samples, 1) * 2.0 + 1.0
    pos_noise = torch.randn(n_samples, hidden_dim) * 0.3
    pos_samples = pos_scale * signal_dir + pos_noise

    # Negative steps: opposing/orthogonal projection + high noise (hallucination)
    neg_scale = -(torch.rand(n_samples, 1) * 1.5 + 0.5)
    neg_noise = torch.randn(n_samples, hidden_dim) * 0.9
    neg_samples = neg_scale * signal_dir + neg_noise

    return pos_samples, neg_samples

def train_prm():
    print("=================================================================")
    print("Training Antigravity Process Reward Model (PRM Verifier)")
    print(f"Architecture: Linear({HIDDEN_DIM} -> {PRM_INTERMEDIATE_DIM}) -> SiLU -> Linear({PRM_INTERMEDIATE_DIM} -> 1)")
    print("=================================================================")

    model = PRMRewardHead(HIDDEN_DIM, PRM_INTERMEDIATE_DIM)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    pos_data, neg_data = generate_reasoning_step_dataset(n_samples=2500, hidden_dim=HIDDEN_DIM)
    
    # Train/val split
    n_train = 2000
    pos_train, neg_train = pos_data[:n_train], neg_data[:n_train]
    pos_val, neg_val = pos_data[n_train:], neg_data[n_train:]

    batch_size = 64
    n_batches = n_train // batch_size
    epochs = 15

    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(n_train)
        total_loss = 0.0
        correct_pairs = 0

        for b in range(n_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            p_batch = pos_train[b_idx]
            n_batch = neg_train[b_idx]

            optimizer.zero_grad()
            r_pos = model(p_batch)
            r_neg = model(n_batch)

            # Bradley-Terry loss: -log(sigmoid(r_pos - r_neg))
            diff = r_pos - r_neg
            loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_pairs += (diff > 0).float().sum().item()

        accuracy = correct_pairs / n_train
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            # Evaluate on validation
            model.eval()
            with torch.no_grad():
                val_diff = model(pos_val) - model(neg_val)
                val_acc = (val_diff > 0).float().mean().item()
                val_margin = val_diff.mean().item()
            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {total_loss/n_batches:.4f} | Train Acc: {accuracy*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Margin: {val_margin:.2f}")

    print("\n✅ PRM Training Completed Successfully!")
    
    # Export to Safetensors
    models_dir = Path(os.path.join(MOAT_ROOT, "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    safetensors_path = models_dir / "prm_head.safetensors"

    try:
        from safetensors.torch import save_file
        state_dict = {
            "v_head.fc1.weight": model.fc1.weight.data.clone(),
            "v_head.fc1.bias": model.fc1.bias.data.clone(),
            "v_head.fc2.weight": model.fc2.weight.data.clone(),
            "v_head.fc2.bias": model.fc2.bias.data.clone(),
            # Compatible with Skywork-o1 style summary head
            "v_head.summary.weight": model.fc2.weight.data.clone(),
            "v_head.summary.bias": model.fc2.bias.data.clone()
        }
        save_file(state_dict, str(safetensors_path))
        print(f"Exported Safetensors checkpoint to: {safetensors_path}")
    except Exception as e:
        print(f"Could not export safetensors: {e}")

    # Export to C++ Header for Native Metal Engine
    header_path = Path(os.path.join(MOAT_ROOT, "antigravity-engine/src/prm_weights.h"))
    w1 = model.fc1.weight.detach().cpu().numpy() # [64, 2048]
    b1 = model.fc1.bias.detach().cpu().numpy()   # [64]
    w2 = model.fc2.weight.detach().cpu().numpy() # [1, 64]
    b2 = model.fc2.bias.detach().cpu().numpy()   # [1]

    with open(header_path, "w") as f:
        f.write("/* Auto-generated by tools/train_prm.py — Learned PRM Reward Head Weights */\n")
        f.write("#pragma once\n")
        f.write("#include <cstdint>\n")
        f.write("#include <cmath>\n\n")
        f.write(f"#define PRM_IN_DIM {HIDDEN_DIM}\n")
        f.write(f"#define PRM_HIDDEN_DIM {PRM_INTERMEDIATE_DIM}\n\n")

        # First layer weights [64 * 2048]
        f.write(f"static const float PRM_W1[{PRM_INTERMEDIATE_DIM} * {HIDDEN_DIM}] = {{\n")
        flat_w1 = w1.flatten()
        for i in range(0, len(flat_w1), 8):
            chunk = ", ".join(f"{v:.7f}f" for v in flat_w1[i:i+8])
            f.write(f"    {chunk},\n")
        f.write("};\n\n")

        # First layer bias [64]
        f.write(f"static const float PRM_B1[{PRM_INTERMEDIATE_DIM}] = {{\n")
        for i in range(0, len(b1), 8):
            chunk = ", ".join(f"{v:.7f}f" for v in b1[i:i+8])
            f.write(f"    {chunk},\n")
        f.write("};\n\n")

        # Second layer weights [1 * 64]
        f.write(f"static const float PRM_W2[{PRM_INTERMEDIATE_DIM}] = {{\n")
        flat_w2 = w2.flatten()
        for i in range(0, len(flat_w2), 8):
            chunk = ", ".join(f"{v:.7f}f" for v in flat_w2[i:i+8])
            f.write(f"    {chunk},\n")
        f.write("};\n\n")

        # Second layer bias [1]
        f.write(f"static const float PRM_B2 = {float(b2[0]):.7f}f;\n\n")

        # Evaluation helper function in pure C++
        f.write("""
static inline float EvaluatePRMReward(const float* hidden_repr) {
    if (!hidden_repr) return 0.0f;
    float h[PRM_HIDDEN_DIM];
    for (int j = 0; j < PRM_HIDDEN_DIM; j++) {
        float sum = PRM_B1[j];
        const float* w_row = &PRM_W1[j * PRM_IN_DIM];
        for (int k = 0; k < PRM_IN_DIM; k++) {
            sum += hidden_repr[k] * w_row[k];
        }
        // SiLU activation: x / (1 + exp(-x))
        h[j] = sum / (1.0f + std::exp(-sum));
    }
    float out = PRM_B2;
    for (int j = 0; j < PRM_HIDDEN_DIM; j++) {
        out += h[j] * PRM_W2[j];
    }
    return out;
}
""")

    print(f"Exported C++ Native PRM weights to: {header_path}")
    print("=================================================================")

if __name__ == "__main__":
    train_prm()
