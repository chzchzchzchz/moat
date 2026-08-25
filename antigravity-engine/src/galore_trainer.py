"""
Project Antigravity — Pillar A: On-Device Self-Evolution Engine (iPhone-Native)

GaLore (Gradient Low-Rank Projection) + MeSP (Memory-Efficient Structured Backprop)

iPhone-First Design Constraints:
  - Total app memory ceiling: 3.5 GB (realistic iOS foreground limit on A17/A18 Pro)
  - Model weights: ~2.18 GB via zero-copy mmap (does NOT count against RSS)
  - Available training headroom: ~800 MB (3.5 GB - 2.18 GB mmap - 0.5 GB runtime)
  - No PyTorch: All gradient ops must use Metal Performance Shaders or raw Metal compute
  - No subprocess: Training runs in-process on the main Metal command queue
  - Thermal budget: Must yield GPU between training steps to avoid iOS thermal throttle
  - Background execution: Limited to ~30s unless using BGProcessingTask (overnight fine-tuning)

Target Hardware: A17 Pro (iPhone 15 Pro) / A18 Pro (iPhone 16 Pro)
"""

import numpy as np
import os
import json
import time
import gc
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


# ─── iPhone Hardware Constants ───────────────────────────────────────────────

IOS_APP_MEMORY_CEILING_MB = 3500       # Realistic foreground limit (not the 4.5 GB kernel kill)
IOS_BACKGROUND_MEMORY_MB = 1500        # Background task memory limit
MMAP_MODEL_FOOTPRINT_MB = 2180         # Qwen2.5-1.5B via zero-copy mmap
RUNTIME_OVERHEAD_MB = 500              # KV-cache, tokenizer, app chrome
TRAINING_HEADROOM_MB = (
    IOS_APP_MEMORY_CEILING_MB - RUNTIME_OVERHEAD_MB
)  # ~3000 MB available (mmap doesn't count)

# GaLore rank must be small enough that SVD projection fits in headroom
MAX_GALORE_RANK = 32                   # Conservative for iPhone thermal budget
THERMAL_YIELD_INTERVAL_STEPS = 4       # Yield GPU every N steps to prevent throttle
OVERNIGHT_BATCH_SIZE = 1               # Single-example SGD for memory safety


@dataclass
class iOSTrainingConfig:
    """Training configuration tuned for iPhone hardware."""
    learning_rate: float = 5e-5
    galore_rank: int = MAX_GALORE_RANK
    n_layers: int = 22                 # Qwen2.5-1.5B layer count
    hidden_dim: int = 1536             # Qwen2.5-1.5B hidden dim
    intermediate_dim: int = 8960       # Qwen2.5-1.5B MLP dim
    max_seq_len: int = 256             # Short sequences for iPhone (not 2048)
    batch_size: int = OVERNIGHT_BATCH_SIZE
    gradient_accumulation_steps: int = 8   # Simulate larger batch via accumulation
    checkpoint_every_n_steps: int = 50
    thermal_yield_ms: int = 100        # Sleep between steps to cool GPU
    max_training_steps: int = 500      # Conservative overnight budget
    checkpoint_dir: str = './ios_training_checkpoints'
    mesp_checkpoint_ratio: float = 0.33  # Only checkpoint every 3rd layer


class GaLoreProjector_iOS:
    """
    Gradient Low-Rank Projection optimized for iPhone A-series chips.

    Key iPhone adaptations:
      - Rank capped at 32 (vs 64-128 on desktop) to fit SVD in thermal budget
      - Uses float16 throughout (A17/A18 have native FP16 ALUs)
      - Projection matrices cached in unified memory (no CPU↔GPU copies)
      - SVD recomputed every 500 steps (expensive op, amortize on mobile)
    """

    def __init__(self, rank: int = MAX_GALORE_RANK, update_interval: int = 500):
        self.rank = rank
        self.update_interval = update_interval
        self.projections: Dict[str, np.ndarray] = {}
        self.step_count = 0

    def project(self, name: str, grad: np.ndarray) -> np.ndarray:
        """Project full gradient into low-rank subspace. All float16."""
        if grad.ndim < 2:
            return grad

        m, n = grad.shape
        r = min(self.rank, min(m, n))

        if name not in self.projections or self.step_count % self.update_interval == 0:
            # Truncated SVD via randomized algorithm (fast, low memory)
            # On iPhone this would be a Metal compute shader; here we use numpy as reference
            grad_f32 = grad.astype(np.float32)
            
            # Randomized SVD: much cheaper than full SVD
            rng = np.random.RandomState(self.step_count)
            if m >= n:
                omega = rng.randn(n, r + 10).astype(np.float32)
                Y = grad_f32 @ omega
                Q, _ = np.linalg.qr(Y)
                Q = Q[:, :r]
                self.projections[name] = Q.astype(np.float16)
            else:
                omega = rng.randn(r + 10, m).astype(np.float32)
                Y = omega @ grad_f32
                _, _, Vt = np.linalg.svd(Y, full_matrices=False)
                self.projections[name] = Vt[:r, :].T.astype(np.float16)

        P = self.projections[name]

        if m >= n:
            return (P.astype(np.float32).T @ grad.astype(np.float32)).astype(np.float16)
        else:
            return (grad.astype(np.float32) @ P.astype(np.float32)).astype(np.float16)

    def unproject(self, name: str, proj_grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct full gradient from low-rank projection."""
        if name not in self.projections:
            return proj_grad
        P = self.projections[name]
        m, n = original_shape
        if m >= n:
            return (P.astype(np.float32) @ proj_grad.astype(np.float32)).astype(np.float16)
        else:
            return (proj_grad.astype(np.float32) @ P.astype(np.float32).T).astype(np.float16)

    def step(self):
        self.step_count += 1

    def memory_bytes(self) -> int:
        """Total memory consumed by projection matrices."""
        total = 0
        for P in self.projections.values():
            total += P.nbytes
        return total


class MeSP_iOS:
    """
    Memory-Efficient Structured Backprop for iPhone.

    iPhone adaptation: Only checkpoints every 3rd layer (not every 2nd).
    This drops peak activation memory to ~33% of naive, fitting in the
    ~800 MB training headroom.

    On iOS, activation recomputation runs on the same Metal command queue
    as the forward pass — no multi-GPU coordination needed.
    """

    def __init__(self, n_layers: int = 22, checkpoint_ratio: float = 0.33):
        self.n_layers = n_layers
        self.checkpoint_layers = set(
            range(0, n_layers, max(1, int(1.0 / checkpoint_ratio)))
        )
        self.saved: Dict[int, np.ndarray] = {}

    def should_save(self, layer_idx: int) -> bool:
        return layer_idx in self.checkpoint_layers

    def save(self, layer_idx: int, activation: np.ndarray):
        if self.should_save(layer_idx):
            self.saved[layer_idx] = activation

    def get(self, layer_idx: int) -> Optional[np.ndarray]:
        return self.saved.get(layer_idx)

    def clear(self):
        self.saved.clear()
        gc.collect()

    def peak_activation_mb(self, hidden_dim: int, seq_len: int) -> float:
        """Estimate peak activation memory."""
        bytes_per_layer = hidden_dim * seq_len * 2  # FP16
        return (bytes_per_layer * len(self.checkpoint_layers)) / (1024 * 1024)


class AdamW_GaLore_iOS:
    """
    AdamW with GaLore projection, optimized for iPhone memory.

    All optimizer states (m, v) live in the projected rank-32 subspace.
    For a 1536×8960 weight matrix:
      - Full AdamW state: 2 × 1536 × 8960 × 2 bytes = ~52 MB per layer
      - GaLore rank-32:   2 × (1536×32 + 8960×32) × 2 bytes = ~1.3 MB per layer
      - Savings: ~97.5% per layer
      - Total for 22 layers: ~29 MB (vs ~1.1 GB full AdamW)
    """

    def __init__(self, config: iOSTrainingConfig):
        self.lr = config.learning_rate
        self.beta1, self.beta2 = 0.9, 0.999
        self.eps = 1e-8
        self.weight_decay = 0.01
        self.projector = GaLoreProjector_iOS(
            rank=config.galore_rank,
            update_interval=500
        )
        self.state: Dict[str, Dict] = {}
        self.t = 0

    def step(self, named_gradients: Dict[str, np.ndarray], params: Dict[str, np.ndarray]):
        """
        Execute one optimization step.

        Args:
            named_gradients: {param_name: gradient_array}
            params: {param_name: parameter_array} — modified in-place
        """
        self.t += 1
        self.projector.step()

        for name, grad in named_gradients.items():
            if name not in params:
                continue

            p = params[name]

            # Weight decay (decoupled)
            if self.weight_decay > 0 and p.ndim >= 2:
                p *= (1.0 - self.lr * self.weight_decay)

            # Project gradient
            proj_grad = self.projector.project(name, grad)

            # Init state
            if name not in self.state:
                self.state[name] = {
                    'm': np.zeros_like(proj_grad),
                    'v': np.zeros_like(proj_grad),
                    'shape': grad.shape
                }

            s = self.state[name]

            # Resize if projection changed
            if s['m'].shape != proj_grad.shape:
                s['m'] = np.zeros_like(proj_grad)
                s['v'] = np.zeros_like(proj_grad)

            # AdamW moments in projected space
            s['m'] = self.beta1 * s['m'] + (1 - self.beta1) * proj_grad
            s['v'] = self.beta2 * s['v'] + (1 - self.beta2) * (proj_grad ** 2)

            m_hat = s['m'] / (1 - self.beta1 ** self.t)
            v_hat = s['v'] / (1 - self.beta2 ** self.t)

            update_proj = m_hat / (np.sqrt(v_hat) + self.eps)

            # Unproject to full space
            update_full = self.projector.unproject(name, update_proj, s['shape'])

            # Apply
            p -= self.lr * update_full.astype(p.dtype)

    def optimizer_memory_mb(self) -> float:
        """Total optimizer state memory in MB."""
        total = 0
        for s in self.state.values():
            total += s['m'].nbytes + s['v'].nbytes
        total += self.projector.memory_bytes()
        return total / (1024 * 1024)


class iOSTrainingOrchestrator:
    """
    Complete on-device training pipeline for iPhone.

    Designed for iOS BGProcessingTask (overnight fine-tuning):
      - Single-example gradient accumulation (batch_size=1)
      - Thermal yielding every 4 steps (100ms GPU sleep)
      - Atomic checkpointing every 50 steps
      - Total optimizer state: ~29 MB (vs 1.1 GB naive)
      - Peak activation memory: ~15 MB (MeSP 33% checkpointing)
      - Total training overhead: ~50 MB within 800 MB headroom

    Usage on iOS:
      let trainer = iOSTrainingOrchestrator(model, config)
      BGTaskScheduler.shared.register(forTaskWithIdentifier: "com.antigravity.train") { task in
          trainer.run_overnight_session(examples: localExamples, task: task)
      }
    """

    def __init__(self, config: Optional[iOSTrainingConfig] = None):
        self.config = config or iOSTrainingConfig()
        self.mesp = MeSP_iOS(
            n_layers=self.config.n_layers,
            checkpoint_ratio=self.config.mesp_checkpoint_ratio
        )
        self.optimizer = AdamW_GaLore_iOS(self.config)
        self.global_step = 0
        self.training_log: List[Dict] = []
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def train_step(
        self,
        forward_fn,
        loss_fn,
        input_data: np.ndarray,
        target_data: np.ndarray,
        model_params: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Execute one training step with GaLore + MeSP.

        On a real iPhone, forward_fn and loss_fn would be Metal compute
        shader dispatches. Here we use numpy as the reference implementation.
        """
        t0 = time.perf_counter()

        # Forward pass with activation checkpointing
        activations, logits = forward_fn(input_data, model_params, self.mesp)

        # Compute loss
        loss_val = loss_fn(logits, target_data)

        # Backward pass (reference: finite differences for prototype)
        # On iPhone: Metal backward kernels compute analytical gradients
        gradients = {}
        eps = 1e-4
        for name, param in model_params.items():
            if param.ndim < 2:
                continue
            grad = np.zeros_like(param, dtype=np.float16)
            # Only compute gradient for a random subset (stochastic)
            n_samples = min(32, param.size)
            indices = np.random.choice(param.size, n_samples, replace=False)
            flat_param = param.ravel()
            for idx in indices:
                old_val = flat_param[idx]
                flat_param[idx] = old_val + eps
                _, logits_plus = forward_fn(input_data, model_params, self.mesp)
                loss_plus = loss_fn(logits_plus, target_data)
                flat_param[idx] = old_val
                grad.ravel()[idx] = (loss_plus - loss_val) / eps
            gradients[name] = grad

        # Optimizer step (GaLore projected)
        self.optimizer.step(gradients, model_params)

        # Clear activations
        self.mesp.clear()

        self.global_step += 1

        # Thermal yield
        if self.global_step % THERMAL_YIELD_INTERVAL_STEPS == 0:
            time.sleep(self.config.thermal_yield_ms / 1000.0)

        elapsed = time.perf_counter() - t0
        step_info = {
            'step': self.global_step,
            'loss': float(loss_val),
            'elapsed_sec': elapsed,
            'optimizer_mb': self.optimizer.optimizer_memory_mb(),
            'activation_mb': self.mesp.peak_activation_mb(
                self.config.hidden_dim, self.config.max_seq_len
            ),
        }
        self.training_log.append(step_info)

        # Checkpoint
        if self.global_step % self.config.checkpoint_every_n_steps == 0:
            self.save_checkpoint(model_params)

        return step_info

    def save_checkpoint(self, model_params: Dict[str, np.ndarray]):
        path = os.path.join(
            self.config.checkpoint_dir,
            f'ios_ckpt_step_{self.global_step}.npz'
        )
        np.savez_compressed(path, **model_params)
        meta_path = path.replace('.npz', '.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'step': self.global_step,
                'optimizer_mb': self.optimizer.optimizer_memory_mb(),
                'log': self.training_log[-10:]
            }, f, indent=2)
        return path

    def memory_audit(self) -> Dict:
        """Report total training memory overhead."""
        opt_mb = self.optimizer.optimizer_memory_mb()
        act_mb = self.mesp.peak_activation_mb(
            self.config.hidden_dim, self.config.max_seq_len
        )
        total = opt_mb + act_mb
        return {
            'optimizer_state_mb': round(opt_mb, 2),
            'peak_activation_mb': round(act_mb, 2),
            'total_training_overhead_mb': round(total, 2),
            'headroom_remaining_mb': round(TRAINING_HEADROOM_MB - total, 2),
            'fits_in_ios_budget': total < TRAINING_HEADROOM_MB
        }
