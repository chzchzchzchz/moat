"""
Project Antigravity — Real Transformer Model (TinyLlama-1.1B)

Implements a complete LLaMA-architecture transformer forward pass:
  - RMSNorm normalization
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA: 32 Q heads, 4 KV heads)
  - SwiGLU MLP (gate + up → SiLU → down)
  - Real weight loading from Safetensors files

Architecture (TinyLlama-1.1B-Chat-v1.0):
  hidden_dim=2048, intermediate_dim=5632, n_layers=22,
  n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000

Target Hardware: Apple Silicon GPU via PyTorch MPS
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import os
import math


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex frequencies for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_c = torch.view_as_complex(xq_r)
    xk_c = torch.view_as_complex(xk_r)

    freqs = freqs_cis[:xq_c.shape[1]]
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(torch.nn.Module):
    """Grouped-Query Attention (GQA) with RoPE."""

    def __init__(self, hidden_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = torch.nn.Linear(hidden_dim, n_heads * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(n_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_kv_cache = (k, v)

        if self.n_rep > 1:
            k_exp = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                batch, k.shape[1], self.n_heads, self.head_dim)
            v_exp = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1).reshape(
                batch, v.shape[1], self.n_heads, self.head_dim)
        else:
            k_exp, v_exp = k, v

        q_t = q.transpose(1, 2)
        k_t = k_exp.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).type_as(q_t)
        out = torch.matmul(attn, v_t)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out), new_kv_cache


class SwiGLUMLP(torch.nn.Module):
    """SwiGLU MLP: gate_proj, up_proj, down_proj with SiLU activation."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(torch.nn.Module):
    """Single transformer layer: RMSNorm -> Attention -> residual -> RMSNorm -> MLP -> residual."""

    def __init__(self, hidden_dim: int, intermediate_dim: int,
                 n_heads: int, n_kv_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.self_attn = Attention(hidden_dim, n_heads, n_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = SwiGLUMLP(hidden_dim, intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.self_attn(self.input_layernorm(x), freqs_cis, mask, kv_cache=kv_cache)
        h = x + attn_out
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out, new_kv


class TinyLlamaModel(torch.nn.Module):
    """
    Complete TinyLlama-1.1B transformer model.

    Loads real weights from Safetensors and executes genuine multi-layer
    forward passes through 22 transformer blocks.
    """

    HIDDEN_DIM = 2048
    INTERMEDIATE_DIM = 5632
    N_LAYERS = 22
    N_HEADS = 32
    N_KV_HEADS = 4
    HEAD_DIM = 64
    VOCAB_SIZE = 32000
    MAX_SEQ_LEN = 2048
    NORM_EPS = 1e-5
    ROPE_THETA = 10000.0

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device_name = device

        self.embed_tokens = torch.nn.Embedding(self.VOCAB_SIZE, self.HIDDEN_DIM)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                self.HIDDEN_DIM, self.INTERMEDIATE_DIM,
                self.N_HEADS, self.N_KV_HEADS, self.NORM_EPS
            )
            for _ in range(self.N_LAYERS)
        ])

        self.norm = RMSNorm(self.HIDDEN_DIM, eps=self.NORM_EPS)
        self.lm_head = torch.nn.Linear(self.HIDDEN_DIM, self.VOCAB_SIZE, bias=False)

        self.freqs_cis = precompute_freqs_cis(self.HEAD_DIM, self.MAX_SEQ_LEN, self.ROPE_THETA)

    def forward(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Full forward pass with KV caching support."""
        batch, seq_len = token_ids.shape
        h = self.embed_tokens(token_ids)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len].to(h.device)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            h, new_kv = layer(h, freqs_cis, mask, kv_cache=layer_kv)
            new_kv_caches.append(new_kv)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits, new_kv_caches

    def forward_last_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits for the last token position."""
        all_logits, _ = self.forward(token_ids)
        return all_logits[:, -1, :]

    @classmethod
    def from_safetensors(cls, model_path: str, device: str = "cpu") -> "TinyLlamaModel":
        """Load a TinyLlama model from a Safetensors weight file."""
        from safetensors.torch import load_file

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at '{model_path}'. "
                                    f"Run download_model.py first.")

        print(f"Loading TinyLlama-1.1B from {model_path}...")
        state_dict = load_file(model_path)

        model = cls(device=device)

        new_state_dict = {}
        for key, tensor in state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)

            if key.startswith("model."):
                mapped_key = key[len("model."):]
            else:
                mapped_key = key

            new_state_dict[mapped_key] = tensor

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

        model = model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Loaded {total_params:,} parameters to {device}")

        return model

    @torch.no_grad()
    def generate_batch(
        self,
        prompt_ids: torch.Tensor,
        n_channels: int = 8,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 2
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Batched autoregressive generation for N parallel reasoning traces using KV Cache.
        """
        device = next(self.parameters()).device

        curr_ids = prompt_ids.expand(n_channels, -1).to(device)
        prompt_len = curr_ids.shape[1]

        channel_tokens: List[List[int]] = [[] for _ in range(n_channels)]
        channel_logprobs: List[float] = [0.0] * n_channels
        active = [True] * n_channels

        kv_caches = None
        start_pos = 0

        for step in range(max_new_tokens):
            if not any(active):
                break

            all_logits, kv_caches = self.forward(curr_ids, start_pos=start_pos, kv_caches=kv_caches)
            logits = all_logits[:, -1, :]
            start_pos += curr_ids.shape[1]

            if temperature > 0:
                logits = logits / temperature

            probs = F.softmax(logits.float(), dim=-1)

            if temperature > 0 and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_tokens = sorted_indices.gather(-1,
                    torch.multinomial(sorted_probs, num_samples=1))
                next_tokens = next_tokens.squeeze(-1)
            elif temperature > 0:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            log_probs = torch.log(torch.clamp(probs, min=1e-10))
            sampled_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

            for c in range(n_channels):
                if active[c]:
                    tok = int(next_tokens[c].item())
                    channel_tokens[c].append(tok)
                    channel_logprobs[c] += float(sampled_logprobs[c].item())
                    if tok == eos_token_id:
                        active[c] = False

            curr_ids = next_tokens.unsqueeze(-1)

        return channel_tokens, channel_logprobs
"""
Project Antigravity — Real Transformer Model (TinyLlama-1.1B)

Implements a complete LLaMA-architecture transformer forward pass:
  - RMSNorm normalization
  - Rotary Position Embeddings (RoPE)
  - Grouped-Query Attention (GQA: 32 Q heads, 4 KV heads)
  - SwiGLU MLP (gate + up → SiLU → down)
  - Real weight loading from Safetensors files

Architecture (TinyLlama-1.1B-Chat-v1.0):
  hidden_dim=2048, intermediate_dim=5632, n_layers=22,
  n_heads=32, n_kv_heads=4, head_dim=64, vocab_size=32000

Target Hardware: Apple Silicon GPU via PyTorch MPS
"""
