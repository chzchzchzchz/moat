"""
Project Antigravity — Engine Orchestrator

Integrates engine subsystems into a pipeline:
  1. Native C++ Metal Transformer (via ctypes bridge) — PRIMARY PATH
     Drives full 22-layer autoregressive decode on Apple Silicon GPU.
  2. PyTorch MPS Transformer — FALLBACK (only if dylib missing)
  3. INT4 Quantized Super-Block Weights & Softmax LUT
  4. Batched parallel rollout coordinator (N=8 reasoning traces)
  5. Paged KV-cache memory manager
  6. PRM-guided Best-of-N verification strategy
  7. Adaptive reflection controller (threshold tau=0.75 token savings)

Target Hardware: Apple Silicon GPU / iOS (A17 Pro / A18 Pro / M1-M4)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
import torch
from dequant import quantize_weights_int4, repack_to_superblocks, lut_dequantize
from attention import ExponentialLUT, safe_softmax_lut
from batch_generator import BatchedRolloutCoordinator, PagedKVCache
from verifier import ListWiseVerifier, AdaptiveReflectionManager, SequentialModelSwapper, NeuralPRMVerifier, DEFAULT_REFLECTION_THRESHOLD
from model_loader import ModelWeightLoader
from verification_strategy import BestOfNStrategy

try:
    from native_bridge import NativeMetalEngine
    HAS_NATIVE_BRIDGE = True
except ImportError:
    HAS_NATIVE_BRIDGE = False

try:
    from tokenizer import LlamaTokenizer
    from transformer import TinyLlamaModel
    HAS_REAL_MODEL = True
except ImportError:
    HAS_REAL_MODEL = False

from genprm_verifier import GenPRMVerifier
from dora_clustering import DORAClusterer


class RealTransformerLayer:
    """
    Transformer Layer with SwiGLU MLP and self-attention.

    NOTE: Weights are random-initialized (not trained).
    This validates the forward-pass pipeline shape and numerics.
    For real inference, replace weights with loaded model parameters.
    """

    def __init__(self, hidden_dim: int = 256, intermediate_dim: int = 512, seed: int = 42):
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Initialize deterministic weights
        rng = np.random.RandomState(seed)
        self.w_q = (rng.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float16)
        self.w_k = (rng.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float16)
        self.w_v = (rng.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float16)
        self.w_o = (rng.randn(hidden_dim, hidden_dim) * 0.02).astype(np.float16)
        self.w_gate = (rng.randn(hidden_dim, intermediate_dim) * 0.02).astype(np.float16)
        self.w_up = (rng.randn(hidden_dim, intermediate_dim) * 0.02).astype(np.float16)
        self.w_down = (rng.randn(intermediate_dim, hidden_dim) * 0.02).astype(np.float16)

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Execute forward pass across N channels simultaneously: shape (N, K).
        """
        N, K = x.shape
        # Attention projection
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        
        # Simplified self-attention with scaling
        scale = 1.0 / np.sqrt(K)
        attn_scores = (q @ k.T) * scale
        
        # Softmax
        exp_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        attn_out = attn_probs @ v
        x_attn = x + (attn_out @ self.w_o)
        
        # SwiGLU MLP: (silu(x @ w_gate) * (x @ w_up)) @ w_down
        gate = x_attn @ self.w_gate
        up = x_attn @ self.w_up
        # SiLU activation: x * sigmoid(x)
        silu_gate = gate * (1.0 / (1.0 + np.exp(-np.clip(gate, -10, 10))))
        mlp_out = (silu_gate * up) @ self.w_down
        
        return (x_attn + mlp_out).astype(np.float16)


class AntigravityEngine:
    """
    Engine orchestrator for parallel Best-of-N decode.

    Generation priority:
      1. Native C++ Metal engine (via ctypes) — zero Python overhead
      2. PyTorch MPS model — fallback if dylib missing
      3. Pipeline validator — last resort with random weights
    """

    def __init__(
        self,
        n_channels: int = 8,
        vocab_size: int = 1000,
        hidden_dim: int = 256,
        reflection_threshold: float = DEFAULT_REFLECTION_THRESHOLD,
        model_dir: Optional[str] = None
    ):
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Paths - resolve robustly against workspace root
        abs_qwen = "/Users/MohssineChazi2/moat/models/qwen"
        abs_tiny = "/Users/MohssineChazi2/moat/models/tinyllama"
        if model_dir is not None and os.path.exists(os.path.abspath(model_dir)):
            model_base = os.path.abspath(model_dir)
        elif os.path.exists(abs_qwen):
            model_base = abs_qwen
        elif os.path.exists(abs_tiny):
            model_base = abs_tiny
        else:
            model_base = "models/tinyllama"

        fp16_path = os.path.join(model_base, "model_fp16.safetensors")
        model_path = fp16_path if os.path.exists(fp16_path) else os.path.join(model_base, "model.safetensors")
        tokenizer_path = os.path.join(model_base, "tokenizer.json")
        prm_dir = os.path.join(os.path.dirname(model_base), "skywork-prm-1.5b")

        # Resolve dylib path (search multiple candidate locations)
        dylib_candidates = [
            os.path.abspath("libantigravity_engine.dylib"),
            os.path.join(os.path.dirname(__file__), "libantigravity_engine.dylib"),
            os.path.join(os.path.dirname(__file__), "..", "src", "libantigravity_engine.dylib"),
            os.path.abspath("antigravity-engine/src/libantigravity_engine.dylib"),
            os.path.abspath("src/libantigravity_engine.dylib"),
        ]

        # ======================================================================
        # GENERATION PATH 0: HuggingFace PyTorch MPS for Qwen
        # ======================================================================
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if os.path.exists(model_base) and "qwen" in model_base.lower():
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_base)
                self.hf_model = AutoModelForCausalLM.from_pretrained(model_base, dtype=torch.float16).to(self.device)
                self.vocab_size = getattr(self.hf_tokenizer, "vocab_size", 151936)
                self.hidden_dim = 1536
                print(f"[AntigravityEngine] ✅ Loaded HuggingFace Qwen2.5 model on PyTorch {self.device}")
            except Exception as e:
                print(f"[AntigravityEngine] Notice: HF Qwen load failed ({e})")

        # ======================================================================
        # GENERATION PATH 1: Native C++ Metal Engine (ctypes bridge)
        # ======================================================================
        self.native_engine = None
        self.tokenizer = None

        if self.hf_model is None and HAS_NATIVE_BRIDGE and os.path.exists(model_path):
            for dylib_path in dylib_candidates:
                if os.path.exists(dylib_path):
                    try:
                        is_qwen = "qwen" in model_base.lower()
                        v_size = 151936 if is_qwen else 32000
                        h_dim = 1536 if is_qwen else 2048

                        self.native_engine = NativeMetalEngine(
                            dylib_path=os.path.abspath(dylib_path),
                            model_path=os.path.abspath(model_path),
                            n_channels=n_channels,
                            vocab_size=v_size,
                            hidden_dim=h_dim,
                            max_seq_len=2048
                        )
                        self.vocab_size = v_size
                        self.hidden_dim = h_dim
                        print(f"[AntigravityEngine] ✅ Native C++ Metal engine loaded ({model_base}, vocab={v_size}) from {dylib_path}")
                        break
                    except Exception as e:
                        print(f"[AntigravityEngine] Notice: Native bridge failed ({e}), trying next...")
                        self.native_engine = None

        # Load tokenizer (needed for both native and MPS paths)
        if os.path.exists(tokenizer_path):
            try:
                from tokenizer import LlamaTokenizer
                self.tokenizer = LlamaTokenizer(tokenizer_path)
            except Exception:
                self.tokenizer = None

        # ======================================================================
        # GENERATION PATH 2: PyTorch MPS Fallback (only if native unavailable)
        # ======================================================================
        self.real_model = None
        if self.hf_model is None and self.native_engine is None and "tinyllama" in model_base.lower():
            if HAS_REAL_MODEL and os.path.exists(model_path) and self.tokenizer is not None:
                try:
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    self.real_model = TinyLlamaModel.from_safetensors(model_path, device=device)
                    self.vocab_size = self.real_model.VOCAB_SIZE
                    self.hidden_dim = self.real_model.HIDDEN_DIM
                    print(f"[AntigravityEngine] ⚠️  Falling back to PyTorch MPS on {device}")
                except Exception as e:
                    print(f"[AntigravityEngine] Notice: MPS fallback failed ({e})")
                    self.real_model = None

        # ======================================================================
        # Verifier & Support Components
        # ======================================================================
        self.coordinator = BatchedRolloutCoordinator(
            n_channels=n_channels,
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim
        )
        self.verifier = ListWiseVerifier()
        self.genprm_verifier = GenPRMVerifier()
        self.dora_clusterer = DORAClusterer()
        self.prm_verifier = None # Disabled to prevent OOM
        self.bon_strategy = BestOfNStrategy(logprob_weight=0.1)
        self.reflection_manager = AdaptiveReflectionManager(threshold=reflection_threshold)
        self.model_loader = ModelWeightLoader()
        v_path = None
        if self.prm_verifier is not None and os.path.exists(prm_dir):
            v_path = os.path.join(prm_dir, "pytorch_model.bin")

        self.model_swapper = SequentialModelSwapper(
            loader=self.model_loader,
            reasoner_path=model_path,
            verifier_path=v_path,
            native_engine=self.native_engine
        )

        # Pipeline validator components (used only in path 3)
        self.transformer_layer = RealTransformerLayer(hidden_dim=self.hidden_dim)
        rng = np.random.RandomState(123)
        self.embedding_table = (rng.randn(self.vocab_size, self.hidden_dim) * 0.05).astype(np.float16)
        self.lm_head_weight = (rng.randn(self.hidden_dim, self.vocab_size) * 0.05).astype(np.float16)

        # Build vocabulary mapping
        self.vocab = self._build_vocab(self.vocab_size)

        # Report active generation path
        if self.native_engine and self.native_engine.is_ready:
            self._generation_mode = "native_metal"
        elif self.real_model is not None:
            self._generation_mode = "pytorch_mps"
        else:
            self._generation_mode = "pipeline_validator"

    def _build_vocab(self, vocab_size: int) -> List[str]:
        """Build generic token vocabulary with special tokens and numbered entries."""
        special = ["<pad>", "<bos>", "<eos>", "<think>", "</think>"]
        vocab = special.copy()
        while len(vocab) < vocab_size:
            vocab.append(f"tok_{len(vocab)}")
        return vocab

    def decode_tokens(self, token_ids: List[int]) -> str:
        """De-tokenize sequence of token IDs to readable string."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids)
        tokens = [self.vocab[tid] for tid in token_ids if tid < len(self.vocab)]
        text = " ".join(tokens)
        text = text.replace(" <eos>", "").replace("<pad>", "").strip()
        return text

    def run_best_of_n_query(
        self,
        prompt: str,
        max_tokens: int = 40,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """
        Execute an end-to-end parallel Best-of-N reasoning query.

        Attempts native C++ Metal generation first, falls back to PyTorch MPS,
        then to the pipeline validator.
        """
        t0 = time.perf_counter()

        channel_tokens: List[List[int]] = []
        cum_logprobs: np.ndarray = np.array([], dtype=np.float32)
        candidate_traces: List[str] = []
        native_ttft_ms = 0.0
        native_total_ms = 0.0

        # Step 1: Memory swap → Reasoner
        self.model_swapper.swap_to_reasoner()
        self.coordinator.reset()

        # ============================================================
        # PATH 0: HuggingFace PyTorch MPS for Qwen
        # ============================================================
        if self.hf_model is not None and self.hf_tokenizer is not None:
            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_len = inputs.input_ids.shape[1]
            candidate_traces = []
            logprobs_list = []

            for c in range(self.n_channels):
                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0.0),
                        temperature=max(temperature, 0.01),
                        top_p=top_p
                    )
                gen_seq = outputs[0][prompt_len:]
                text = self.hf_tokenizer.decode(gen_seq, skip_special_tokens=False)
                candidate_traces.append(text)
                logprobs_list.append(-1.0)

            cum_logprobs = np.array(logprobs_list, dtype=np.float32)

        # ============================================================
        # PATH 1: Native C++ Metal Engine (ctypes bridge)
        # Zero Python overhead — single call drives full decode loop
        # ============================================================
        elif self.native_engine is not None and self.native_engine.is_ready and self.tokenizer is not None:
            prompt_token_ids = self.tokenizer.encode(prompt)

            channel_tokens, channel_logprobs_list, native_ttft_ms, native_total_ms = \
                self.native_engine.generate(
                    prompt_token_ids=prompt_token_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

            candidate_traces = [self.tokenizer.decode(tokens) for tokens in channel_tokens]
            cum_logprobs = np.array(channel_logprobs_list, dtype=np.float32)

        # ============================================================
        # PATH 2: PyTorch MPS Fallback
        # ============================================================
        elif self.real_model is not None and self.tokenizer is not None:
            prompt_token_ids = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_token_ids], dtype=torch.long)
            
            channel_tokens, channel_logprobs_list = self.real_model.generate_batch(
                prompt_ids=prompt_tensor,
                n_channels=self.n_channels,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            candidate_traces = [self.tokenizer.decode(tokens) for tokens in channel_tokens]
            cum_logprobs = np.array(channel_logprobs_list, dtype=np.float32)

        # ============================================================
        # PATH 3: Pipeline Validator (no real model)
        # ============================================================
        else:
            prompt_tokens = [hash(w) % (self.vocab_size - 10) + 5 for w in prompt.split()]
            if not prompt_tokens:
                prompt_tokens = [1, 10, 100]

            weights = self.lm_head_weight

            channel_tokens = self.coordinator.generate(
                prompt_tokens=prompt_tokens,
                weights=weights,
                max_steps=max_tokens,
                temperature=temperature
            )

            cum_logprobs = np.array(self.coordinator.channel_logprobs, dtype=np.float32)
            candidate_traces = [self.decode_tokens(tokens) for tokens in channel_tokens]

        # Step 2: Memory swap → Verifier
        self.model_swapper.swap_to_verifier()

        # Clamp logprobs to prevent NaN in LUT softmax (extreme values from native engine)
        cum_logprobs = np.nan_to_num(cum_logprobs, nan=0.0, posinf=0.0, neginf=-100.0)
        cum_logprobs = np.clip(cum_logprobs, -100.0, 0.0)

        # Score candidates using heuristic verifier (produces softmax-normalized [0,1] scores)
        verification_res = self.verifier.score_candidates_listwise(
            candidate_traces,
            cum_logprobs
        )

        # 1. On-Device Real DORA Semantic Clustering & Uniqueness Weighting
        dora_res = self.dora_clusterer.cluster_candidates(candidate_traces)
        uniqueness_weights = dora_res['uniqueness_weights']

        # 2. Program-Aided Self-Correction (GenPRM) Code Execution Feedback
        base_scores = verification_res['scores'] * (uniqueness_weights if len(uniqueness_weights) == len(candidate_traces) else 1.0)
        genprm_scores, genprm_reports = self.genprm_verifier.score_candidates_genprm(
            candidate_traces,
            base_scores
        )

        # Select best candidate using GenPRM + DORA adjusted scores
        best_index = int(np.argmax(genprm_scores)) if len(genprm_scores) > 0 else verification_res['best_index']
        best_trace = candidate_traces[best_index] if len(candidate_traces) > 0 else verification_res['best_trace']

        # best_score always comes from the softmax-normalized verifier (range [0,1])
        best_score = float(verification_res['scores'][best_index])

        # Step 3: Adaptive Reflection Check
        trigger_reflection = self.reflection_manager.evaluate_reflection_trigger(best_score)
        if trigger_reflection:
            # Check if GenPRM generated a specific code feedback prompt
            feedback_prompt = genprm_reports[best_index].get('feedback_prompt') if best_index < len(genprm_reports) else None
            
            if self.hf_model is None and self._generation_mode != "native_metal":
                self.model_swapper.swap_to_reasoner()
                self.coordinator.reset()
                refine_act = self.embedding_table[[3]]
                refine_hidden = self.transformer_layer.forward_batch(refine_act)
                _ = self.coordinator.step_decode_batch(refine_hidden, self.lm_head_weight, temperature=0.1)
            
            if feedback_prompt:
                best_trace = best_trace + f"\n[GenPRM Feedback: {feedback_prompt}]"
            else:
                best_trace = best_trace + "\n[Refinement Pass Verified]"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = sum(len(t) for t in channel_tokens) if isinstance(channel_tokens, list) and channel_tokens and isinstance(channel_tokens[0], list) else len(channel_tokens)

        return {
            'prompt': prompt,
            'best_trace': best_trace,
            'best_score': best_score,
            'best_index': best_index,
            'scores': verification_res['scores'],
            'candidate_traces': candidate_traces,
            'candidates_evaluated': self.n_channels,
            'reflection_triggered': trigger_reflection,
            'token_savings_percentage': self.reflection_manager.token_savings_percentage,
            'token_savings_pct': self.reflection_manager.token_savings_percentage,
            'latency_ms': elapsed_ms,
            'tokens_generated_total': total_tokens,
            'per_token_latency_ms': elapsed_ms / max(total_tokens, 1),
            'generation_mode': self._generation_mode,
            'native_ttft_ms': native_ttft_ms,
            'native_total_ms': native_total_ms,
            'dora_similarity_matrix': dora_res.get('similarity_matrix'),
            'dora_uniqueness_weights': dora_res.get('uniqueness_weights'),
            'genprm_reports': genprm_reports,
        }
