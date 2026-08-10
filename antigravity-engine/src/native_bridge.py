"""
Project Antigravity — Native Metal Engine ctypes Bridge

Bypasses PyTorch MPS entirely by calling the C++ Metal transformer engine
(libantigravity_engine.dylib) through Python's ctypes FFI.

Pipeline:
  Python tokenizer.encode() → ctypes → C++ AntigravityEngineNativeGenerate()
  → 22-layer Metal GPU forward pass → N×max_tokens generated tokens → ctypes → Python

This eliminates the 272× Python↔C boundary crossings that throttled throughput
from 24,000 tok/s (raw Metal) down to 6.4 tok/s (PyTorch MPS wrapper).

Target Hardware: Apple Silicon GPU / iOS (A17 Pro / A18 Pro / M1-M4)
"""

import ctypes
import ctypes.util
import numpy as np
import os
from typing import List, Tuple, Optional


# ============================================================================
# C struct mirroring AntigravityConfig from antigravity_c_api.h
# ============================================================================

class AntigravityConfig(ctypes.Structure):
    _fields_ = [
        ("n_channels", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("hidden_dim", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
        ("use_metal_gpu", ctypes.c_bool),
    ]


class NativeMetalEngine:
    """
    ctypes wrapper around libantigravity_engine.dylib.

    Provides a Python-friendly interface to the full C++ Metal transformer
    engine without any PyTorch dependency in the generation hot path.
    """

    def __init__(
        self,
        dylib_path: str,
        model_path: Optional[str] = None,
        n_channels: int = 8,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        max_seq_len: int = 2048
    ):
        """
        Load the native engine dylib and optionally load model weights.

        Args:
            dylib_path:  Absolute path to libantigravity_engine.dylib.
            model_path:  Path to Safetensors model weights (loaded immediately if provided).
            n_channels:  Number of parallel reasoning channels (N).
            vocab_size:  Model vocabulary size.
            hidden_dim:  Model hidden dimension.
            max_seq_len: Maximum sequence length for KV cache.
        """
        if not os.path.exists(dylib_path):
            raise FileNotFoundError(f"Native engine dylib not found: {dylib_path}")

        self.lib = ctypes.CDLL(dylib_path)
        self.n_channels = n_channels
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self._ctx = None

        # ---- Bind function signatures ----
        self._bind_functions()

        # ---- Create engine context ----
        config = AntigravityConfig(
            n_channels=n_channels,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            use_metal_gpu=True
        )
        self._ctx = self.lib.AntigravityEngineCreate(ctypes.byref(config))
        if not self._ctx:
            raise RuntimeError("AntigravityEngineCreate returned NULL — Metal not available?")

        # ---- Load model weights if path provided ----
        self._model_loaded = False
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)

    def _bind_functions(self):
        """Declare C function argtypes and restypes for type safety."""
        lib = self.lib

        # AntigravityEngineCreate
        lib.AntigravityEngineCreate.argtypes = [ctypes.POINTER(AntigravityConfig)]
        lib.AntigravityEngineCreate.restype = ctypes.c_void_p

        # AntigravityEngineDestroy
        lib.AntigravityEngineDestroy.argtypes = [ctypes.c_void_p]
        lib.AntigravityEngineDestroy.restype = None

        # AntigravityEngineLoadModel
        lib.AntigravityEngineLoadModel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.AntigravityEngineLoadModel.restype = ctypes.c_int32

        # AntigravityEngineNativeGenerate
        lib.AntigravityEngineNativeGenerate.argtypes = [
            ctypes.c_void_p,                                     # ctx
            ctypes.POINTER(ctypes.c_int32),                      # prompt_tokens
            ctypes.c_int32,                                      # prompt_len
            ctypes.c_int32,                                      # max_new_tokens
            ctypes.c_float,                                      # temperature
            ctypes.c_float,                                      # top_p
            ctypes.POINTER(ctypes.c_int32),                      # out_tokens
            ctypes.POINTER(ctypes.c_float),                      # out_logprobs
            ctypes.POINTER(ctypes.c_int32),                      # out_token_counts
            ctypes.POINTER(ctypes.c_double),                     # out_ttft_ms
            ctypes.POINTER(ctypes.c_double),                     # out_total_ms
        ]
        lib.AntigravityEngineNativeGenerate.restype = ctypes.c_int32

        # AntigravityEngineNativeGenerateMultimodal
        if hasattr(lib, 'AntigravityEngineNativeGenerateMultimodal'):
            lib.AntigravityEngineNativeGenerateMultimodal.argtypes = [
                ctypes.c_void_p,                                     # ctx
                ctypes.POINTER(ctypes.c_int32),                      # text_tokens
                ctypes.c_int32,                                      # text_len
                ctypes.POINTER(ctypes.c_float),                      # image_embeddings
                ctypes.c_int32,                                      # n_image_patches
                ctypes.c_int32,                                      # max_new_tokens
                ctypes.c_float,                                      # temperature
                ctypes.c_float,                                      # top_p
                ctypes.POINTER(ctypes.c_int32),                      # out_tokens
                ctypes.POINTER(ctypes.c_float),                      # out_logprobs
                ctypes.POINTER(ctypes.c_int32),                      # out_token_counts
                ctypes.POINTER(ctypes.c_double),                     # out_ttft_ms
                ctypes.POINTER(ctypes.c_double),                     # out_total_ms
            ]
            lib.AntigravityEngineNativeGenerateMultimodal.restype = ctypes.c_int32

        # AntigravityEngineUnloadWeights
        lib.AntigravityEngineUnloadWeights.argtypes = [ctypes.c_void_p]
        lib.AntigravityEngineUnloadWeights.restype = None

        # AntigravityEngineHasWeights
        lib.AntigravityEngineHasWeights.argtypes = [ctypes.c_void_p]
        lib.AntigravityEngineHasWeights.restype = ctypes.c_bool

        # AntigravityEngineGetAllocatedMemoryBytes
        lib.AntigravityEngineGetAllocatedMemoryBytes.argtypes = [ctypes.c_void_p]
        lib.AntigravityEngineGetAllocatedMemoryBytes.restype = ctypes.c_uint64

        # AntigravityEngineSanitizeBuffers
        lib.AntigravityEngineSanitizeBuffers.argtypes = [ctypes.c_void_p]
        lib.AntigravityEngineSanitizeBuffers.restype = None

    def load_weights(self, model_path: str) -> bool:
        """
        Load model weights from Safetensors into Metal GPU buffers.

        Args:
            model_path: Path to .safetensors file.

        Returns:
            True on success.
        """
        ret = self.lib.AntigravityEngineLoadModel(
            self._ctx, model_path.encode('utf-8')
        )
        self._model_loaded = (ret == 0)
        if self._model_loaded:
            print(f"[NativeMetalEngine] Loaded weights from {model_path}")
        else:
            print(f"[NativeMetalEngine] WARNING: Failed to load weights from {model_path} (ret={ret})")
        return self._model_loaded

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[List[List[int]], List[float], float, float]:
        """
        Execute full native 22-layer transformer decode across N channels.

        This is the key function that replaces PyTorch MPS generate_batch().
        A single ctypes call drives the entire autoregressive loop in C++/Metal.

        Args:
            prompt_token_ids: List of integer token IDs from the tokenizer.
            max_new_tokens:   Maximum tokens to generate per channel.
            temperature:      Sampling temperature.
            top_p:            Nucleus sampling threshold.

        Returns:
            Tuple of:
              - channel_tokens:  List[List[int]] — generated token IDs per channel.
              - channel_logprobs: List[float] — cumulative logprob per channel.
              - ttft_ms:         Time to first token (milliseconds).
              - total_ms:        Total wall time (milliseconds).
        """
        if not self._ctx:
            raise RuntimeError("Engine context is NULL — was destroy() called?")

        N = self.n_channels
        prompt_len = len(prompt_token_ids)

        # Allocate C-compatible input/output buffers
        c_prompt = (ctypes.c_int32 * prompt_len)(*prompt_token_ids)
        c_out_tokens = (ctypes.c_int32 * (N * max_new_tokens))()
        c_out_logprobs = (ctypes.c_float * N)()
        c_out_token_counts = (ctypes.c_int32 * N)()
        c_ttft = ctypes.c_double(0.0)
        c_total = ctypes.c_double(0.0)

        # === THE GAS PEDAL ===
        # Single ctypes call → C++ drives the full autoregressive loop on Metal GPU.
        # Zero Python context-switching during generation.
        ret = self.lib.AntigravityEngineNativeGenerate(
            self._ctx,
            c_prompt,
            ctypes.c_int32(prompt_len),
            ctypes.c_int32(max_new_tokens),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            c_out_tokens,
            c_out_logprobs,
            c_out_token_counts,
            ctypes.byref(c_ttft),
            ctypes.byref(c_total)
        )

        if ret != 0:
            error_msg = "weights not loaded" if ret == -1 else f"error code {ret}"
            raise RuntimeError(f"AntigravityEngineNativeGenerate failed: {error_msg}")

        # Unpack flat C buffers into Python lists
        channel_tokens: List[List[int]] = []
        channel_logprobs: List[float] = []

        for c in range(N):
            n_toks = int(c_out_token_counts[c])
            tokens = [int(c_out_tokens[c * max_new_tokens + t]) for t in range(n_toks)]
            channel_tokens.append(tokens)
            channel_logprobs.append(float(c_out_logprobs[c]))

        return channel_tokens, channel_logprobs, float(c_ttft.value), float(c_total.value)

    def generate_multimodal(
        self,
        text_token_ids: List[int],
        image_embeddings: List[float],
        n_image_patches: int,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[List[List[int]], List[float], float, float]:
        """
        Execute full native multimodal transformer decode across N channels.

        Args:
            text_token_ids:   List of integer token IDs for text.
            image_embeddings: Flattened list of float image patch embeddings [n_patches * hidden_dim].
            n_image_patches:  Number of image patches.
            max_new_tokens:   Maximum tokens to generate per channel.
            temperature:      Sampling temperature.
            top_p:            Nucleus sampling threshold.

        Returns:
            Tuple of (channel_tokens, channel_logprobs, ttft_ms, total_ms).
        """
        if not self._ctx:
            raise RuntimeError("Engine context is NULL — was destroy() called?")

        N = self.n_channels
        text_len = len(text_token_ids)

        c_text = (ctypes.c_int32 * text_len)(*text_token_ids) if text_len > 0 else None
        c_img = (ctypes.c_float * len(image_embeddings))(*image_embeddings) if image_embeddings else None
        c_out_tokens = (ctypes.c_int32 * (N * max_new_tokens))()
        c_out_logprobs = (ctypes.c_float * N)()
        c_out_token_counts = (ctypes.c_int32 * N)()
        c_ttft = ctypes.c_double(0.0)
        c_total = ctypes.c_double(0.0)

        ret = self.lib.AntigravityEngineNativeGenerateMultimodal(
            self._ctx,
            c_text,
            ctypes.c_int32(text_len),
            c_img,
            ctypes.c_int32(n_image_patches),
            ctypes.c_int32(max_new_tokens),
            ctypes.c_float(temperature),
            ctypes.c_float(top_p),
            c_out_tokens,
            c_out_logprobs,
            c_out_token_counts,
            ctypes.byref(c_ttft),
            ctypes.byref(c_total)
        )

        if ret != 0:
            error_msg = "weights not loaded" if ret == -1 else f"error code {ret}"
            raise RuntimeError(f"AntigravityEngineNativeGenerateMultimodal failed: {error_msg}")

        channel_tokens: List[List[int]] = []
        channel_logprobs: List[float] = []

        for c in range(N):
            n_toks = int(c_out_token_counts[c])
            tokens = [int(c_out_tokens[c * max_new_tokens + t]) for t in range(n_toks)]
            channel_tokens.append(tokens)
            channel_logprobs.append(float(c_out_logprobs[c]))

        return channel_tokens, channel_logprobs, float(c_ttft.value), float(c_total.value)

    def unload_weights(self):
        """Flush all model weights from Metal GPU buffers. Used for VRAM swapping."""
        if self._ctx:
            self.lib.AntigravityEngineUnloadWeights(self._ctx)
            self._model_loaded = False
            print("[NativeMetalEngine] Weights unloaded — VRAM freed")

    def has_weights(self) -> bool:
        """Query whether model weights are currently loaded."""
        if not self._ctx:
            return False
        return bool(self.lib.AntigravityEngineHasWeights(self._ctx))

    def get_allocated_bytes(self) -> int:
        """Get total VRAM/RAM allocated by the native engine."""
        if not self._ctx:
            return 0
        return int(self.lib.AntigravityEngineGetAllocatedMemoryBytes(self._ctx))

    def destroy(self):
        """Release all Metal resources and destroy the engine context."""
        if self._ctx:
            self.lib.AntigravityEngineDestroy(self._ctx)
            self._ctx = None
            self._model_loaded = False
            print("[NativeMetalEngine] Destroyed")

    def __del__(self):
        self.destroy()

    @property
    def is_ready(self) -> bool:
        """True if engine is created and weights are loaded."""
        return self._ctx is not None and self._model_loaded
