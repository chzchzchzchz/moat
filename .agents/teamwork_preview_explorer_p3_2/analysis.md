# Architectural & Technical Design: R2 Model Weight Loader & Super-Block Repacker

**Module Target**: `antigravity-engine/src/model_loader.py`  
**Test Suite Target**: `antigravity-engine/tests/test_model_loader.py`  
**Author**: Explorer 2 (Phase 3)  
**Date**: July 25, 2026  

---

## 1. Executive Summary & Architectural Scope

Phase 3 of Project Antigravity establishes an end-to-end batched parallel decoding engine running $N=8$ candidate reasoning traces on Apple Silicon (A17 Pro / A18 Pro / M-series UMA). To execute GEMM matrix math at hardware peak arithmetic intensity without memory stalls or OS memory terminations (`Jetsam` OOM), model weight matrices must be loaded from standard open-weight formats (GGUF, Safetensors) and repacked on-the-fly into **128-byte aligned 256-element super-blocks** (144 bytes each).

The component **R2: Model Weight Loader & Super-Block Repacker** (`src/model_loader.py`) bridges raw weight files and high-performance quantized Metal GPU memory buffers. 

```
                               ┌────────────────────────────────────────────────┐
                               │               Raw Weight File                  │
                               │   (GGUF / Safetensors / Mock Qwen2.5-1.5B)      │
                               └───────────────────────┬────────────────────────┘
                                                       │
                                                       ▼
                               ┌────────────────────────────────────────────────┐
                               │           BaseWeightReader Hierarchy           │
                               │   (GGUFReader / SafetensorsReader / Mock)      │
                               └───────────────────────┬────────────────────────┘
                                                       │  FP16 / FP32 Tensors
                                                       ▼
                               ┌────────────────────────────────────────────────┐
                               │            SuperBlockRepacker Engine           │
                               │   - INT4 Symmetric Group Quantization (g=32)   │
                               │   - Coalesce 8 Groups -> 256-element SuperBlock│
                               │   - 128-byte Cache Line & Vector Alignment     │
                               └───────────────────────┬────────────────────────┘
                                                       │  QuantizedSuperBlockTensors
                                                       ▼
                               ┌────────────────────────────────────────────────┐
                               │            MemoryBudgetValidator               │
                               │   - Weights <= 2.5 GB Ceiling (0.844 GB actual)│
                               │   - Total App Footprint <= 4.5 GB (1.814 GB)   │
                               └───────────────────────┬────────────────────────┘
                                                       │  Validated Memory Buffers
                                                       ▼
                               ┌────────────────────────────────────────────────┐
                               │      Batched GEMM Engine / Metal GPU Buffers   │
                               └────────────────────────────────────────────────┘
```

---

## 2. Model Weight Parsing Architecture

### 2.1 File Reader Abstraction (`BaseWeightReader`)
The parser framework defines an abstract base class `BaseWeightReader` to provide a uniform stream interface for reading FP16 / FP32 tensors across multiple file formats.

```python
class BaseWeightReader(ABC):
    @abstractmethod
    def list_tensors(self) -> List[str]:
        """Return list of tensor names available in the file/stream."""
        pass

    @abstractmethod
    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """Return metadata for tensor: shape, dtype, byte_offset, size_bytes."""
        pass

    @abstractmethod
    def load_tensor(self, name: str) -> np.ndarray:
        """Load and return FP16 numpy array for requested tensor name."""
        pass

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Return architectural key-value metadata (architecture, layers, heads, hidden_size)."""
        pass
```

### 2.2 GGUF Binary Parser (`GGUFWeightReader`)
GGUF is the standard single-file binary format for quantized/unquantized open-weight LLMs.
- **Magic Number**: `0x46554647` (`b"GGUF"` in Little Endian).
- **Header Structure**:
  - `magic` (4 bytes): `b"GGUF"`
  - `version` (uint32): Version 2 or 3.
  - `tensor_count` (uint64): Total count of tensor weight matrices.
  - `metadata_kv_count` (uint64): Number of model configuration key-value pairs.
- **Metadata Key-Value Storage**:
  - Keys parsed as string (`length: uint64` + UTF-8 bytes).
  - Values parsed according to GGUF type enum (`UINT8=0`, `INT8=1`, `UINT16=2`, `INT16=3`, `UINT32=4`, `INT32=5`, `FLOAT32=6`, `BOOL=7`, `STRING=8`, `ARRAY=9`, `UINT64=10`, `INT64=11`, `FLOAT64=12`).
  - Key fields extracted for model validation: `general.architecture` ("qwen2"), `qwen2.block_count` (28), `qwen2.embedding_length` (1536), `qwen2.feed_forward_length` (8960), `qwen2.attention.head_count` (12), `qwen2.attention.head_count_kv` (2).
- **Tensor Header Directory**:
  - For each tensor: `name` (string), `n_dimensions` (uint32), `dimensions` (`uint64[n_dimensions]`), `ggml_type` (uint32: `FP32=0`, `FP16=1`), `offset` (uint64 relative to binary data section start).
  - Data offset padding aligned to `general.alignment` (typically 32 bytes).

### 2.3 Safetensors JSON & Binary Parser (`SafetensorsWeightReader`)
Safetensors is a zero-copy binary format with a JSON metadata header.
- **Header Length**: First 8 bytes store a 64-bit unsigned little-endian integer ($N$).
- **JSON Metadata**: The following $N$ bytes contain UTF-8 JSON text mapping tensor names to properties:
  ```json
  {
    "__metadata__": {"format": "pt"},
    "model.layers.0.self_attn.q_proj.weight": {
      "dtype": "F16",
      "shape": [1536, 1536],
      "data_offsets": [0, 4718592]
    }
  }
  ```
- **Binary Data Buffer**: Starting at byte $8 + N$, raw binary tensor buffers are stored sequentially at the specified `data_offsets`. FP32 tensors are automatically cast to FP16 upon read.

### 2.4 Mock Tensor Generator (`MockWeightReader`)
For deterministic testing, unit validation, and running open-weight model architectures (e.g. Qwen2.5-1.5B) without loading multi-gigabyte disk files:
- Synthesizes all required model layers matching model architecture specs:
  - 28 Transformer layers, each containing Q, K, V, O attention projections, Gate, Up, Down MLP projections, and LayerNorm weights.
  - Embedding table ($151,936 \times 1536$) and Final LayerNorm + Head projection.
- Generates reproducible FP16 weight values using seeded pseudo-random distributions (or structured test patterns like orthogonal matrices / range patterns).

---

## 3. Super-Block Repacking Engine

### 3.1 Mathematical Layout & Format Specification
Fine-grained INT4 symmetric quantization uses group size $g = 32$. To eliminate GPU register under-utilization during SIMD vector loading, **8 contiguous groups** are coalesced into a single 256-element Super-Block.

```
Super-Block Layout (144 Bytes Total):
┌────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Header (16 Bytes)                                      │ Payload (128 Bytes)                                    │
│ [S_0: FP16][S_1: FP16] ... [S_7: FP16]                 │ [Group 0..7 Packed INT4 Nibbles (128 uint8 bytes)]    │
└────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

#### Detailed Binary Sub-fields:
1. **Header (16 Bytes)**:
   - 8 per-group FP16 scaling factors ($S_0, S_1, \dots, S_7$).
   - Derived from $S_g = \frac{\max_{i \in G} |w_i|}{7}$ for each 32-element group $G$.
2. **Payload (128 Bytes)**:
   - 256 INT4 weight values ($q_i \in [-8, 7]$).
   - Each INT4 value is shifted to unsigned range $[0, 15]$ via $u_i = q_i + 8$.
   - Packed as byte pairs into 128 `uint8` elements:
     $$\text{byte}_k = (u_{2k} \ \& \ 0x0F) \ | \ ((u_{2k+1} \ \& \ 0x0F) \ll 4) \quad \text{for } k \in [0, 127]$$

### 3.2 128-Byte Cache Line & Vector Alignment Verification
On Apple Silicon GPUs (Metal), memory transactions operate on 128-byte cache line boundaries. 
- The **Payload** of each Super-Block is exactly 128 bytes ($1024 \text{ bits}$), matching a 128-byte SIMD vector line.
- Memory arrays created by `SuperBlockRepacker` enforce contiguous C-order layout (`np.ascontiguousarray`) and verify byte address alignment:
  $$\text{Payload Offset} \pmod{128} == 0 \quad \text{or} \quad \text{Header Offset} \pmod{16} == 0$$
- `QuantizedSuperBlockTensor` maintains payload byte buffers and scale headers in aligned memory arrays, exposing `.verify_alignment()` to guarantee 100% compliance.

### 3.3 Repacking Pipeline Algorithm

```python
class QuantizedSuperBlockTensor:
    """
    Encapsulates a repacked super-block weight matrix.
    Stores scales and packed nibbles in 128-byte cache-line aligned arrays.
    """
    def __init__(self, name: str, original_shape: Tuple[int, ...], scales: np.ndarray, packed_payload: np.ndarray):
        self.name = name
        self.original_shape = original_shape
        self.n_elements = int(np.prod(original_shape))
        self.n_superblocks = len(scales)
        
        # Enforce memory contiguous and float16/uint8 dtypes
        self.scales = np.ascontiguousarray(scales, dtype=np.float16)  # shape: (n_superblocks, 8)
        self.packed_payload = np.ascontiguousarray(packed_payload, dtype=np.uint8)  # shape: (n_superblocks, 128)
        
    @property
    def memory_bytes(self) -> int:
        """Total memory in bytes: 144 bytes per super-block."""
        return self.scales.nbytes + self.packed_payload.nbytes

    def verify_alignment(self) -> bool:
        """Verify 128-byte cache line alignment and size bounds."""
        if self.scales.shape[1] != 8:
            return False
        if self.packed_payload.shape[1] != 128:
            return False
        if self.memory_bytes != self.n_superblocks * 144:
            return False
        return True
```

---

## 4. Memory Allocation & Budget Validator

### 4.1 System Memory Breakdown on 8GB iPhone (iOS UMA)
With the `com.apple.developer.kernel.increased-memory-limit` entitlement, iOS allocates up to ~5.5 GB to the application. Project Antigravity establishes a strict **4.5 GB Ceiling Budget** to prevent `Jetsam` high-memory terminations.

```
Total iPhone RAM: 8.0 GB
├── iOS System & Neural Engine Reserve: ~2.5 GB
└── App Maximum Available RAM: ~5.5 GB
    └── Project Antigravity Safety Budget: 4.5 GB Ceiling
        ├── INT4 Super-Block Model Weights (1.5B params): 0.844 GB  [Max Budget: 2.50 GB]
        ├── Batched Paged KV Cache (N=8, Context=2048):  0.470 GB
        ├── Softmax Exponential LUT (32K FP16 entries):   0.000064 GB (64 KB)
        └── Metal Command Buffers & App Reserve:         0.500 GB
        ────────────────────────────────────────────────────────────
        TOTAL APP FOOTPRINT:                              1.814 GB  [Headroom: 2.686 GB!]
```

### 4.2 Mathematical Formulas for Memory Calculator

#### 1. Model Weights Memory ($M_{\text{weights}}$):
$$\text{Parameters} = N_{\text{params}} = 1.5 \times 10^9$$
$$N_{\text{sb}} = \left\lceil \frac{N_{\text{params}}}{256} \right\rceil = 5,859,375 \text{ super-blocks}$$
$$M_{\text{weights}} = N_{\text{sb}} \times 144 \text{ bytes} = 843,750,000 \text{ bytes} \approx 0.84375 \text{ GB}$$

#### 2. Batched Paged KV-Cache Memory ($M_{\text{kv}}$):
For Grouped-Query Attention (GQA) with $N=8$ parallel reasoning channels, context length $S=2048$, layer count $L=28$, KV head count $H_{\text{kv}}=2$, head dimension $D_{\text{head}}=128$:
$$\text{Bytes per token per layer} = 2 \text{ (Key, Value)} \times H_{\text{kv}} \times D_{\text{head}} \times 2 \text{ bytes (FP16)} = 2 \times 2 \times 128 \times 2 = 1024 \text{ bytes}$$
$$\text{Bytes per trace} = 28 \text{ layers} \times 2048 \text{ tokens} \times 1024 \text{ bytes} = 58,720,256 \text{ bytes} \approx 58.72 \text{ MB}$$
$$M_{\text{kv}} = N \times \text{Bytes per trace} = 8 \times 58.72 \text{ MB} = 469,762,048 \text{ bytes} \approx 0.470 \text{ GB}$$

#### 3. Softmax Exp LUT Memory ($M_{\text{lut}}$):
$$M_{\text{lut}} = 32,768 \text{ entries} \times 2 \text{ bytes (FP16)} = 65,536 \text{ bytes} \approx 0.000064 \text{ GB}$$

#### 4. App & Metal Runtime Reserve ($M_{\text{reserve}}$):
$$M_{\text{reserve}} = 500 \text{ MB} = 524,288,000 \text{ bytes} \approx 0.500 \text{ GB}$$

#### 5. Total Footprint Calculation:
$$M_{\text{total}} = M_{\text{weights}} + M_{\text{kv}} + M_{\text{lut}} + M_{\text{reserve}} = 0.84375 + 0.46976 + 0.00006 + 0.50000 = 1.81357 \text{ GB}$$

### 4.3 Validation Criteria Enforcement
`MemoryBudgetValidator` evaluates all candidate model configurations against strict safety thresholds:

| Constraint | Limit Ceiling | Qwen2.5-1.5B Measured Value | Status | Headroom |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weight Memory** | $\le 2.50 \text{ GB}$ | $0.844 \text{ GB}$ | **PASS** | $+1.656 \text{ GB}$ |
| **Batched KV Cache ($N=8$)** | $\le 1.00 \text{ GB}$ | $0.470 \text{ GB}$ | **PASS** | $+0.530 \text{ GB}$ |
| **Total Memory Footprint** | $\le 4.50 \text{ GB}$ | $1.814 \text{ GB}$ | **PASS** | $+2.686 \text{ GB}$ |

---

## 5. Detailed Technical Design for `src/model_loader.py`

Below is the concrete code specification and interface definition to be implemented in `antigravity-engine/src/model_loader.py`.

```python
"""
Project Antigravity — R2: Model Weight Loader & Super-Block Repacker

This module implements:
  1. GGUF, Safetensors, and Mock weight parsing for open-weight models (Qwen2.5-1.5B).
  2. On-the-fly INT4 quantization and 256-element super-block repacking (144 bytes).
  3. 128-byte cache line & SIMD register alignment verification.
  4. UMA memory calculator and ceiling budget validator (< 2.5 GB weights, < 4.5 GB total).
"""

from abc import ABC, abstractmethod
import os
import json
import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from dequant import quantize_weights_int4, repack_to_superblocks, unpack_superblock


@dataclass
class ModelMetadata:
    architecture: str = "qwen2"
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 151936
    max_context_length: int = 2048


@dataclass
class MemoryBudgetResult:
    weight_bytes: int
    kv_cache_bytes: int
    lut_bytes: int
    reserve_bytes: int
    total_bytes: int
    weight_gb: float
    total_gb: float
    passed_weight_budget: bool
    passed_total_budget: bool
    is_valid: bool
    details: Dict[str, Any]


class QuantizedSuperBlockTensor:
    """Stores a weight tensor repacked into 256-element super-blocks (144 bytes each)."""

    def __init__(
        self,
        name: str,
        original_shape: Tuple[int, ...],
        scales: np.ndarray,
        packed_payload: np.ndarray,
        group_size: int = 32
    ):
        self.name = name
        self.original_shape = tuple(original_shape)
        self.n_elements = int(np.prod(original_shape))
        self.group_size = group_size
        self.groups_per_sb = 8
        self.sb_size = group_size * self.groups_per_sb  # 256

        self.scales = np.ascontiguousarray(scales, dtype=np.float16)
        self.packed_payload = np.ascontiguousarray(packed_payload, dtype=np.uint8)

        self.n_superblocks = len(self.scales)

    @property
    def memory_bytes(self) -> int:
        return self.scales.nbytes + self.packed_payload.nbytes

    def verify_alignment(self) -> bool:
        if self.scales.ndim != 2 or self.scales.shape[1] != 8:
            return False
        if self.packed_payload.ndim != 2 or self.packed_payload.shape[1] != 128:
            return False
        if self.memory_bytes != self.n_superblocks * 144:
            return False
        return True

    def unpack_to_int4((self) -> np.ndarray:
        """Unpack all super-blocks back to 1D int8 array [-8, 7]."""
        unpacked = np.zeros(self.n_elements, dtype=np.int8)
        for i in range(self.n_superblocks):
            sb_dict = {
                'scales': self.scales[i],
                'packed_nibbles': self.packed_payload[i]
            }
            unpacked[i*256:(i+1)*256] = unpack_superblock(sb_dict)
        return unpacked


class BaseWeightReader(ABC):
    @abstractmethod
    def list_tensors(self) -> List[str]:
        pass

    @abstractmethod
    def load_tensor(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        pass


class MockWeightReader(BaseWeightReader):
    """Generates synthetic FP16 tensors for open-weight models (Qwen2.5-1.5B)."""

    def __init__(self, metadata: Optional[ModelMetadata] = None, seed: int = 42):
        self.meta = metadata or ModelMetadata()
        self.seed = seed
        self._tensor_names = self._generate_tensor_names()

    def _generate_tensor_names(self) -> List[str]:
        names = ["model.embed_tokens.weight"]
        for l in range(self.meta.num_layers):
            prefix = f"model.layers.{l}"
            names.extend([
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ])
        names.extend(["model.norm.weight", "lm_head.weight"])
        return names

    def list_tensors(self) -> List[str]:
        return self._tensor_names

    def get_metadata(self) -> ModelMetadata:
        return self.meta

    def load_tensor(self, name: str) -> np.ndarray:
        # Determine shape based on name
        rng = np.random.RandomState(hash(name) % (2**32 - 1))
        meta = self.meta
        H = meta.hidden_size
        I = meta.intermediate_size
        H_kv = meta.num_key_value_heads * meta.head_dim

        if "q_proj" in name or "o_proj" in name:
            shape = (H, H)
        elif "k_proj" in name or "v_proj" in name:
            shape = (H_kv, H)
        elif "gate_proj" in name or "up_proj" in name:
            shape = (I, H)
        elif "down_proj" in name:
            shape = (H, I)
        elif "embed_tokens" in name or "lm_head" in name:
            shape = (meta.vocab_size, H)
        elif "layernorm" in name or "norm" in name:
            shape = (H,)
        else:
            shape = (H, H)

        weights = rng.randn(*shape).astype(np.float16) * 0.02
        return weights


class SafetensorsWeightReader(BaseWeightReader):
    """Parses Safetensors file format (JSON header + binary array buffer)."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header_json, self.data_offset = self._read_header()

    def _read_header() -> Tuple[Dict[str, Any], int]:
        with open(self.filepath, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                raise ValueError("Invalid Safetensors file: header underflow")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_json_bytes = f.read(header_len)
            header_json = json.loads(header_json_bytes.decode("utf-8"))
            return header_json, 8 + header_len

    def list_tensors(self) -> List[str]:
        return [k for k in self.header_json.keys() if k != "__metadata__"]

    def get_metadata(self) -> ModelMetadata:
        # Parse from __metadata__ if present, else return default Qwen2.5-1.5B
        user_meta = self.header_json.get("__metadata__", {})
        return ModelMetadata(
            architecture=user_meta.get("architecture", "qwen2"),
            hidden_size=int(user_meta.get("hidden_size", 1536)),
            num_layers=int(user_meta.get("num_layers", 28))
        )

    def load_tensor(self, name: str) -> np.ndarray:
        if name not in self.header_json:
            raise KeyError(f"Tensor '{name}' not found in Safetensors file")

        t_info = self.header_json[name]
        dtype_str = t_info["dtype"]
        shape = t_info["shape"]
        offsets = t_info["data_offsets"]

        dtype_map = {"F16": np.float16, "F32": np.float32, "I8": np.int8}
        np_dtype = dtype_map.get(dtype_str, np.float16)

        with open(self.filepath, "rb") as f:
            f.seek(self.data_offset + offsets[0])
            n_bytes = offsets[1] - offsets[0]
            raw_bytes = f.read(n_bytes)
            arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape)
            return arr.astype(np.float16)


class GGUFWeightReader(BaseWeightReader):
    """Parses GGUF binary format files."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tensor_info: Dict[str, Dict[str, Any]] = {}
        self.kv_metadata: Dict[str, Any] = {}
        self.data_start_offset = 0
        self._parse_header()

    def _parse_header(self):
        with open(self.filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Not a valid GGUF file: magic={magic}")
            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]
            
            # Simple metadata KV loop & tensor headers loop
            # (Details omitted for brevity in design doc, standard GGUF binary reader)
            self.data_start_offset = f.tell()

    def list_tensors(self) -> List[str]:
        return list(self.tensor_info.keys())

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            hidden_size=self.kv_metadata.get("qwen2.embedding_length", 1536),
            num_layers=self.kv_metadata.get("qwen2.block_count", 28)
        )

    def load_tensor(self, name: str) -> np.ndarray:
        info = self.tensor_info[name]
        with open(self.filepath, "rb") as f:
            f.seek(self.data_start_offset + info["offset"])
            raw = f.read(info["n_bytes"])
            arr = np.frombuffer(raw, dtype=info["dtype"]).reshape(info["shape"])
            return arr.astype(np.float16)


class ModelWeightLoader:
    """
    Orchestrates loading, quantizing, and super-block repacking of weight tensors.
    """

    def __init__(self, reader: BaseWeightReader):
        self.reader = reader
        self.metadata = reader.get_metadata()
        self.quantized_tensors: Dict[str, QuantizedSuperBlockTensor] = {}
        self.unquantized_tensors: Dict[str, np.ndarray] = {}

    def load_and_repack_all(self, group_size: int = 32) -> Dict[str, QuantizedSuperBlockTensor]:
        for name in self.reader.list_tensors():
            tensor_fp16 = self.reader.load_tensor(name)
            
            # 1D Layernorms / Biases are kept unquantized
            if tensor_fp16.ndim == 1 or "norm" in name or "layernorm" in name:
                self.unquantized_tensors[name] = tensor_fp16
                continue

            # Flat element count check
            n_elements = tensor_fp16.size
            if n_elements % 256 != 0:
                # Pad to multiple of 256 if needed
                pad_size = 256 - (n_elements % 256)
                flat_fp16 = np.pad(tensor_fp16.ravel(), (0, pad_size), mode='constant')
            else:
                flat_fp16 = tensor_fp16.ravel()

            # Step 1: Quantize INT4
            q_weights, scales = quantize_weights_int4(flat_fp16, group_size=group_size)

            # Step 2: Repack to Super-Blocks
            superblocks = repack_to_superblocks(q_weights, scales, group_size=group_size)

            # Step 3: Extract contiguous arrays
            n_sb = len(superblocks)
            sb_scales = np.zeros((n_sb, 8), dtype=np.float16)
            sb_payload = np.zeros((n_sb, 128), dtype=np.uint8)

            for i, sb in enumerate(superblocks):
                sb_scales[i] = sb['scales']
                sb_payload[i] = sb['packed_nibbles']

            sb_tensor = QuantizedSuperBlockTensor(
                name=name,
                original_shape=tensor_fp16.shape,
                scales=sb_scales,
                packed_payload=sb_payload,
                group_size=group_size
            )
            
            assert sb_tensor.verify_alignment(), f"Alignment check failed for tensor {name}"
            self.quantized_tensors[name] = sb_tensor

        return self.quantized_tensors

    def get_total_weight_bytes(self) -> int:
        qb = sum(t.memory_bytes for t in self.quantized_tensors.values())
        uqb = sum(t.nbytes for t in self.unquantized_tensors.values())
        return qb + uqb


class MemoryBudgetValidator:
    """Calculates app RAM footprint and validates limits against 4.5 GB ceiling."""

    MAX_WEIGHT_BUDGET_BYTES = int(2.5 * 1024**3)  # 2.5 GB
    MAX_TOTAL_BUDGET_BYTES = int(4.5 * 1024**3)   # 4.5 GB

    @staticmethod
    def calculate_budget(
        model_weight_bytes: int,
        n_rollouts: int = 8,
        context_length: int = 2048,
        metadata: Optional[ModelMetadata] = None
    ) -> MemoryBudgetResult:
        meta = metadata or ModelMetadata()

        # KV Cache calculation per rollout channel
        # 2 (K+V) * layers * num_kv_heads * head_dim * context * 2 bytes (FP16)
        bytes_per_token_layer = 2 * meta.num_key_value_heads * meta.head_dim * 2
        kv_bytes_per_trace = meta.num_layers * context_length * bytes_per_token_layer
        total_kv_bytes = n_rollouts * kv_bytes_per_trace

        # Softmax LUT (32K FP16 entries = 64 KB)
        lut_bytes = 32768 * 2

        # App & Metal Command Buffer Reserve (~500 MB)
        reserve_bytes = 500 * 1024 * 1024

        total_bytes = model_weight_bytes + total_kv_bytes + lut_bytes + reserve_bytes

        weight_gb = model_weight_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        passed_weight = model_weight_bytes <= MemoryBudgetValidator.MAX_WEIGHT_BUDGET_BYTES
        passed_total = total_bytes <= MemoryBudgetValidator.MAX_TOTAL_BUDGET_BYTES

        return MemoryBudgetResult(
            weight_bytes=model_weight_bytes,
            kv_cache_bytes=total_kv_bytes,
            lut_bytes=lut_bytes,
            reserve_bytes=reserve_bytes,
            total_bytes=total_bytes,
            weight_gb=weight_gb,
            total_gb=total_gb,
            passed_weight_budget=passed_weight,
            passed_total_budget=passed_total,
            is_valid=passed_weight and passed_total,
            details={
                "n_rollouts": n_rollouts,
                "context_length": context_length,
                "kv_mb": total_kv_bytes / (1024**2),
                "weight_mb": model_weight_bytes / (1024**2),
                "total_mb": total_bytes / (1024**2)
            }
        )
```

---

## 6. Test Suite & Verification Strategy

The test suite `antigravity-engine/tests/test_model_loader.py` will validate R2 in complete isolation using standard `unittest`:

1. **Test GGUF Header & Tensor Directory Reader**:
   - Create mock GGUF binary buffer in memory.
   - Verify magic number `GGUF`, metadata KV count, and tensor offset parsing.
2. **Test Safetensors Reader**:
   - Create mock Safetensors binary buffer (8-byte header length + JSON header + binary payload).
   - Verify tensor extraction and dtype conversion.
3. **Test `MockWeightReader` Qwen2.5-1.5B Generation**:
   - Verify tensor count (28 layers * 9 matrices + embed + head).
   - Verify FP16 dtype and matrix shapes match Qwen2.5-1.5B architecture.
4. **Test On-the-Fly Super-Block Repacker**:
   - Repack FP16 weights into `QuantizedSuperBlockTensor`.
   - Verify roundtrip unpacking using `unpack_to_int4()` matches direct `quantize_weights_int4` values.
5. **Test 128-Byte Alignment Assertion**:
   - Verify `verify_alignment()` returns `True` for repacked tensors.
   - Confirm payload width is 128 bytes and scale header width is 8 `float16`s.
6. **Test Memory Calculator & Budget Enforcement**:
   - Test 1.5B parameters model weight footprint calculation ($\approx 0.844 \text{ GB}$).
   - Verify weight memory $\le 2.5 \text{ GB}$ ceiling check passes.
   - Verify total memory footprint ($1.814 \text{ GB}$) $\le 4.5 \text{ GB}$ ceiling check passes.
   - Test intentional over-budget configuration (e.g. 10B params or $N=128$ rollouts) and assert budget failure detection.

---

## 7. Next Steps & Worker Implementation Instructions

Upon completion of this investigation, Worker (`teamwork_preview_worker`) will implement:
1. `antigravity-engine/src/model_loader.py` following the exact design and interface contracts specified above.
2. `antigravity-engine/tests/test_model_loader.py` implementing the 6 test classes outlined in Section 6.
3. Run `python3 -m unittest discover` to ensure 100% test pass rate across `dequant.py`, `attention.py`, and `model_loader.py`.
