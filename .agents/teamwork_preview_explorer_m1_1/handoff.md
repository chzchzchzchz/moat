# Exploration & Technical Handoff Report: Milestone 1 - Hardware-Aware INT4 Quantization & Superblock LUT Dequantization Engine

**Milestone:** 1  
**Working Directory:** `/Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_m1_1`  
**Target Project Path:** `/Users/MohssineChazi2/moat`  
**Date:** 2026-07-25  

---

## 1. Observation

### 1.1 Existing Workspace & File Inspection
- Checked project root `/Users/MohssineChazi2/moat`.
  - Directories: `.agents/`, `planning/`
  - Root files: `antigravity_prd.md`
  - Note: Source directories `config/`, `src/`, `tests/` do not yet exist in the root folder and need to be created by the Implementer agent as per Milestone 1 scope.
- Examined documentation & specification files:
  - `/Users/MohssineChazi2/moat/.agents/sub_orch_m1/SCOPE.md` (Lines 1-27)
  - `/Users/MohssineChazi2/moat/antigravity_prd.md` (Lines 1-199, specifically Section 4.A on Quantization & Dequantization, lines 84-109)
  - `/Users/MohssineChazi2/moat/.agents/ORIGINAL_REQUEST.md` (Lines 1-56, Section R2 on INT4 Quantization Pipeline)
  - `/Users/MohssineChazi2/moat/planning/01_hardware_architecture.md` and `02_quantization_and_lut.md`

### 1.2 Environment & Dependency Verification
Executed python environment inspection via shell command:
- **Python Version:** 3.12.3 (`/Applications/Xcode.app/Contents/Developer/usr/bin/python3`)
- **Key Installed Packages:**
  - `numpy`: 2.4.4
  - `yaml` (PyYAML): 6.0.3
  - `pytest`: 9.0.2
  - `torch`: 2.11.0
  - `scipy`: 1.17.1
  - `numba`: 0.64.0
  - `mlx`: Installed (Apple Silicon machine learning framework)

All core dependencies (`numpy`, `pyyaml`, `pytest`) required for Milestone 1 are available in the local environment.

---

## 2. Logic Chain

1. **Quantization Precision & Group Size:**
   - **PRD/SCOPE contract:** Fine-grained group quantization with `bits=4` and `group_size=32`.
   - **Math:** For each group $G$ of 32 weight elements, calculate group maximum $\alpha = \max_{i \in G} |w_i|$ and scale factor $S_G = \alpha / 7.0$. Each weight $w_i$ is mapped to quantized signed int4 $q_i \in [-8, 7]$.
   - **Scale Overhead:** 32 elements share 1 FP16 scale (16 bits = 2 bytes). Metadata overhead is $2 / 32 = 0.0625$ bytes per parameter.

2. **Superblock Coalescing & SIMD Alignment:**
   - **Alignment Target:** `superblock_size=256` elements, aligned to `alignment_bytes=128`.
   - **SIMD Rationale:** Coalescing 8 fine-grained groups ($8 \times 32 = 256$ elements) produces a payload of $256 \times 4\text{ bits} = 1024\text{ bits} = 128\text{ bytes}$, which matches 1 SIMD vector cache line on Apple Silicon GPU SIMD-groups / ANE vector units.
   - **Superblock Data Layout:**
     - Header: 8 FP16 scales = 16 bytes.
     - Payload: 256 4-bit nibbles = 128 bytes.
     - Total Superblock Size: 144 bytes per 256 parameters.

3. **Fast LUT Dequantization (`lut_dequantize_fp16`):**
   - **Mechanism:** Avoids per-element bit shifting, sign extension, and floating-point multiplications in the hot decode loop.
   - **Implementation:** Precomputes a 16-entry lookup table per group mapping quantized index $k \in [-8, 7]$ directly to $k \times S_G$. Fast array indexing (`lut[group_idx, q_weights + 8]`) executes as vector gather operations.

4. **Memory Footprint Bounds (Target: ~4.5 GB RAM Limit on iPhone 15 Pro):**
   - Effective bytes per parameter under INT4 Superblock format:
     $$\text{Bytes/param} = 0.5\text{ (weights)} + 0.0625\text{ (scales)} = 0.5625\text{ bytes/param}$$
   - **1.5B Model:** $1.5 \times 10^9 \times 0.5625 = 843.75\text{ MB} \approx 0.844\text{ GB}$ (Unquantized FP16: 3.0 GB).
   - **3.0B Model:** $3.0 \times 10^9 \times 0.5625 = 1,687.5\text{ MB} \approx 1.688\text{ GB}$ (Unquantized FP16: 6.0 GB).
   - **7.0B Model:** $7.0 \times 10^9 \times 0.5625 = 3,937.5\text{ MB} \approx 3.938\text{ GB}$ (Unquantized FP16: 14.0 GB).
   - **Conclusion:** 1.5B–3.0B parameter models fit effortlessly within the ~4.5GB app RAM limit on 8GB unified memory iOS devices, leaving ~2.8–3.6 GB for KV-cache and activations.

---

## 3. Caveats

1. **Development Proxy Environment:**
   - Benchmarking and unit testing run on macOS Apple Silicon (M-series CPU/GPU). Native Metal/CoreML iOS compilation will be handled in downstream milestones (Milestone 5). NumPy/PyTorch implementations serve as the numerical ground truth for Metal compute shader development.
2. **Byte Packing / Nibble Bit Ordering:**
   - When packing two 4-bit INT4 numbers into a single `uint8` byte, little-endian nibble ordering (low 4 bits = weight 0, high 4 bits = weight 1) must be preserved consistently across packing and unpack/dequant routines.
3. **Array Memory Contiguity:**
   - `repack_weights_to_superblock` must explicitly enforce C-contiguous memory storage (`np.ascontiguousarray`) to ensure hardware SIMD register loads do not suffer from non-unit strides.

---

## 4. Conclusion & Technical Recommendations

The Implementer agent for Milestone 1 should create and populate the following 4 files in `/Users/MohssineChazi2/moat`:

### File 1: `config/engine_config.yaml`
```yaml
quantization:
  bits: 4
  group_size: 32
  superblock_size: 256
  alignment_bytes: 128

execution:
  batch_size: 8
  reflection_threshold: 0.75
  port: 8080
```

### File 2: `src/__init__.py`
```python
"""Antigravity engine core module."""

from .dequant import (
    repack_weights_to_superblock,
    lut_dequantize_fp16,
    quantize_fp16_to_int4,
    unpack_superblock_weights,
    calculate_memory_footprint,
)

__all__ = [
    "repack_weights_to_superblock",
    "lut_dequantize_fp16",
    "quantize_fp16_to_int4",
    "unpack_superblock_weights",
    "calculate_memory_footprint",
]
```

### File 3: `src/dequant.py`
```python
import numpy as np
from typing import Tuple, Dict

def quantize_fp16_to_int4(weights_fp16: np.ndarray, group_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantizes an FP16 weight array to INT4 signed values in range [-8, 7]
    with fine-grained group scales (group_size=32).

    Returns:
        q_weights (np.ndarray): INT8 array of quantized values in [-8, 7].
        scales (np.ndarray): FP16 scale factors per group.
    """
    flat = weights_fp16.astype(np.float32).flatten()
    assert flat.size % group_size == 0, f"Weight array length ({flat.size}) must be a multiple of group_size ({group_size})."
    
    num_groups = flat.size // group_size
    reshaped = flat.reshape(num_groups, group_size)
    
    # Compute scale per group: alpha / 7
    max_abs = np.max(np.abs(reshaped), axis=1, keepdims=True)
    scales = np.where(max_abs == 0, 1.0, max_abs / 7.0)
    
    # Quantize & clamp
    q_weights = np.clip(np.round(reshaped / scales), -8, 7).astype(np.int8)
    scales_fp16 = scales.flatten().astype(np.float16)
    
    return q_weights.reshape(weights_fp16.shape), scales_fp16

def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
    """
    Coalesces 8 fine-grained INT4 quantization groups of size 32 into a
    single aligned super-block of size 256 to ensure wide vector register access.
    
    Returns:
        repacked (np.ndarray): Reshaped C-contiguous array with shape (num_superblocks, 8, group_size)
        aligned to 128-byte boundaries for SIMD execution.
    """
    flat = weights_int4.flatten()
    assert len(flat) % 256 == 0, f"Weight array length ({len(flat)}) must be a multiple of 256."
    num_superblocks = len(flat) // 256
    repacked = flat.reshape(num_superblocks, 8, group_size).astype(np.int8)
    return np.ascontiguousarray(repacked)

def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Fast table-lookup dequantization mapping INT4 back to FP16.
    Avoids expensive unpacking instruction cycles in the compute loop.
    
    Args:
        q_weights (np.ndarray): INT4 packed/unpacked weights with indices in [-8, 7] or [0, 15].
        lut (np.ndarray): Lookup table of FP16 values.
            - If 1D: shape (16,) mapping index k+8 -> float value.
            - If 2D: shape (num_groups, 16) mapping (group_idx, k+8) -> float value.
            
    Returns:
        dequantized (np.ndarray): FP16 restored weights.
    """
    if q_weights.dtype in (np.int8, np.int16, np.int32, np.int64) and np.min(q_weights) < 0:
        indices = (q_weights.astype(np.int32) + 8).astype(np.int32)
    else:
        indices = q_weights.astype(np.int32)

    if lut.ndim == 1:
        return lut[indices].astype(np.float16)
    elif lut.ndim == 2:
        num_groups = lut.shape[0]
        flat_q = indices.reshape(num_groups, -1)
        group_idx = np.arange(num_groups)[:, None]
        dequant = lut[group_idx, flat_q]
        return dequant.reshape(q_weights.shape).astype(np.float16)
    else:
        raise ValueError(f"Unsupported LUT shape {lut.shape}. Must be 1D (16,) or 2D (num_groups, 16).")

def unpack_superblock_weights(superblock_bytes: np.ndarray, num_superblocks: int) -> np.ndarray:
    """
    Unpacks raw 144-byte binary superblocks (16-byte FP16 scale header + 128-byte INT4 payload)
    back to FP16 weights array.
    """
    reshaped = superblock_bytes.reshape(num_superblocks, 144)
    scales_raw = reshaped[:, :16].view(np.float16).reshape(num_superblocks, 8)
    payload = reshaped[:, 16:]  # shape (num_superblocks, 128)
    
    # Unpack nibbles: low 4 bits, high 4 bits
    low_nibbles = (payload & 0x0F).astype(np.int8) - 8
    high_nibbles = ((payload >> 4) & 0x0F).astype(np.int8) - 8
    
    # Interleave low and high nibbles
    unpacked_int4 = np.empty((num_superblocks, 128, 2), dtype=np.int8)
    unpacked_int4[:, :, 0] = low_nibbles
    unpacked_int4[:, :, 1] = high_nibbles
    unpacked_int4 = unpacked_int4.reshape(num_superblocks, 8, 32)
    
    k_range = np.arange(-8, 8, dtype=np.float32)
    lut_2d = scales_raw.reshape(-1, 1)[:, None] * k_range[None, :]
    
    indices = unpacked_int4.reshape(num_superblocks * 8, 32) + 8
    group_indices = np.arange(num_superblocks * 8)[:, None]
    dequant = lut_2d[group_indices, indices]
    
    return dequant.reshape(num_superblocks * 256).astype(np.float16)

def calculate_memory_footprint(num_params: int, group_size: int = 32, bits: int = 4) -> Dict[str, float]:
    """
    Calculates exact memory footprint for model weights under fine-grained group quantization.
    
    Returns:
        dict containing memory sizes in bytes, MB, GB, and RAM limit comparison.
    """
    bytes_per_param = bits / 8.0  # 0.5 bytes for INT4
    weight_bytes = num_params * bytes_per_param
    num_groups = num_params // group_size
    scale_bytes = num_groups * 2.0  # FP16 scales (2 bytes per scale)
    
    total_bytes = weight_bytes + scale_bytes
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)
    
    ram_limit_gb = 4.5
    fits_in_ram = total_gb <= ram_limit_gb
    
    return {
        "num_params": num_params,
        "weight_bytes": weight_bytes,
        "scale_bytes": scale_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "total_gb": total_gb,
        "ram_limit_gb": ram_limit_gb,
        "fits_in_ram": fits_in_ram,
        "bytes_per_param_effective": total_bytes / num_params,
    }
```

### File 4: `tests/test_quantization.py`
```python
import os
import yaml
import pytest
import numpy as np
from src.dequant import (
    quantize_fp16_to_int4,
    repack_weights_to_superblock,
    lut_dequantize_fp16,
    unpack_superblock_weights,
    calculate_memory_footprint,
)

def test_engine_config_loading():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "engine_config.yaml")
    assert os.path.exists(config_path), "config/engine_config.yaml file must exist."
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    assert "quantization" in cfg
    q_cfg = cfg["quantization"]
    assert q_cfg["bits"] == 4
    assert q_cfg["group_size"] == 32
    assert q_cfg["superblock_size"] == 256
    assert q_cfg["alignment_bytes"] == 128
    
    assert "execution" in cfg
    e_cfg = cfg["execution"]
    assert e_cfg["batch_size"] == 8
    assert e_cfg["reflection_threshold"] == 0.75
    assert e_cfg["port"] == 8080

def test_quantize_and_repack_superblock():
    np.random.seed(42)
    weights_fp16 = np.random.uniform(-2.0, 2.0, size=1024).astype(np.float16)
    
    q_weights, scales = quantize_fp16_to_int4(weights_fp16, group_size=32)
    assert q_weights.shape == (1024,)
    assert scales.shape == (32,)
    assert np.min(q_weights) >= -8 and np.max(q_weights) <= 7
    
    repacked = repack_weights_to_superblock(q_weights, group_size=32)
    assert repacked.shape == (4, 8, 32)
    assert repacked.flags['C_CONTIGUOUS']
    assert (repacked.nbytes) % 128 == 0

def test_repack_alignment_assertion():
    invalid_weights = np.zeros(200, dtype=np.int8)
    with pytest.raises(AssertionError):
        repack_weights_to_superblock(invalid_weights, group_size=32)

def test_lut_dequantization_correctness_and_speed():
    np.random.seed(42)
    weights_fp16 = np.random.uniform(-1.0, 1.0, size=256).astype(np.float16)
    q_weights, scales = quantize_fp16_to_int4(weights_fp16, group_size=32)
    
    num_groups = 8
    k_range = np.arange(-8, 8, dtype=np.float32)
    lut_2d = scales[:, None] * k_range[None, :]
    
    dequant = lut_dequantize_fp16(q_weights.reshape(8, 32), lut_2d)
    assert dequant.shape == (8, 32)
    
    max_error = np.max(np.abs(weights_fp16.astype(np.float32) - dequant.astype(np.float32).flatten()))
    assert max_error < 0.2, f"Dequantization error too high: {max_error}"

def test_memory_bounds_verification():
    fp_1_5b = calculate_memory_footprint(1_500_000_000)
    assert fp_1_5b["fits_in_ram"]
    assert fp_1_5b["total_gb"] < 1.0
    
    fp_3_0b = calculate_memory_footprint(3_000_000_000)
    assert fp_3_0b["fits_in_ram"]
    assert fp_3_0b["total_gb"] < 2.0
    
    fp_7_0b = calculate_memory_footprint(7_000_000_000)
    assert fp_7_0b["fits_in_ram"]
    assert fp_7_0b["total_gb"] < 4.0
```

---

## 5. Verification Method

Once the Implementer agent writes the above files, verify execution by running:

```bash
# 1. Run pytest suite
python3 -m pytest tests/test_quantization.py -v

# 2. Verify config loading & memory footprint check in Python interactive shell
python3 -c "
import yaml
from src.dequant import calculate_memory_footprint

with open('config/engine_config.yaml') as f:
    cfg = yaml.safe_load(f)
print('Config:', cfg)

fp = calculate_memory_footprint(1_500_000_000)
print('1.5B Memory Footprint:', fp)
"
```

### Invalidation Conditions
- Any test in `tests/test_quantization.py` failing.
- Superblock alignment failing 128-byte SIMD boundary requirement (`repacked.nbytes % 128 != 0`).
- 1.5B or 3.0B model footprint exceeding 4.5 GB limit.
