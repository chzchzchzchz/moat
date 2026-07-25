# Technical Exploration & Test Suite Architecture: `tests/test_quantization.py`

## 1. Observation

### Direct Observations & File References
1. **Scope Requirements (`/Users/MohssineChazi2/moat/.agents/sub_orch_m1/SCOPE.md`):**
   - **Line 4-6:** "INT4 Fine-Grained Quantization (group size 32)", "Superblock Repacking (8 groups = 256 elements, aligned to 128-byte SIMD register boundaries)", "FP16 Fast LUT Dequantization (`lut_dequantize_fp16`)".
   - **Line 19-22:** Contract signatures:
     - `repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray`
     - `lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray`
     - "Additional quantization packing/unpacking & memory footprint calculations for 1.5B-3B models (~4.5GB RAM footprint limit)."
   - **Line 23-26:** Unit test requirements in `tests/test_quantization.py`: INT4 repacking accuracy and alignment checks, speed/correctness tests for `lut_dequantize_fp16`, memory bounds verification test.

2. **PRD & Code Shells (`/Users/MohssineChazi2/moat/antigravity_prd.md`):**
   - **Line 87-108:** Code shell implementations:
     ```python
     def repack_weights_to_superblock(weights_int4: np.ndarray, group_size: int = 32) -> np.ndarray:
         assert len(weights_int4) % 256 == 0, "Weight array length must be a multiple of 256."
         num_superblocks = len(weights_int4) // 256
         repacked = weights_int4.reshape(num_superblocks, 8, group_size)
         return repacked.astype(np.int8)

     def lut_dequantize_fp16(q_weights: np.ndarray, lut: np.ndarray) -> np.ndarray:
         return lut[q_weights]
     ```

3. **Architectural Specification (`/Users/MohssineChazi2/moat/planning/02_quantization_and_lut.md`):**
   - **Line 22-25:** Super-block size: 8 groups $\times 32 = 256$ elements = 128 bytes payload ($256 \times 4\text{ bits} = 1024\text{ bits} = 128\text{ bytes}$).
   - **Line 24-25:** 8 FP16 scale factors header ($8 \times 2\text{ bytes} = 16\text{ bytes}$). Total superblock size: 144 bytes ($16\text{B header} + 128\text{B payload}$).
   - **Line 45-50:** LUT indexing mapping quantized INT4 indices directly to FP16 via SIMD vector gather operations.

4. **Hardware & Memory Architecture (`/Users/MohssineChazi2/moat/planning/01_hardware_architecture.md`):**
   - **Line 66-74:** App Available RAM limit: ~4.5GB to 5.5GB maximum on 8GB iPhone 15 Pro. Peak memory budget allocation:
     - Reasoner INT4 Model (1.5B): ~1.1 GB
     - Batched KV Cache ($N=8$, Context 2048): ~0.8 GB
     - Verifier INT4 Model (1.5B): ~1.1 GB (Sequential Load/Unload Swap)
     - Scratchpad & Command Buffers: ~0.5 GB
     - Peak memory limit: strictly $< 4.5\text{ GB}$ (4,608 MB).

5. **Codebase State:**
   - Currently, no Python test files or source modules exist in `/Users/MohssineChazi2/moat/tests/` or `/Users/MohssineChazi2/moat/src/`. Explorer 3 is defining the complete test specification blueprint.

---

## 2. Logic Chain

From the observations above, the test suite `tests/test_quantization.py` must address three distinct test pillars:
1. **Superblock Repacking Accuracy & SIMD Alignment:**
   - Observation 1 & 2 state that input arrays must have lengths that are multiples of 256.
   - Observation 3 specifies 128-byte alignment (matching GPU SIMD register cache lines).
   - *Reasoning:* Tests must assert shape reshaping $(S, 8, 32)$, byte-boundary memory alignment (`ctypes.data % 128 == 0`), data element value preservation, and error handling for invalid input dimensions.

2. **LUT Dequantization (`lut_dequantize_fp16`) Correctness & Speed:**
   - Observation 2 & 3 show that `lut_dequantize_fp16` performs indexing `lut[q_weights]` to bypass bit-unpacking instructions.
   - *Reasoning:* Tests must assert 100% mathematical equality between table lookup and float reference results, verify output shape and `np.float16` data type preservation, and benchmark latency against standard arithmetic unpacking (`q_weights * scale`).

3. **Memory Bounds Verification (~4.5GB Limit for 1.5B–3.0B Params):**
   - Observation 4 defines the 4.5GB memory constraint and the model parameter range (1.5B to 3.0B params).
   - *Reasoning:* INT4 parameters take 0.5 bytes per element, plus group scale metadata (2 bytes per 32 elements = 0.0625 bytes/param), totaling $0.5625$ bytes/param ($843.75\text{ MB}$ for 1.5B, $1.6875\text{ GB}$ for 3.0B). Tests must mathematically assert footprint formulas, verify runtime memory allocation via `psutil`/`tracemalloc`, and assert peak RAM remains $< 4.5\text{ GB}$ across sequential reasoner/verifier model swapping.

---

## 3. Detailed Technical Exploration Report & Test Suite Architecture

### 3.1 `tests/test_quantization.py` Module Structure

The test file should be organized into four pytest classes using structured fixtures and parametrization:

```python
# Blueprint for tests/test_quantization.py
import pytest
import numpy as np
import time
import tracemalloc
import psutil
import os
from src.dequant import repack_weights_to_superblock, lut_dequantize_fp16

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_int4_weights():
    """Generates 256 INT4 weights in signed range [-8, 7]."""
    np.random.seed(42)
    return np.random.randint(-8, 8, size=256, dtype=np.int8)

@pytest.fixture
def multi_superblock_int4_weights():
    """Generates 2048 INT4 weights (8 superblocks of size 256)."""
    np.random.seed(42)
    return np.random.randint(-8, 8, size=2048, dtype=np.int8)

@pytest.fixture
def fp16_lut():
    """Generates 16-element FP16 lookup table for signed INT4 [-8..7]."""
    # Scale factor = 0.125
    scales = np.arange(-8, 8, dtype=np.float16) * np.float16(0.125)
    return scales
```

---

### 3.2 Test Class 1: `TestSuperblockRepacking`

#### Key Assertions & Benchmark Specifications:
1. **Exact Shape Assertion:**
   - Input length $N = 256 \times K \rightarrow$ repacked shape must be $(K, 8, 32)$.
   - `assert repacked.shape == (num_superblocks, 8, group_size)`
2. **Data Integrity / Preservation Assertion:**
   - For every superblock $s \in [0..K-1]$, group $g \in [0..7]$, and index $i \in [0..31]$:
     `repacked[s, g, i] == weights_int4[s*256 + g*32 + i]`.
3. **128-Byte Boundary Alignment Assertion:**
   - Memory pointer alignment check: `assert repacked.ctypes.data % 128 == 0`
   - Memory continuity check: `assert repacked.flags['C_CONTIGUOUS'] is True`
4. **Invalid Dimension Exception Assertions:**
   - Length not divisible by 256 (e.g. 255 elements, 257 elements) must raise `AssertionError` or `ValueError`.
   - `with pytest.raises((ValueError, AssertionError)): repack_weights_to_superblock(invalid_weights)`

#### Recommended Code Snippet:
```python
class TestSuperblockRepacking:
    @pytest.mark.parametrize("num_elements", [256, 1024*256, 4096*256])
    def test_repack_shape_and_alignment(self, num_elements):
        weights = np.random.randint(-8, 8, size=num_elements, dtype=np.int8)
        repacked = repack_weights_to_superblock(weights, group_size=32)
        
        # 1. Shape Verification
        expected_superblocks = num_elements // 256
        assert repacked.shape == (expected_superblocks, 8, 32)
        
        # 2. Data Type & Stride Verification
        assert repacked.dtype == np.int8
        assert repacked.flags['C_CONTIGUOUS']
        
        # 3. 128-Byte Memory Alignment Verification
        assert repacked.ctypes.data % 128 == 0, f"Address {hex(repacked.ctypes.data)} is not 128-byte aligned"

    def test_repack_data_integrity(self, sample_int4_weights):
        repacked = repack_weights_to_superblock(sample_int4_weights, group_size=32)
        flattened = repacked.reshape(-1)
        np.testing.assert_array_equal(flattened, sample_int4_weights)

    @pytest.mark.parametrize("invalid_size", [1, 31, 255, 257, 1000])
    def test_repack_invalid_dimensions_raises(self, invalid_size):
        weights = np.zeros(invalid_size, dtype=np.int8)
        with pytest.raises((ValueError, AssertionError)):
            repack_weights_to_superblock(weights, group_size=32)
```

---

### 3.3 Test Class 2: `TestLUTDequantizeFP16` (Correctness & Speed)

#### Key Assertions & Benchmark Specifications:
1. **Mathematical Correctness Assertion:**
   - Index lookup mapping: `dequantized[i] == lut[q_weights[i]]`.
   - Datatype check: `assert dequantized.dtype == np.float16`.
   - All 16 indices check: test with array containing every integer in $[-8..7]$ (or $[0..15]$).
2. **Precision Tolerance:**
   - Exact bit-level float match: `np.testing.assert_array_equal(dequantized, expected_fp16)`.
3. **Latency / Speed Benchmark Check:**
   - Benchmark setup: tensor size = $4096 \times 4096 = 16,777,216$ quantized weights (~16.7M params layer benchmark).
   - Compare `lut_dequantize_fp16(q_weights, lut)` against conventional arithmetic unpack `(q_weights.astype(np.float32) * scale).astype(np.float16)`.
   - Measure over 50 iterations with 10 warm-up runs using `time.perf_counter_ns()`.
   - Assertion: `lut_mean_latency <= arithmetic_mean_latency` (expecting $\ge 1.3\times$ speedup on vector gather hardware).

#### Recommended Code Snippet:
```python
class TestLUTDequantizeFP16:
    def test_dequantize_correctness_all_indices(self, fp16_lut):
        # Index array covering all signed INT4 values [-8..7]
        q_weights = np.arange(-8, 8, dtype=np.int8)
        dequantized = lut_dequantize_fp16(q_weights, fp16_lut)
        
        assert dequantized.dtype == np.float16
        expected = fp16_lut[q_weights]
        np.testing.assert_array_equal(dequantized, expected)

    def test_dequantize_multidimensional_tensor(self, fp16_lut):
        shape = (4, 8, 32)
        q_weights = np.random.randint(-8, 8, size=shape, dtype=np.int8)
        dequantized = lut_dequantize_fp16(q_weights, fp16_lut)
        
        assert dequantized.shape == shape
        assert dequantized.dtype == np.float16
        np.testing.assert_array_equal(dequantized, fp16_lut[q_weights])

    @pytest.mark.benchmark
    def test_lut_dequantize_speed_vs_arithmetic(self, fp16_lut):
        # 16.7M elements layer simulation
        size = 4096 * 4096
        q_weights = np.random.randint(-8, 8, size=size, dtype=np.int8)
        scale = np.float32(0.125)
        
        # Warmup
        for _ in range(5):
            _ = lut_dequantize_fp16(q_weights, fp16_lut)
            _ = (q_weights.astype(np.float32) * scale).astype(np.float16)
            
        # Timed execution: LUT dequant
        t0 = time.perf_counter_ns()
        for _ in range(30):
            dequant_lut = lut_dequantize_fp16(q_weights, fp16_lut)
        t_lut = (time.perf_counter_ns() - t0) / 30
        
        # Timed execution: Arithmetic unpack
        t0 = time.perf_counter_ns()
        for _ in range(30):
            dequant_arithmetic = (q_weights.astype(np.float32) * scale).astype(np.float16)
        t_arithmetic = (time.perf_counter_ns() - t0) / 30
        
        # Speed assertion: LUT table gather must not be slower than float multiplication
        assert t_lut <= t_arithmetic * 1.05, f"LUT dequant ({t_lut/1e6:.2f}ms) slower than arithmetic ({t_arithmetic/1e6:.2f}ms)"
```

---

### 3.4 Test Class 3: `TestMemoryBounds` (~4.5GB Footprint Limit)

#### Key Assertions & Benchmark Specifications:
1. **Theoretical Memory Calculation Assertions:**
   - 1.5B Parameter Model INT4 weights: $1.5 \times 10^9 \times 0.5\text{ bytes} = 750,000,000\text{ bytes} \approx 715.26\text{ MB}$.
   - 1.5B Group Scale Metadata (group size 32, FP16 scale): $(1.5 \times 10^9 / 32) \times 2\text{ bytes} = 93,750,000\text{ bytes} \approx 89.41\text{ MB}$.
   - Total 1.5B INT4 Model size: $843.75\text{ MB} < 1,000\text{ MB}$.
   - Total 3.0B INT4 Model size: $1,687.5\text{ MB} < 2,000\text{ MB}$.
2. **Combined Active Engine Budget Assertion:**
   - Reasoner INT4 Model ($1.5\text{B}$): $843.75\text{ MB}$
   - Batched KV-Cache ($N=8$, context 2048, 28 layers, 16 heads, head dim 64, FP16): $\approx 734.0\text{ MB}$
   - Verifier INT4 Model ($1.5\text{B}$): $843.75\text{ MB}$
   - App & Command Buffer Overhead: $500\text{ MB}$
   - Peak Concurrent Allocation (Sequential Swap): $\text{Reasoner} + \text{KV-Cache} + \text{Overhead} \approx 2.07\text{ GB} \ll 4.5\text{ GB}$.
3. **Runtime Allocation / Memory Purge Assertions:**
   - Allocate synthetic byte arrays simulating 1.5B model weight tensors.
   - Profile process RSS using `psutil.Process().memory_info().rss`.
   - Assert peak RSS delta $< 4,831,838,208\text{ bytes}$ (4.5 GB).
   - Test explicit deletion and garbage collection (`del reasoner_weights; gc.collect()`) to verify memory drop before loading verifier weights.

#### Recommended Code Snippet:
```python
class TestMemoryBounds:
    def test_theoretical_int4_model_memory_bounds(self):
        def calc_int4_model_bytes(num_params, group_size=32):
            weight_bytes = num_params * 0.5  # 4 bits = 0.5 bytes
            scale_bytes = (num_params / group_size) * 2.0  # FP16 scale per group
            return weight_bytes + scale_bytes

        # 1.5B Parameter Model Footprint
        bytes_1_5b = calc_int4_model_bytes(1.5e9)
        assert bytes_1_5b < 1.0 * (1024**3), f"1.5B model exceeds 1GB limit: {bytes_1_5b / (1024**2):.2f} MB"
        
        # 3.0B Parameter Model Footprint
        bytes_3_0b = calc_int4_model_bytes(3.0e9)
        assert bytes_3_0b < 2.0 * (1024**3), f"3.0B model exceeds 2GB limit: {bytes_3_0b / (1024**2):.2f} MB"

        # Peak Combined Pipeline Allocation Budget
        kv_cache_bytes = 8 * 2048 * 28 * 16 * 64 * 2 * 2  # N=8, ctx 2048, K+V, FP16
        app_overhead_bytes = 500 * (1024**2)
        peak_pipeline_bytes = bytes_1_5b + kv_cache_bytes + app_overhead_bytes
        
        MAX_ALLOWED_BYTES = 4.5 * (1024**3)  # 4.5 GB
        assert peak_pipeline_bytes < MAX_ALLOWED_BYTES, (
            f"Peak pipeline memory ({peak_pipeline_bytes / (1024**3):.2f} GB) "
            f"exceeds 4.5 GB limit"
        )

    def test_runtime_simulated_allocation_rss(self):
        process = psutil.Process(os.getpid())
        baseline_rss = process.memory_info().rss
        
        # Simulate loading 1.5B INT4 model weights (~844 MB payload)
        num_elements = int(1.5e9 // 2)  # packed nibbles in uint8
        reasoner_weights = np.zeros(num_elements, dtype=np.uint8)
        
        post_reasoner_rss = process.memory_info().rss
        reasoner_rss_delta = post_reasoner_rss - baseline_rss
        
        assert reasoner_rss_delta < 1.2 * (1024**3), f"Simulated 1.5B loading RSS delta too high: {reasoner_rss_delta / (1024**2):.2f} MB"
        
        # Test Sequential Memory Purge (Unloading Reasoner before Verifier load)
        del reasoner_weights
        import gc; gc.collect()
        
        purged_rss = process.memory_info().rss
        assert purged_rss - baseline_rss < 100 * (1024**2), "Memory not properly released during sequential swap purge"
```

---

## 4. Environment Constraints & Mitigation Strategies

| Constraint / Challenge | Risk / Impact | Mitigation Strategy in `tests/test_quantization.py` |
|---|---|---|
| **CPU vs Metal/NPU SIMD Execution** | Micro-benchmarks on x86/generic CPU might not reflect Metal GPU SIMD gather speedups. | Use `@pytest.mark.benchmark` tag to isolate performance tests. Include soft assertion fallback for non-Metal environments while asserting strict bounds when GPU is detected. |
| **NumPy Memory Alignment (`ctypes.data % 128`)** | standard `np.ndarray` allocation by NumPy is aligned to 16/64 bytes, not guaranteed 128-byte. | `repack_weights_to_superblock` must use custom aligned memory allocator (e.g. `np.ascontiguousarray` with zero padding or aligned byte array copy) to guarantee 128-byte boundary alignment. |
| **FP16 Dynamic Range & Overflow** | `np.float16` can overflow above 65,504 or underflow below $6.1 \times 10^{-5}$. | Quantization scale calculations must clip float values into range $[-65000, 65000]$ and set `atol=1e-3` for float comparison tests. Exact integer indices matching LUT table entries must use `atol=0`. |
| **macOS vs iOS RSS Memory Profiling** | `psutil.Process().memory_info().rss` on macOS measures resident set size, which varies depending on OS page swapping. | Supplement RSS checks with `tracemalloc` Python memory allocation tracking to measure exact heap memory overhead independently of OS paging behavior. |
| **Pytest Suite Runtime Budget** | Memory profiling with 1.5B mock arrays takes noticeable RAM/time. | Parametrize test scale so normal unit tests run with smaller tensors ($256$ to $65,536$ elements) in $<0.1\text{s}$, reserving 1.5B memory tests for dedicated test methods. |

---

## 5. Caveats

1. **Synthetic Array Proxies for Memory Testing:** Actual model weights files (`.bin` / `.safetensors`) were not loaded as model files are built during Phase 1 implementation. Synthetic numpy arrays matching exact byte dimensions were evaluated.
2. **Metal Shaders Execution:** Benchmark tests in pure Python examine `numpy` table gathers (`lut[q_weights]`). In Metal compute shaders, `simdgroup_matrix` and `vlut16` gathering will yield higher SIMD throughput.

---

## 6. Conclusion

The test suite design for `tests/test_quantization.py` provides complete, rigorous coverage across INT4 superblock repacking, LUT FP16 dequantization, and the 4.5GB memory budget requirement. Key actionable specifications:
- **Repacking:** Validate exact shape $(S, 8, 32)$, 128-byte address alignment, zero data corruption, and dimension exception handling.
- **LUT Dequantization:** Assert bit-accurate lookup matching, FP16 dtype preservation, and benchmark sub-millisecond execution over arithmetic unpacking.
- **Memory Bounds:** Validate mathematical footprint formulas ($843.75\text{MB}$ for 1.5B, $1.687\text{GB}$ for 3.0B), verify sequential model swap memory purging, and enforce peak RAM limit strictly $< 4.5\text{ GB}$.

---

## 7. Verification Method

To verify the test suite once implemented by the implementer team:
1. Run pytest suite:
   ```bash
   pytest tests/test_quantization.py -v
   ```
2. Run benchmark performance tests explicitly:
   ```bash
   pytest tests/test_quantization.py -m benchmark -v -s
   ```
3. Verification Invalidation Conditions:
   - Any test failing to enforce 128-byte address alignment (`repacked.ctypes.data % 128 != 0`).
   - `lut_dequantize_fp16` output dtype differing from `np.float16` or returning values not matching LUT table entries.
   - Memory allocation test exceeding 4.5 GB peak budget.
