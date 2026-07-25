# Handoff Report: R2 Model Weight Loader & Super-Block Repacker Architecture

**Working Directory**: `/Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_p3_2`  
**Target Module**: `antigravity-engine/src/model_loader.py`  
**Target Test File**: `antigravity-engine/tests/test_model_loader.py`  
**Date**: July 25, 2026  

---

## 1. Observation

Direct observations from codebase inspection:
- **`antigravity-engine/src/dequant.py`**:
  - `quantize_weights_int4` (lines 22-72): INT4 symmetric group quantization ($g=32$). Computes scale $S_G = \alpha / 7$ and clamps values to $[-8, 7]$.
  - `repack_to_superblocks` (lines 79-142): Coalesces 8 groups (256 elements) into 1 super-block containing 8 FP16 scale factors (16 bytes header) + 128 uint8 packed nibbles (128 bytes payload) = 144 bytes total per super-block.
  - `unpack_superblock` (lines 145-167): Unpacks 128 `uint8` nibble pairs back to 256 `int8` values in $[-8, 7]$.
  - `build_dequant_lut` (lines 173-191) & `lut_dequantize` (lines 193-230): Precomputes 16-element FP16 lookup tables for fast vector-gather dequantization.
- **`planning/01_hardware_architecture.md`**:
  - Section 5 (lines 66-74): Specifies 8GB UMA memory limits on iPhone 15 Pro / A17 Pro. Available App RAM ceiling is ~4.5 GB to 5.5 GB (with entitlement). Model weight limit is capped at 2.5 GB.
- **`planning/02_quantization_and_lut.md`**:
  - Section 2.2 (lines 21-33): Super-block structure specifications: $256 \text{ elements} \times 4 \text{ bits} = 128 \text{ bytes}$ payload (1 SIMD vector cache line) + 16 bytes header = 144 bytes total.
- **`config/engine_config.yaml`**:
  - Quantization params: `bits: 4`, `group_size: 32`, `superblock_size: 256`, `alignment_bytes: 128`. Execution params: `batch_size: 8`.

---

## 2. Logic Chain

1. **Quantization & Super-Block Math**:
   - `dequant.py` provides the reference implementation for INT4 quantization (`quantize_weights_int4`) and repacking (`repack_to_superblocks`).
   - Group size $g = 32$, 8 groups per super-block $\rightarrow 256$ elements per super-block.
   - Scale header: $8 \times 2 \text{ bytes (FP16)} = 16 \text{ bytes}$. Payload: $256 \times 0.5 \text{ bytes} = 128 \text{ bytes}$. Total super-block size $= 144 \text{ bytes}$.

2. **128-Byte Alignment**:
   - Apple Metal GPU SIMD vector cache lines are 128 bytes wide.
   - Storing packed payloads in contiguous `(N_sb, 128)` `uint8` arrays ensures every 256-element weight payload is aligned to 128 bytes, preventing GPU cache line split penalties during SIMD vector loading.

3. **Weight Memory Calculator**:
   - For a 1.5B parameter model (e.g. Qwen2.5-1.5B):
     $$\text{Super-blocks required} = \left\lceil \frac{1,500,000,000}{256} \right\rceil = 5,859,375$$
     $$\text{Weight Footprint} = 5,859,375 \times 144 \text{ bytes} = 843,750,000 \text{ bytes} \approx 0.84375 \text{ GB}$$
   - Capped budget for weights: $2.5 \text{ GB}$.
   - $0.844 \text{ GB} \le 2.50 \text{ GB}$ $\rightarrow$ **PASS** (utilizes 33.75% of weight budget, leaving 1.656 GB headroom).

4. **Total App Memory Footprint**:
   - Batched Paged KV Cache ($N=8$ channels, $S=2048$ context length, $L=28$ layers, $H_{kv}=2$, $D_{head}=128$):
     $$\text{KV Cache} = 8 \times 28 \times 2048 \times (2 \times 2 \times 128 \times 2) \text{ bytes} = 469,762,048 \text{ bytes} \approx 0.470 \text{ GB}$$
   - Softmax Exponential LUT: $32,768 \times 2 \text{ bytes} = 65,536 \text{ bytes} \approx 64 \text{ KB}$.
   - Metal Command Buffers & System Reserve: $\approx 0.500 \text{ GB}$.
   - Total App Footprint $= 0.844 + 0.470 + 0.000064 + 0.500 = 1.814 \text{ GB}$.
   - Ceiling budget limit: $4.5 \text{ GB}$.
   - $1.814 \text{ GB} \le 4.50 \text{ GB}$ $\rightarrow$ **PASS** (headroom of 2.686 GB).

5. **Weight Reader Design**:
   - Class hierarchy (`BaseWeightReader` $\rightarrow$ `GGUFWeightReader`, `SafetensorsWeightReader`, `MockWeightReader`) enables reading production GGUF/Safetensors files as well as generating synthetic Qwen2.5-1.5B weights for automated test suites without disk dependencies.

---

## 3. Caveats

1. **Non-Divisible Tensor Sizes**:
   - If a custom model tensor has total elements not divisible by 256, zero-padding must be appended to the flat FP16 array prior to super-block repacking. (Note: Qwen2.5-1.5B weight dimensions are all multiples of 256).
2. **Unquantized Vector Exception**:
   - 1D LayerNorm scales and small bias vectors are retained in FP16 precision without super-block repacking to prevent precision loss.
3. **GGUF Alignment Variations**:
   - GGUF files may specify `general.alignment` (usually 32 bytes). The parser must respect data offset alignment when seeking binary buffers.

---

## 4. Conclusion

The technical design and architecture for **R2: Model Weight Loader & Super-Block Repacker** (`src/model_loader.py`) is complete, fully specified, and mathematically verified:
- **Parser abstraction** supporting GGUF, Safetensors, and synthetic Qwen2.5-1.5B mock reader.
- **On-the-fly repacker** converting FP16 weights to 144-byte 256-element super-blocks with 128-byte SIMD vector alignment.
- **Memory budget validator** confirming 1.5B model weight memory is $0.844 \text{ GB}$ ($\le 2.5 \text{ GB}$ limit) and total app footprint is $1.814 \text{ GB}$ ($\le 4.5 \text{ GB}$ ceiling limit).
- Concrete specification ready for implementation by Worker (`teamwork_preview_worker`).

---

## 5. Verification Method

To verify this architecture and its eventual implementation:

1. **Inspect Analysis Artifacts**:
   - Review `/Users/MohssineChazi2/moat/.agents/teamwork_preview_explorer_p3_2/analysis.md` for class specifications and memory math calculations.

2. **Execute Unit Test Suite (Post-Implementation)**:
   ```bash
   cd /Users/MohssineChazi2/moat/antigravity-engine
   python3 -m unittest discover -s tests -p "test_*.py"
   ```
   - Expect 100% pass rate across `test_quantization.py`, `test_attention.py`, and `test_model_loader.py`.

3. **Verify Budget Invalidation Conditions**:
   - Passing $N=128$ rollouts or $10\text{B}+$ parameter weight sizes into `MemoryBudgetValidator` must raise budget violation flags (`is_valid == False`).
