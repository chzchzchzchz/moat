## 2026-07-25T13:18:34Z
Implement the complete Phase 3 software components:
1. /Users/MohssineChazi2/moat/antigravity-engine/src/batch_generator.py
   - Implement PagedKVCache with 16-token physical blocks, virtual-to-physical block tables, shared prompt prefix block allocation across N candidate channels (N=4, 8, 16), reference counting, Copy-on-Write (CoW) block cloning, and get_memory_bytes() verifying <= 128 MB per trace for sequence length 2048.
   - Implement BatchedRolloutCoordinator managing N candidate traces, combining active tokens into [N x D] matrix, executing batched GEMM (using PyTorch MPS backend or Metal compute shaders), multi-channel temperature sampling (T > 0, top-p/top-k), and integration with attention.py (ExponentialLUT and safe_softmax_lut).

2. /Users/MohssineChazi2/moat/antigravity-engine/src/model_loader.py
   - Implement BaseWeightReader -> GGUFWeightReader, SafetensorsWeightReader, MockWeightReader (generating synthetic Qwen2.5-1.5B weights for testing without downloading full binary weights).
   - Implement SuperBlockRepacker / QuantizedSuperBlockTensor converting FP16/FP32 matrices on-the-fly into 256-element super-blocks (144 bytes: 16-byte scale header + 128-byte payload) using dequant.py functions (quantize_weights_int4, repack_to_superblocks, build_dequant_lut, lut_dequantize) with 128-byte cache line alignment.
   - Implement MemoryBudgetValidator confirming model weight footprint is <= 2.5 GB for 1.5B parameters (~0.844 GB) and total app footprint is <= 4.5 GB ceiling.

3. /Users/MohssineChazi2/moat/antigravity-engine/tests/test_batch_generator.py
   - Test single-trace decode (N=1) vs Parallel rollout decode (N=8) mathematical parity (atol <= 1e-2).
   - Test paged KV-cache memory isolation across rollout channels.
   - Test 0 NaN, 0 Inf, and 0 memory leaks across 100+ generation steps.
   - Test coherent non-identical outputs under sampling temperature T > 0.
   - Test performance benchmark: N=8 batched rollout coordinator completes 50 generation steps in <= 1.0s on GPU/MPS.
   - Test KV-cache memory overhead per candidate trace <= 128 MB for seq len 2048.

4. /Users/MohssineChazi2/moat/antigravity-engine/tests/test_model_loader.py
   - Test GGUF/Safetensors/Mock weight reading and super-block repacking accuracy.
   - Test super-block 144-byte structure validation (16-byte scale header + 128-byte payload).
   - Test 1.5B parameter model weight memory calculation and budget assertion (<= 2.5 GB).

5. Verification & Testing:
   - Run unit test suite: python3 -m unittest discover -s antigravity-engine/tests -p "test_*.py"
   - Document build & test execution results.
