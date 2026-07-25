import os
import time
import yaml
import pytest
import numpy as np

from src.dequant import (
    quantize_fp16_to_int4,
    repack_weights_to_superblock,
    lut_dequantize_fp16,
    unpack_superblock_weights,
    calculate_memory_footprint,
    ensure_aligned_array,
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


def test_quantize_fp16_to_int4_correctness():
    np.random.seed(42)
    weights_fp16 = np.random.uniform(-2.0, 2.0, size=1024).astype(np.float16)
    
    q_weights, scales = quantize_fp16_to_int4(weights_fp16, group_size=32)
    
    assert q_weights.shape == (1024,)
    assert scales.shape == (32,)
    assert q_weights.dtype == np.int8
    assert scales.dtype == np.float16
    assert np.min(q_weights) >= -8
    assert np.max(q_weights) <= 7


def test_quantize_and_repack_superblock():
    np.random.seed(42)
    weights_fp16 = np.random.uniform(-2.0, 2.0, size=1024).astype(np.float16)
    
    q_weights, scales = quantize_fp16_to_int4(weights_fp16, group_size=32)
    repacked = repack_weights_to_superblock(q_weights, group_size=32)
    
    assert repacked.shape == (4, 8, 32)
    assert repacked.flags['C_CONTIGUOUS']
    assert repacked.ctypes.data % 128 == 0, f"Memory address {hex(repacked.ctypes.data)} is not 128-byte aligned"
    
    # Value preservation check
    np.testing.assert_array_equal(repacked.reshape(-1), q_weights)


def test_repack_alignment_assertion():
    invalid_weights = np.zeros(200, dtype=np.int8)
    with pytest.raises(AssertionError):
        repack_weights_to_superblock(invalid_weights, group_size=32)


def test_ensure_aligned_array_utility():
    arr = np.random.randint(-8, 8, size=256, dtype=np.int8)
    aligned_arr = ensure_aligned_array(arr, alignment=128)
    
    assert aligned_arr.ctypes.data % 128 == 0
    assert aligned_arr.flags['C_CONTIGUOUS']
    np.testing.assert_array_equal(aligned_arr, arr)


def test_lut_dequantization_correctness_and_precision():
    np.random.seed(42)
    weights_fp16 = np.random.uniform(-1.0, 1.0, size=256).astype(np.float16)
    q_weights, scales = quantize_fp16_to_int4(weights_fp16, group_size=32)
    
    k_range = np.arange(-8, 8, dtype=np.float32)
    lut_2d = scales[:, None] * k_range[None, :]  # shape (8, 16)
    
    dequant = lut_dequantize_fp16(q_weights.reshape(8, 32), lut_2d)
    
    assert dequant.shape == (8, 32)
    assert dequant.dtype == np.float16
    
    # Verification against original FP16 weights: low error tolerance
    max_error = np.max(np.abs(weights_fp16.astype(np.float32) - dequant.astype(np.float32).flatten()))
    assert max_error < 0.2, f"Dequantization error too high: {max_error}"


def test_lut_dequantization_signed_index_offset_mapping():
    # Verify index mapping [-8, 7] -> [0, 15] works correctly without reverse indexing wrapping
    q_weights = np.array([-8, -7, 0, 7], dtype=np.int8)
    lut_1d = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=np.float16)
    
    dequant = lut_dequantize_fp16(q_weights, lut_1d)
    
    # Index -8 + 8 = 0 -> lut_1d[0] = 10.0
    # Index -7 + 8 = 1 -> lut_1d[1] = 11.0
    # Index 0 + 8 = 8  -> lut_1d[8] = 18.0
    # Index 7 + 8 = 15 -> lut_1d[15] = 25.0
    expected = np.array([10.0, 11.0, 18.0, 25.0], dtype=np.float16)
    np.testing.assert_array_equal(dequant, expected)


def test_lut_dequantize_speed_benchmark():
    np.random.seed(42)
    num_params = 1024 * 1024  # 1M params benchmark
    q_weights = np.random.randint(-8, 8, size=num_params, dtype=np.int8)
    scale = np.float32(0.125)
    lut_1d = np.arange(-8, 8, dtype=np.float32) * scale
    
    # Warmup
    for _ in range(5):
        _ = lut_dequantize_fp16(q_weights, lut_1d)
        _ = (q_weights.astype(np.float32) * scale).astype(np.float16)
        
    t0 = time.perf_counter_ns()
    for _ in range(20):
        dequant_lut = lut_dequantize_fp16(q_weights, lut_1d)
    t_lut = (time.perf_counter_ns() - t0) / 20
    
    t0 = time.perf_counter_ns()
    for _ in range(20):
        dequant_arithmetic = (q_weights.astype(np.float32) * scale).astype(np.float16)
    t_arithmetic = (time.perf_counter_ns() - t0) / 20
    
    assert dequant_lut.dtype == np.float16
    # Fast table gather should run smoothly
    assert t_lut <= t_arithmetic * 1.5, f"LUT gather ({t_lut/1e6:.2f}ms) slower than expected vs arithmetic ({t_arithmetic/1e6:.2f}ms)"


def test_memory_bounds_verification():
    fp_1_5b = calculate_memory_footprint(1_500_000_000)
    assert fp_1_5b["fits_in_ram"]
    assert fp_1_5b["total_gb"] < 1.0, f"1.5B footprint unexpectedly large: {fp_1_5b['total_gb']} GB"
    assert pytest.approx(fp_1_5b["bytes_per_param_effective"], 0.001) == 0.5625
    
    fp_3_0b = calculate_memory_footprint(3_000_000_000)
    assert fp_3_0b["fits_in_ram"]
    assert fp_3_0b["total_gb"] < 2.0, f"3.0B footprint unexpectedly large: {fp_3_0b['total_gb']} GB"
    
    fp_7_0b = calculate_memory_footprint(7_000_000_000)
    assert fp_7_0b["fits_in_ram"]
    assert fp_7_0b["total_gb"] < 4.0, f"7.0B footprint unexpectedly large: {fp_7_0b['total_gb']} GB"


def test_pipeline_combined_memory_budget():
    # Simulate active allocation budget for iPhone 15 Pro (4.5GB RAM ceiling)
    fp_1_5b = calculate_memory_footprint(1_500_000_000)
    
    weight_bytes = fp_1_5b["total_bytes"]
    # Batched KV cache (N=8 rollouts, context length 2048, 28 layers, 2 GQA heads, dim 128)
    kv_cache_bytes = 8 * 2048 * 28 * 2 * 128 * 2 * 2  # ~366.93 MB
    scratchpad_bytes = 200 * 1024 * 1024  # 200 MB scratchpad
    
    total_pipeline_bytes = weight_bytes + kv_cache_bytes + scratchpad_bytes
    total_pipeline_gb = total_pipeline_bytes / (1024 ** 3)
    
    RAM_LIMIT_GB = 4.5
    assert total_pipeline_gb < RAM_LIMIT_GB, (
        f"Peak pipeline memory budget ({total_pipeline_gb:.2f} GB) exceeds {RAM_LIMIT_GB} GB ceiling."
    )
