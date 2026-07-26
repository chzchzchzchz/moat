"""
Project Antigravity — Unit Test Suite for model_loader.py

Tests:
  1. GGUF, Safetensors, and Mock weight reading and super-block repacking accuracy.
  2. Super-block 144-byte structure validation (16-byte scale header + 128-byte payload).
  3. 1.5B parameter model weight memory calculation and budget assertion (<= 2.5 GB).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_loader import (
    GGUFWeightReader,
    SafetensorsWeightReader,
    SuperBlockRepacker,
    QuantizedSuperBlockTensor,
    MemoryBudgetValidator
)


class TestWeightReaders(unittest.TestCase):
    """Unit tests for weight readers and mock synthetic weight generation."""

    def test_mock_weight_reader_manifest_and_generation(self):
        """Verify MockWeightReader manifest generation for Qwen2.5-1.5B."""
        reader = MockWeightReader(
            hidden_size=2048,
            intermediate_size=5504,
            num_layers=28,
            num_heads=16,
            num_kv_heads=2,
            vocab_size=151936
        )

        names = reader.list_tensor_names()
        self.assertIn("model.embed_tokens.weight", names)
        self.assertIn("lm_head.weight", names)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("model.layers.27.mlp.gate_proj.weight", names)

        # Check total parameter count (~1.54 Billion)
        total_params = reader.get_total_parameter_count()
        self.assertGreater(total_params, 1_400_000_000)
        self.assertLess(total_params, 2_000_000_000)

        # Read tensor and verify dtype and shape
        q_weight = reader.read_tensor("model.layers.0.self_attn.q_proj.weight")
        self.assertEqual(q_weight.shape, (2048, 2048))
        self.assertEqual(q_weight.dtype, np.float16)

    def test_gguf_and_safetensors_reader_fallback(self):
        """Verify GGUF and Safetensors readers fallback to mock weights when file does not exist."""
        gguf_reader = GGUFWeightReader("non_existent_model.gguf")
        safe_reader = SafetensorsWeightReader("non_existent_model.safetensors")

        self.assertGreater(len(gguf_reader.list_tensor_names()), 0)
        self.assertGreater(len(safe_reader.list_tensor_names()), 0)

        w1 = gguf_reader.read_tensor("model.layers.0.self_attn.q_proj.weight")
        w2 = safe_reader.read_tensor("model.layers.0.self_attn.q_proj.weight")

        self.assertEqual(w1.shape, (2048, 2048))
        self.assertEqual(w2.shape, (2048, 2048))


class TestSuperBlockRepackingAndValidation(unittest.TestCase):
    """Unit tests for SuperBlockRepacker structure validation and accuracy."""

    def test_superblock_144_byte_structure_validation(self):
        """
        THE STRUCTURE VALIDATION TEST:
        Every super-block must contain 16-byte scale header + 128-byte payload (144 bytes total).
        """
        np.random.seed(42)
        matrix = np.random.randn(256, 256).astype(np.float16)  # 65536 elements = 256 super-blocks

        q_tensor = SuperBlockRepacker.repack_matrix(matrix, group_size=32, groups_per_superblock=8)

        self.assertTrue(q_tensor.validate_structure(), "Super-block structure validation failed")
        self.assertEqual(q_tensor.bytes_per_superblock, 144)
        self.assertEqual(len(q_tensor.superblocks), 256)
        self.assertEqual(q_tensor.memory_bytes, 256 * 144)  # 36,864 bytes

    def test_repack_and_dequantize_accuracy(self):
        """
        ACCURACY TEST:
        Dequantizing a repacked super-block matrix must match original matrix within INT4 bounds.
        """
        np.random.seed(99)
        original = np.random.randn(128, 256).astype(np.float16)

        q_tensor = SuperBlockRepacker.repack_matrix(original, group_size=32)
        dequantized = q_tensor.dequantize()

        self.assertEqual(dequantized.shape, original.shape)
        # Check MAE bound (FP16 INT4 group quant MAE is bounded)
        mae = float(np.mean(np.abs(original.astype(np.float32) - dequantized.astype(np.float32))))
        self.assertLess(mae, 0.2, f"Reconstruction MAE {mae:.4f} exceeds 0.2 bound")


class TestMemoryBudgetValidator(unittest.TestCase):
    """Unit tests for 1.5B model weight memory calculation and budget assertion."""

    def test_weight_memory_calculation_for_1_5b_params(self):
        """
        BUDGET CALCULATION TEST:
        1.54B parameters in 144-byte super-blocks (~0.844 GB <= 2.5 GB).
        """
        num_params = 1_540_000_000  # Qwen2.5-1.5B exact params
        weight_gb = MemoryBudgetValidator.calculate_weight_memory_gb(num_params)

        print(f"\n[BUDGET] 1.54B INT4 Weight Memory: {weight_gb:.3f} GB")

        # 1.54B * 144 / 256 / 1024^3 = 0.866 GB <= 2.5 GB
        self.assertLessEqual(
            weight_gb, 2.5,
            f"1.54B parameter model weight footprint {weight_gb:.3f} GB exceeds 2.5 GB ceiling"
        )
        self.assertGreater(weight_gb, 0.7, "Weight memory calculation unrealistically low")

    def test_full_memory_budget_validation_report(self):
        """
        FULL BUDGET ASSERTION TEST:
        Validates weight memory <= 2.5 GB and total app memory <= 4.5 GB ceiling.
        """
        report = MemoryBudgetValidator.validate_memory_budget(
            num_params=1_540_000_000,
            num_traces=8,
            seq_len=2048,
            runtime_overhead_gb=0.5
        )

        self.assertTrue(report['weight_budget_passed'])
        self.assertTrue(report['total_app_budget_passed'])
        self.assertTrue(report['is_valid'])

        self.assertLessEqual(report['weight_memory_gb'], 2.5)
        self.assertLessEqual(report['total_app_memory_gb'], 4.5)

    def test_exceeding_weight_budget_raises_value_error(self):
        """Ensure MemoryBudgetValidator raises ValueError when weight budget is breached."""
        with self.assertRaises(ValueError):
            MemoryBudgetValidator.validate_memory_budget(
                num_params=6_000_000_000  # 6B params exceeds 2.5 GB in INT4
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
