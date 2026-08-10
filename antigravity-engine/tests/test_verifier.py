"""
Project Antigravity — Micro-Unit Test Suite for verifier.py

Tests the ListWiseVerifier, AdaptiveReflectionManager, and SequentialModelSwapper
in COMPLETE ISOLATION with strict assertions.

Test hierarchy:
  1. ListWiseVerifier (step extraction, list-wise scoring, best candidate selection)
  2. AdaptiveReflectionManager (threshold trigger logic, token savings calculation)
  3. SequentialModelSwapper (model state transitions, RAM isolation)
  4. Fuzz test for zero NaN / Inf across random candidate inputs
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from verifier import ListWiseVerifier, AdaptiveReflectionManager, SequentialModelSwapper, DEFAULT_REFLECTION_THRESHOLD


class TestListWiseVerifier(unittest.TestCase):
    """Isolated unit tests for ListWiseVerifier."""

    def setUp(self):
        self.verifier = ListWiseVerifier()

    def test_extract_reasoning_steps(self):
        """Extract step-by-step reasoning steps from trace text."""
        trace = "<think>\nStep 1: Calculate 2^5 = 32.\nStep 2: Calculate 5^2 = 25.\nTherefore 32 > 25.\n</think>"
        steps = self.verifier.extract_reasoning_steps(trace)

        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0], "Step 1: Calculate 2^5 = 32.")
        self.assertEqual(steps[1], "Step 2: Calculate 5^2 = 25.")

    def test_score_candidates_listwise_output_structure(self):
        """score_candidates_listwise must return valid probabilities and rankings."""
        candidates = [
            "Step 1: 2^5 = 32. Therefore 32 > 25.",
            "Step 1: 2^5 = 30.",
            "Step 1: 2^5 = 32. Step 2: 5^2 = 25. Therefore 32 > 25.",
            "Step 1: Wrong.",
        ]
        logprobs = np.array([-1.2, -5.0, -0.8, -10.0], dtype=np.float32)

        res = self.verifier.score_candidates_listwise(candidates, logprobs)

        self.assertEqual(len(res['scores']), 4)
        # Scores must sum to ~1.0 (softmax distribution over candidates)
        self.assertAlmostEqual(float(np.sum(res['scores'])), 1.0, places=2)
        # Best candidate should be a valid index with score > 0
        self.assertIn(res['best_index'], [0, 1, 2, 3])
        self.assertGreater(res['best_score'], 0.0)

    def test_scores_are_non_negative_and_no_nan(self):
        """Candidate scores must be non-negative and free of NaN/Inf."""
        candidates = [f"Step {i}: Candidate answer {i}." for i in range(8)]
        logprobs = -np.random.rand(8).astype(np.float32) * 5.0

        res = self.verifier.score_candidates_listwise(candidates, logprobs)

        self.assertFalse(np.any(np.isnan(res['scores'])), "NaN in candidate scores")
        self.assertFalse(np.any(np.isinf(res['scores'])), "Inf in candidate scores")
        self.assertTrue(np.all(res['scores'] >= 0), "Negative probability detected")


class TestAdaptiveReflectionManager(unittest.TestCase):
    """Isolated unit tests for AdaptiveReflectionManager."""

    def test_threshold_trigger_logic(self):
        """Scores < tau trigger reflection (True), scores >= tau accept (False)."""
        manager = AdaptiveReflectionManager(threshold=0.75)

        # High quality score (0.85 >= 0.75) -> Accept (False)
        self.assertFalse(manager.evaluate_reflection_trigger(0.85))

        # Low quality score (0.50 < 0.75) -> Trigger Reflection (True)
        self.assertTrue(manager.evaluate_reflection_trigger(0.50))

        # Exactly at threshold (0.75 >= 0.75) -> Accept (False)
        self.assertFalse(manager.evaluate_reflection_trigger(0.75))

    def test_token_savings_calculation(self):
        """Verify token savings percentage calculation across multiple queries."""
        manager = AdaptiveReflectionManager(threshold=0.75)

        # 7 high quality queries (accepted without reflection)
        for _ in range(7):
            manager.evaluate_reflection_trigger(0.80)

        # 3 low quality queries (triggered reflection)
        for _ in range(3):
            manager.evaluate_reflection_trigger(0.40)

        # 7 out of 10 skipped reflection -> 70% token savings
        self.assertEqual(manager.total_queries, 10)
        self.assertEqual(manager.reflection_triggered_count, 3)
        self.assertAlmostEqual(manager.token_savings_percentage, 70.0, places=1)


class TestSequentialModelSwapper(unittest.TestCase):
    """Isolated unit tests for SequentialModelSwapper."""

    def test_swapper_state_transitions(self):
        """Verify model swapping between reasoner and verifier."""
        swapper = SequentialModelSwapper()
        self.assertIsNone(swapper.currently_loaded_model)

        # Load Reasoner
        m1 = swapper.swap_to_reasoner()
        self.assertEqual(m1, "reasoner")
        self.assertEqual(swapper.currently_loaded_model, "reasoner")

        # Swap to Verifier
        m2 = swapper.swap_to_verifier()
        self.assertEqual(m2, "verifier")
        self.assertEqual(swapper.currently_loaded_model, "verifier")

        # Unload all
        swapper.unload_all()
        self.assertIsNone(swapper.currently_loaded_model)


class TestVerifierFuzzing(unittest.TestCase):
    """Fuzz testing for verifier under random candidate lists."""

    def test_fuzz_random_candidate_lists(self):
        """Fuzz test across 50 random inputs — zero NaN / Inf ever."""
        verifier = ListWiseVerifier()
        for seed in range(50):
            np.random.seed(seed)
            N = np.random.randint(2, 16)
            candidates = [f"Trace {i} step {np.random.randint(1, 5)}" for i in range(N)]
            logprobs = -np.random.rand(N).astype(np.float32) * 10.0

            res = verifier.score_candidates_listwise(candidates, logprobs)

            self.assertFalse(np.any(np.isnan(res['scores'])), f"NaN at seed={seed}")
            self.assertFalse(np.any(np.isinf(res['scores'])), f"Inf at seed={seed}")
            self.assertEqual(len(res['scores']), N)


if __name__ == '__main__':
    print("=" * 70)
    print("Project Antigravity — Phase 4 Micro-Unit Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)
