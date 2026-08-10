"""
Project Antigravity — PRM-Guided Verification Strategies

Implements candidate selection strategies for Best-of-N test-time scaling:
  1. BestOfNStrategy:  Select the candidate with highest cumulative PRM reward.
  2. RebaseStrategy:   REBASE — allocate rollout budget proportional to per-step
                       PRM rewards, then select the highest-scoring candidate.

Reference: "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than
           Scaling Model Parameters" (Snell et al., 2024)

Target Hardware: Apple Silicon GPU / iOS (A17 Pro / A18 Pro / M1-M4)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class BestOfNStrategy:
    """
    Vanilla Best-of-N: Select the candidate with the highest total PRM reward.

    Given N candidate reasoning traces scored by a Process Reward Model,
    returns the index of the candidate with the highest cumulative reward.
    Optionally blends in log-probability density for tie-breaking.
    """

    def __init__(self, logprob_weight: float = 0.1, alpha: float = 0.6):
        """
        Args:
            logprob_weight: Weight given to length-normalized logprob in final score.
            alpha: Exponent for length normalization of logprob (higher = less length penalty).
        """
        self.logprob_weight = logprob_weight
        self.alpha = alpha

    def select(
        self,
        prm_scores: np.ndarray,
        logprobs: np.ndarray,
        trace_lengths: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Select the best candidate from PRM scores and logprobs.

        Args:
            prm_scores:    Shape (N,) — cumulative PRM reward per candidate.
            logprobs:      Shape (N,) — cumulative log-probability per candidate.
            trace_lengths: Shape (N,) — token count per trace (for length normalization).

        Returns:
            Dict with 'best_index', 'best_score', 'scores', 'rankings'.
        """
        N = len(prm_scores)
        if trace_lengths is None:
            trace_lengths = np.ones(N, dtype=np.float32)

        # Length-normalized logprob density
        norm_logprobs = logprobs / np.maximum(trace_lengths.astype(np.float32) ** self.alpha, 1.0)

        # Combined score: PRM reward + weighted logprob density
        combined = prm_scores.astype(np.float32) + self.logprob_weight * norm_logprobs

        rankings = np.argsort(combined)[::-1]
        best_idx = int(rankings[0])

        return {
            'best_index': best_idx,
            'best_score': float(combined[best_idx]),
            'scores': combined,
            'rankings': rankings,
            'strategy': 'best_of_n',
        }


class RebaseStrategy:
    """
    REBASE: Reward-Balanced Search for test-time compute allocation.

    Instead of uniformly allocating the rollout token budget across N channels,
    REBASE redistributes budget proportional to intermediate PRM step rewards.
    Candidates with higher step-level rewards receive more generation budget.

    This implements a simplified version of the REBASE algorithm:
      1. After initial generation, score each candidate's intermediate steps.
      2. Compute per-candidate allocation weights via softmax over mean step rewards.
      3. Return recommended token budgets for a potential second-pass generation.
      4. Select the highest-scoring candidate from the weighted final scores.
    """

    def __init__(self, temperature: float = 1.0, logprob_weight: float = 0.1):
        """
        Args:
            temperature: Softmax temperature for budget allocation (lower = more greedy).
            logprob_weight: Weight for logprob blending in final selection.
        """
        self.temperature = temperature
        self.logprob_weight = logprob_weight

    def compute_allocation_weights(
        self,
        step_rewards: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute per-candidate allocation weights from step-level PRM rewards.

        Args:
            step_rewards: List of N arrays, each shape (n_steps_i,) with per-step rewards.

        Returns:
            Shape (N,) allocation weights that sum to 1.0.
        """
        N = len(step_rewards)
        mean_rewards = np.zeros(N, dtype=np.float32)

        for i, rewards in enumerate(step_rewards):
            if len(rewards) > 0:
                mean_rewards[i] = float(np.mean(rewards))
            else:
                mean_rewards[i] = -1e6  # heavily penalize empty traces

        # Softmax with temperature
        shifted = mean_rewards - np.max(mean_rewards)
        exp_vals = np.exp(shifted / max(self.temperature, 1e-6))
        weights = exp_vals / np.sum(exp_vals)

        return weights

    def allocate_budget(
        self,
        step_rewards: List[np.ndarray],
        total_budget: int
    ) -> np.ndarray:
        """
        Allocate a token generation budget across N candidates.

        Args:
            step_rewards: List of N step-reward arrays.
            total_budget: Total tokens to distribute (e.g., 256 * N).

        Returns:
            Shape (N,) integer array of per-candidate token budgets.
        """
        weights = self.compute_allocation_weights(step_rewards)
        raw_budgets = weights * total_budget

        # Round to integers while preserving total
        budgets = np.floor(raw_budgets).astype(np.int32)
        remainder = total_budget - int(np.sum(budgets))

        # Distribute remainder to highest-weighted candidates
        if remainder > 0:
            frac_parts = raw_budgets - budgets.astype(np.float32)
            top_indices = np.argsort(frac_parts)[::-1][:remainder]
            for idx in top_indices:
                budgets[idx] += 1

        # Ensure minimum budget of 1 per channel
        budgets = np.maximum(budgets, 1)

        return budgets

    def select(
        self,
        prm_scores: np.ndarray,
        logprobs: np.ndarray,
        step_rewards: Optional[List[np.ndarray]] = None,
        trace_lengths: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Select the best candidate using REBASE-weighted scoring.

        Args:
            prm_scores:    Shape (N,) — cumulative PRM reward per candidate.
            logprobs:      Shape (N,) — cumulative log-probability per candidate.
            step_rewards:  Optional list of per-step reward arrays (for allocation weights).
            trace_lengths: Shape (N,) — token count per trace.

        Returns:
            Dict with 'best_index', 'best_score', 'scores', 'rankings',
            'allocation_weights', 'strategy'.
        """
        N = len(prm_scores)
        if trace_lengths is None:
            trace_lengths = np.ones(N, dtype=np.float32)

        # Compute allocation weights if step rewards available
        if step_rewards is not None:
            alloc_weights = self.compute_allocation_weights(step_rewards)
        else:
            alloc_weights = np.ones(N, dtype=np.float32) / N

        # Length-normalized logprob
        norm_logprobs = logprobs / np.maximum(trace_lengths.astype(np.float32) ** 0.6, 1.0)

        # REBASE-weighted score: allocation weight * PRM + logprob density
        combined = alloc_weights * prm_scores.astype(np.float32) + self.logprob_weight * norm_logprobs

        rankings = np.argsort(combined)[::-1]
        best_idx = int(rankings[0])

        return {
            'best_index': best_idx,
            'best_score': float(combined[best_idx]),
            'scores': combined,
            'rankings': rankings,
            'allocation_weights': alloc_weights,
            'strategy': 'rebase',
        }
