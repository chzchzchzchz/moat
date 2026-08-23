"""
Project Antigravity — On-Device Real Semantic Embeddings (DORA Clustering)

This module implements:
  - DORAClusterer: Dynamic Optimal Resource Allocation Clusterer.
    Computes real text/token vector embeddings, pairwise cosine similarity matrices S_ij,
    and dynamic uniqueness weights gamma_i = 1 / sum_j(S_ij) to penalize redundant rollouts.

Target Hardware: Apple Silicon GPU / iOS (A17 Pro / A18 Pro / M1-M4)
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional


class DORAClusterer:
    """
    On-Device Real DORA Semantic Clusterer.

    Computes real semantic vector embeddings (subword/character n-gram frequency vectors),
    cosine similarity matrices, and uniqueness weights across active candidate channels.
    """

    def __init__(self, ngram_range: Tuple[int, int] = (2, 4), vector_dim: int = 512):
        self.ngram_range = ngram_range
        self.vector_dim = vector_dim

    def compute_embedding(self, text_or_tokens) -> np.ndarray:
        """
        Compute a dense vector embedding v in R^D for a text string or list of token IDs.
        Uses token n-gram hashing for robust semantic embedding.
        """
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        import zlib

        if isinstance(text_or_tokens, (list, tuple, np.ndarray)):
            # Token-level n-grams
            tokens = list(text_or_tokens)
            if not tokens:
                return vec
            
            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    idx = zlib.crc32(str(ngram).encode('utf-8')) % self.vector_dim
                    vec[idx] += 1.0 / n
        else:
            text = str(text_or_tokens).lower().strip()
            if not text:
                return vec
            words = text.split()
            for w in words:
                idx = zlib.crc32(f"w:{w}".encode('utf-8')) % self.vector_dim
                vec[idx] += 3.0
            for i in range(len(words) - 1):
                w_pair = f"w2:{words[i]}_{words[i+1]}"
                idx = zlib.crc32(w_pair.encode('utf-8')) % self.vector_dim
                vec[idx] += 2.0
            min_n, max_n = self.ngram_range
            for n in range(min_n, max_n + 1):
                for i in range(len(text) - n + 1):
                    ngram = text[i:i + n]
                    idx = zlib.crc32(ngram.encode('utf-8')) % self.vector_dim
                    vec[idx] += 0.5

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm

        return vec

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute N x N pairwise cosine similarity matrix S_ij.
        """
        N = len(embeddings)
        if N == 0:
            return np.zeros((0, 0), dtype=np.float32)

        # Matrix multiplication of L2-normalized embeddings gives cosine similarity
        sim_matrix = np.dot(embeddings, embeddings.T)
        # Clip numerical inaccuracies to [0, 1]
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)

        # Ensure diagonal self-similarity is exactly 1.0
        np.fill_diagonal(sim_matrix, 1.0)
        return sim_matrix

    def compute_uniqueness_weights(self, sim_matrix: np.ndarray, tau_d: float = 0.5) -> np.ndarray:
        """
        Compute uniqueness weight vector gamma_i = 1 / sum_j(exp(S_ij / tau_d)).
        Higher uniqueness score means candidate i is distinct from other candidates.
        """
        N = len(sim_matrix)
        if N == 0:
            return np.array([], dtype=np.float32)

        # Exponential scaling with temperature calibration
        density = np.sum(np.exp(sim_matrix / tau_d), axis=1)
        density = np.maximum(density, 1e-5)
        gamma = 1.0 / density

        # Normalize gamma to sum to N (so baseline mean weight is 1.0)
        gamma = (gamma / np.sum(gamma)) * float(N)
        return gamma

    def cluster_candidates(self, candidate_traces: List[str]) -> Dict:
        """
        Perform complete DORA semantic clustering over candidate traces.

        Returns:
            Dict containing:
                'embeddings': ndarray of shape (N, D)
                'similarity_matrix': ndarray of shape (N, N)
                'uniqueness_weights': ndarray of shape (N,)
                'redundancy_pairs': List of pairs (i, j) with similarity > 0.85
        """
        N = len(candidate_traces)
        if N == 0:
            return {
                'embeddings': np.zeros((0, self.vector_dim)),
                'similarity_matrix': np.zeros((0, 0)),
                'uniqueness_weights': np.array([]),
                'redundancy_pairs': []
            }

        embeddings = np.zeros((N, self.vector_dim), dtype=np.float32)
        for i in range(N):
            embeddings[i] = self.compute_embedding(candidate_traces[i])

        sim_matrix = self.compute_similarity_matrix(embeddings)
        gamma = self.compute_uniqueness_weights(sim_matrix)

        # Identify highly redundant pairs (similarity > 0.85)
        redundant_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if sim_matrix[i, j] > 0.85:
                    redundant_pairs.append((i, j, float(sim_matrix[i, j])))

        return {
            'embeddings': embeddings,
            'similarity_matrix': sim_matrix,
            'uniqueness_weights': gamma,
            'redundancy_pairs': redundant_pairs
        }
