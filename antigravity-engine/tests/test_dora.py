import sys
import os
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dora_clustering import DORAClusterer


def test_compute_embedding():
    clusterer = DORAClusterer(vector_dim=128)
    vec1 = clusterer.compute_embedding("Prove that 2^n > n^2 for n >= 5")
    vec2 = clusterer.compute_embedding("Prove that 2^n > n^2 for n >= 5")

    assert vec1.shape == (128,)
    assert np.allclose(vec1, vec2)
    assert np.isclose(np.linalg.norm(vec1), 1.0)


def test_similarity_matrix_identical():
    clusterer = DORAClusterer(vector_dim=128)
    text = "Mathematical induction step 1"
    res = clusterer.cluster_candidates([text, text])

    sim = res['similarity_matrix']
    assert sim.shape == (2, 2)
    assert np.isclose(sim[0, 1], 1.0)
    assert np.isclose(sim[1, 0], 1.0)


def test_similarity_matrix_distinct():
    clusterer = DORAClusterer(vector_dim=128)
    t1 = "Mathematical induction proof step 1"
    t2 = "Quantum physics wave function Schrödinger equation"
    res = clusterer.cluster_candidates([t1, t2])

    sim = res['similarity_matrix']
    assert sim[0, 1] < 0.5


def test_uniqueness_weights():
    clusterer = DORAClusterer(vector_dim=128)
    # 2 identical traces + 1 unique trace
    t1 = "Let n = 5. Then 2^5 = 32."
    t2 = "Let n = 5. Then 2^5 = 32."
    t3 = "Base case verification for integer n=5."

    res = clusterer.cluster_candidates([t1, t2, t3])
    weights = res['uniqueness_weights']

    assert len(weights) == 3
    # Unique trace t3 should receive higher uniqueness weight than redundant t1/t2
    assert weights[2] > weights[0]
