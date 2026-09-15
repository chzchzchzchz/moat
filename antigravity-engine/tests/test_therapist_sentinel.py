"""
Test Suite: Zero-Trust Therapist Sentinel & Clinical Memory Engine
Validates on-device database queries, longitudinal trajectory computation,
zero-egress network isolation auditing, and verifier scoring.
"""

import pytest
import sys
import os
import sqlite3
import numpy as np

# Include paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples', 'TherapistAgent')))

from clinical_memory_engine import ClinicalMemoryStore, ZeroEgressNetworkAuditor
from verifier import ListWiseVerifier


def test_clinical_memory_store_initialization_and_seed():
    store = ClinicalMemoryStore(":memory:")
    store.populate_longitudinal_benchmark_patient()
    
    history = store.get_longitudinal_history("PT-8492")
    assert len(history) == 7, f"Expected 7 historical sessions, got {len(history)}"
    
    # Check chronological ordering and score trajectory
    assert history[0]["session"] == 1
    assert history[0]["phq9"] == 18
    assert history[0]["gad7"] == 14
    
    assert history[3]["session"] == 4
    assert history[3]["phq9"] == 12
    assert history[3]["gad7"] == 10
    
    assert history[6]["session"] == 7
    assert history[6]["phq9"] == 15
    assert history[6]["gad7"] == 15


def test_cross_session_historical_retrieval():
    store = ClinicalMemoryStore(":memory:")
    store.populate_longitudinal_benchmark_patient()
    
    # Search for prior exposure therapy interventions
    elevator_sessions = store.search_past_interventions("PT-8492", "elevator")
    assert len(elevator_sessions) >= 1
    assert elevator_sessions[0]["session"] == 4
    assert "elevator" in elevator_sessions[0]["transcript"].lower()

    # Search for Sertraline titration history
    sertraline_sessions = store.search_past_interventions("PT-8492", "Sertraline")
    assert len(sertraline_sessions) >= 2


def test_zero_egress_network_auditor(monkeypatch):
    auditor = ZeroEgressNetworkAuditor()
    monkeypatch.setattr(auditor, "_get_network_bytes", lambda: 100)
    auditor.start_audit()
    
    # Perform local computation
    x = np.random.randn(1000, 1000)
    y = x @ x.T
    assert y.shape == (1000, 1000)
    
    report = auditor.stop_audit()
    assert report["external_egress_bytes"] == 0
    assert report["wan_connections_opened"] == 0
    assert report["hipaa_compliant_airgap"] is True


def test_clinical_verifier_listwise_scoring():
    verifier = ListWiseVerifier()
    
    traces = [
        "S: Patient presents with chest tightness. O: Anxious. A: Panic Disorder F41.0. P: CBT thought records, continue Sertraline.",
        "Patient seems sad and said something about work. Plan: talk more next week.",
        "Notes: Sarah was tearful. We reviewed deep breathing and diaphragmatic pacing. Homework assigned.",
    ]
    logprobs = np.array([-0.05, -0.85, -0.12], dtype=np.float32)
    
    res = verifier.score_candidates_listwise(traces, logprobs)
    assert "scores" in res
    assert "best_index" in res
    assert 0 <= res["best_index"] < len(traces)
    assert res["best_score"] > 0.0
    assert np.isclose(np.sum(res["scores"]), 1.0, atol=1e-3)


def test_real_clinical_metal_rollout_and_verification():
    """
    Rigorously tests real on-device clinical generation and verifier scoring.
    Loads real weights, generates clinical rollouts on Metal GPU, validates
    psychiatric terminology, and records the verified encounter in ClinicalMemoryStore.
    """
    from native_bridge import NativeMetalEngine
    from tokenizer import LlamaTokenizer

    weights_path = "models/tinyllama/model_fp16.safetensors"
    tok_path = "models/tinyllama/tokenizer.json"

    if not os.path.exists(weights_path) or not os.path.exists(tok_path):
        pytest.skip("TinyLlama model weights or tokenizer not present")

    tok = LlamaTokenizer(tok_path)
    engine = NativeMetalEngine(
        dylib_path="libantigravity_engine.dylib",
        model_path=weights_path,
        n_channels=3,
        vocab_size=32000,
        hidden_dim=2048
    )

    try:
        # Prompt simulating clinical psychiatric intake
        prompt = "CLINICAL EVALUATION: Patient presents with persistent chest tightness and elevator panic. Diagnosis:"
        prompt_ids = [tok.bos_token_id] + tok.encode(prompt)

        tokens, logprobs, ttft_ms, total_ms = engine.generate(
            prompt_token_ids=prompt_ids,
            max_new_tokens=25,
            temperature=0.7,
            top_p=0.9
        )

        assert len(tokens) == 3, "Expected 3 parallel clinical rollout channels"
        assert ttft_ms > 0.0, "TTFT must be measured on GPU"
        assert all(lp < 0.0 for lp in logprobs), "Logprobs must be negative log-likelihoods"

        traces = [tok.decode(t).strip() for t in tokens]
        print(f"\n[Clinical Rollout Traces]: {traces}")
        print(f"[Clinical Logprobs]: {logprobs}")

        # Linguistic & Psychiatric Verification: assert genuine clinical terms
        clinical_vocab = [
            "panic", "disorder", "treatment", "cbt", "medication", "agoraphobia", "anxiety", "patient",
            "pulmonary", "embolism", "syndrome", "symptom", "diagnosis", "evaluation", "therapy", "acute"
        ]
        for i, trace in enumerate(traces):
            assert len(trace) > 10, f"Channel {i} trace too short: '{trace}'"
            found_terms = [t for t in clinical_vocab if t in trace.lower()]
            assert len(found_terms) >= 1, f"Channel {i} missing clinical psychiatric vocabulary: '{trace}'"

        # Verifier scoring of live generated candidates
        verifier = ListWiseVerifier()
        verif_res = verifier.score_candidates_listwise(traces, np.array(logprobs, dtype=np.float32))

        assert 0 <= verif_res["best_index"] < len(traces)
        assert verif_res["best_score"] > 0.0
        assert np.isclose(np.sum(verif_res["scores"]), 1.0, atol=1e-3)
        print(f"[Winning Trace]: {verif_res['best_trace']} (Score: {verif_res['best_score']:.4f})")

        # Record verified encounter in local encrypted store
        store = ClinicalMemoryStore(":memory:")
        cur = store.conn.cursor()
        cur.execute("INSERT OR REPLACE INTO patients VALUES (?, ?, ?, ?, ?)",
                    ("PT-LIVE-01", "Live Patient", "1990-01-01", "Panic Disorder", "Dr. Vance"))
        store.conn.commit()

        store.add_session(
            patient_id="PT-LIVE-01",
            session_number=1,
            date="2026-09-06",
            modality="CBT",
            phq9_score=16,
            gad7_score=14,
            medications="Sertraline 50mg",
            raw_transcript=prompt,
            soap_note=f"S: Chest tightness in elevators. O: Diaphoresis. A: {verif_res['best_trace']}. P: Weekly CBT.",
            risk_level="Low"
        )
        stored_history = store.get_longitudinal_history("PT-LIVE-01")
        assert len(stored_history) == 1
        assert "Chest tightness" in stored_history[0]["soap"]
        print("✅ Real clinical Metal rollout, verifier scoring, and local EHR persistence verified.")

    finally:
        engine.destroy()

