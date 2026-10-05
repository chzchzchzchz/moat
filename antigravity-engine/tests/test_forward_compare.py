"""Tests for src/forward_compare.py — the logic that names the broken layer.

Each scenario is built the way a real fault looks: a defect at one layer propagates to every
layer after it, so "the first block that departs" has to be found among many that do. The
healthy baseline carries noise at the level a correct FP16 engine was measured to show (about
1% per layer), so the thresholds are tested against what they will actually see.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from forward_compare import (CHANNELS_DIVERGED, DIVERGED, EMBEDDING_DIVERGED,  # noqa: E402
                             block_name, compare_forward)

L, C, H, V = 22, 8, 64, 200


def reference(seed=0):
    rng = np.random.default_rng(seed)
    # Residual-stream norms grow with depth, as they do in a real model.
    ref = np.stack([rng.normal(0, 1 + 0.3 * b, H) for b in range(L + 1)]).astype(np.float32)
    logits = rng.normal(0, 3, V).astype(np.float32)
    return ref, logits


def healthy_engine(ref, logits, noise=0.008, seed=1):
    """The reference, per channel, with ~0.8% relative noise per block — what a correct FP16
    engine was measured to show — identical across channels, the embedding exact."""
    rng = np.random.default_rng(seed)
    blocks = []
    for b in range(L + 1):
        r = ref[b]
        n = 0.0 if b == 0 else noise
        e = r + rng.normal(0, 1, H).astype(np.float32) * (n * np.linalg.norm(r) / math.sqrt(H))
        blocks.append(np.tile(e, (C, 1)))
    eng = np.stack(blocks).astype(np.float32)
    lg = logits + rng.normal(0, 1, V).astype(np.float32) * (0.004 * np.linalg.norm(logits) / math.sqrt(V))
    return eng, np.tile(lg, (C, 1)).astype(np.float32)


def break_from(eng, block, scale=0.8, seed=2):
    """A defect at `block` that every later block inherits."""
    rng = np.random.default_rng(seed)
    out = eng.copy()
    for b in range(block, L + 1):
        ref_norm = np.linalg.norm(out[b, 0])
        noise = rng.normal(0, 1, H).astype(np.float32) * (scale * ref_norm / math.sqrt(H))
        out[b] = out[b] + noise          # same corruption on every channel
    return out


def test_a_healthy_engine_agrees():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["ok"], r["verdict"]["message"]
    assert r["verdict"]["kind"] == "agrees"
    assert r["logits"]["top1_agrees"]
    assert all(row["nonfinite"] == 0 for row in r["blocks"])


@pytest.mark.parametrize("layer", [0, 5, 13, 21])
def test_the_first_broken_layer_is_named_not_a_later_one(layer):
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    eng = break_from(eng, layer + 1)                 # layer l's output is block l + 1
    r = compare_forward(eng, ref, elog, logits)
    assert not r["verdict"]["ok"]
    assert r["verdict"]["kind"] == "diverged"
    assert r["verdict"]["block"] == layer + 1
    assert r["verdict"]["name"] == f"layer {layer}"


def test_a_broken_embedding_is_held_to_its_tighter_threshold():
    # The embedding is a lookup and exact in a correct engine, so 2% there is a defect even
    # though 2% would be tolerated in a later layer.
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    eng = break_from(eng, 0, scale=0.02)
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["kind"] == "diverged" and r["verdict"]["block"] == 0
    assert r["verdict"]["name"] == "embedding"
    assert 0.02 * 0.5 < r["blocks"][0]["rel_err"] < DIVERGED, "within the per-layer limit, over the embedding's"


def test_the_first_non_finite_block_is_named():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    for b in range(9, L + 1):
        eng[b, :, 5] = np.nan                         # appears at block 9, never leaves
    elog[:, 3] = np.nan
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["kind"] == "nonfinite"
    assert r["verdict"]["block"] == 9 and r["verdict"]["name"] == "layer 8"
    assert r["blocks"][9]["rel_err"] == math.inf
    assert "给" in r["verdict"]["message"], "the message ties it back to the artifact"


def test_channels_that_disagree_are_caught_even_when_channel_0_is_right():
    # A batch-stride defect can leave channel 0 correct and corrupt the others. The reference
    # comparison, which looks at channel 0, cannot see it; the channel check can.
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    rng = np.random.default_rng(7)
    for b in range(4, L + 1):
        eng[b, 3] = eng[b, 3] + rng.normal(0, 0.5, H).astype(np.float32)
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["kind"] == "channels"
    assert r["verdict"]["block"] == 4
    assert r["blocks"][4]["rel_err"] < DIVERGED, "channel 0 itself still agrees with the reference"


def test_the_earliest_problem_wins():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    eng = break_from(eng, 6)
    for b in range(15, L + 1):
        eng[b, :, 0] = np.inf
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["block"] == 6, "the divergence at block 6 precedes the Inf at 15"


def test_a_nonfinite_block_outranks_a_divergence_at_the_same_block():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    eng = break_from(eng, 7)
    eng[7, 2, 0] = np.nan
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["block"] == 7 and r["verdict"]["kind"] == "nonfinite"


def test_layers_fine_but_logits_wrong_points_at_the_head():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    rng = np.random.default_rng(3)
    elog = np.tile(rng.normal(0, 3, V).astype(np.float32), (C, 1))
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["kind"] == "logits"
    assert "lm_head" in r["verdict"]["message"]


def test_non_finite_logits_alone_point_at_the_head():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    elog[:, 10] = np.nan
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["kind"] == "logits"
    assert r["logits"]["nonfinite"] == C


def test_error_between_warn_and_diverged_passes_with_a_note():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits, noise=0.05)    # 5%: noisier than measured, not broken
    r = compare_forward(eng, ref, elog, logits)
    assert r["verdict"]["ok"]
    assert "larger than a correct FP16 engine" in r["verdict"]["message"]


def test_a_different_model_is_refused_not_compared():
    ref, logits = reference()
    eng, elog = healthy_engine(ref, logits)
    with pytest.raises(ValueError, match="not the same model"):
        compare_forward(eng, ref[:-1], elog, logits)     # one layer short


def test_thresholds_are_the_calibrated_ones():
    # Changing these changes what the tool calls broken; the docstring says where they come
    # from, and this makes changing them a deliberate act.
    assert DIVERGED == 0.10
    assert EMBEDDING_DIVERGED == 0.01
    assert CHANNELS_DIVERGED == 1e-3


def test_block_names():
    assert block_name(0) == "embedding"
    assert block_name(1) == "layer 0"
    assert block_name(22) == "layer 21"
