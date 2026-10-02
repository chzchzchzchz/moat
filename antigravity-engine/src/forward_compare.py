"""Compare the engine's forward pass with a reference, layer by layer.

The engine's GSM8K run in this repository emitted non-finite logits on 413 of 587 problems and
word salad on the rest, and nothing could say where in the 22 layers it went wrong. This module
takes what AntigravityEngineDebugForward returns — every layer's output for every channel, and
the logits — and what a reference implementation computes for the same token ids, and names
the first place they part.

It is pure numpy, so the logic that decides "this layer is broken" is tested on every commit;
tools/compare_forward.py supplies the engine and the reference.

Layout, matching AntigravityEngineDebugForward:
    engine_hidden  [n_layers + 1, n_channels, hidden_dim]   block 0 the embedding lookup,
                                                            block l + 1 the output of layer l
    ref_hidden     [n_layers + 1, hidden_dim]
    engine_logits  [n_channels, vocab_size]
    ref_logits     [vocab_size]

## Thresholds, and where they come from

Not chosen by feel. Emulating a CORRECT FP16 engine against fp32 on TinyLlama — every linear
layer's output and the residual stream rounded to FP16, with FP16 accumulation in the matrix
products (tools/experiments/fp16_accumulation.py) — on three GSM8K prompts gave a per-layer
relative error of at most 1.05%, growing gently with depth, 0.6% on the logits, and the same
top token every time. The embedding block is a lookup, so it was exact.

The emulation does not model attention's FP16 internals, so the engine may be noisier than
that. The divergence threshold is 10% — about ten times the emulated worst case — with a
warning band from 3%. A layer that is actually broken, with the wrong weights or the wrong
positions, typically lands between 50% and 150%. The embedding block, being exact, is held to
1%. Channels fed the same input should be identical; they are held to 0.1%.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

DIVERGED = 0.10
WARN = 0.03
EMBEDDING_DIVERGED = 0.01
CHANNELS_DIVERGED = 1e-3


def _rel_err(got: np.ndarray, want: np.ndarray) -> float:
    """||got - want|| / ||want||; inf if got is not finite, or if want is zero and got is not."""
    if not np.all(np.isfinite(got)):
        return math.inf
    denom = float(np.linalg.norm(want))
    diff = float(np.linalg.norm(got.astype(np.float64) - want.astype(np.float64)))
    if denom == 0.0:
        return 0.0 if diff == 0.0 else math.inf
    return diff / denom


def block_name(index: int) -> str:
    return "embedding" if index == 0 else f"layer {index - 1}"


def compare_forward(engine_hidden: np.ndarray, ref_hidden: np.ndarray,
                    engine_logits: np.ndarray, ref_logits: np.ndarray,
                    diverged: float = DIVERGED, warn: float = WARN) -> Dict:
    """Everything the comparison found, and a verdict.

    The verdict names the EARLIEST problem, because everything after a broken layer is broken
    too and says nothing new: the first non-finite block, the first block where the channels
    stop agreeing, or the first block that departs from the reference.
    """
    engine_hidden = np.asarray(engine_hidden, dtype=np.float32)
    ref_hidden = np.asarray(ref_hidden, dtype=np.float32)
    engine_logits = np.asarray(engine_logits, dtype=np.float32)
    ref_logits = np.asarray(ref_logits, dtype=np.float32)

    if engine_hidden.ndim != 3:
        raise ValueError(f"engine_hidden must be [blocks, channels, hidden], got {engine_hidden.shape}")
    blocks, channels, hidden = engine_hidden.shape
    if ref_hidden.shape != (blocks, hidden):
        raise ValueError(f"ref_hidden is {ref_hidden.shape}, expected {(blocks, hidden)}: the "
                         f"engine and the reference disagree on the number of layers or the "
                         f"hidden size, so they are not the same model")
    if engine_logits.ndim != 2 or engine_logits.shape[0] != channels:
        raise ValueError(f"engine_logits must be [channels, vocab], got {engine_logits.shape}")
    if ref_logits.shape != (engine_logits.shape[1],):
        raise ValueError(f"ref_logits is {ref_logits.shape}, expected ({engine_logits.shape[1]},)")

    rows: List[Dict] = []
    first_nonfinite: Optional[int] = None
    first_channel_split: Optional[int] = None
    first_diverged: Optional[int] = None
    first_warned: Optional[int] = None

    for b in range(blocks):
        e0 = engine_hidden[b, 0]
        nonfinite = int(np.sum(~np.isfinite(engine_hidden[b])))
        err = _rel_err(e0, ref_hidden[b])
        # How far the other channels are from channel 0. They were given identical input.
        spread = max((_rel_err(engine_hidden[b, c], e0) for c in range(1, channels)), default=0.0)
        limit = EMBEDDING_DIVERGED if b == 0 else diverged
        cos = float("nan")
        if nonfinite == 0:
            n = float(np.linalg.norm(e0)) * float(np.linalg.norm(ref_hidden[b]))
            cos = float(np.dot(e0.astype(np.float64), ref_hidden[b].astype(np.float64)) / n) if n else float("nan")
        rows.append({
            "block": b, "name": block_name(b), "rel_err": err, "cosine": cos,
            "nonfinite": nonfinite, "channel_spread": spread,
            "max_abs_engine": float(np.nanmax(np.abs(np.where(np.isfinite(e0), e0, np.nan))))
                              if np.any(np.isfinite(e0)) else float("nan"),
            "max_abs_ref": float(np.max(np.abs(ref_hidden[b]))),
        })
        if nonfinite and first_nonfinite is None:
            first_nonfinite = b
        if spread > CHANNELS_DIVERGED and first_channel_split is None:
            first_channel_split = b
        if err > limit and first_diverged is None:
            first_diverged = b
        if err > warn and first_warned is None:
            first_warned = b

    # Logits, channel 0 against the reference.
    l0 = engine_logits[0]
    logits_nonfinite = int(np.sum(~np.isfinite(engine_logits)))
    logit_err = _rel_err(l0, ref_logits)
    ref_top5 = [int(i) for i in np.argsort(-ref_logits)[:5]]
    eng_top5 = [int(i) for i in np.argsort(-np.where(np.isfinite(l0), l0, -np.inf))[:5]]
    logits = {
        "rel_err": logit_err, "nonfinite": logits_nonfinite,
        "ref_top5": ref_top5, "engine_top5": eng_top5,
        "top1_agrees": bool(eng_top5[0] == ref_top5[0]),
        "top5_overlap": len(set(ref_top5) & set(eng_top5)),
        "channel_spread": max((_rel_err(engine_logits[c], l0) for c in range(1, channels)), default=0.0),
    }

    # The verdict: the earliest thing that is wrong.
    candidates = []
    if first_nonfinite is not None:
        candidates.append((first_nonfinite, "nonfinite",
                           f"non-finite values first appear at the output of the "
                           f"{block_name(first_nonfinite)} ({rows[first_nonfinite]['nonfinite']} of "
                           f"{channels * hidden}). Everything downstream inherits them; in "
                           f"generation they become the NaN logits behind the 给 rows."))
    if first_channel_split is not None:
        candidates.append((first_channel_split, "channels",
                           f"channels given identical input disagree from the "
                           f"{block_name(first_channel_split)} on (spread "
                           f"{rows[first_channel_split]['channel_spread']:.3g}): a batch "
                           f"indexing or stride defect, independent of the reference."))
    if first_diverged is not None:
        r = rows[first_diverged]
        candidates.append((first_diverged, "diverged",
                           f"the {block_name(first_diverged)} departs from the reference: "
                           f"relative error {r['rel_err']:.3g} (threshold "
                           f"{EMBEDDING_DIVERGED if first_diverged == 0 else diverged:g}), "
                           f"cosine {r['cosine']:.3f}. The previous block agreed, so the defect is "
                           f"in this one." if first_diverged > 0 else
                           f"the embedding lookup departs from the reference: relative error "
                           f"{r['rel_err']:.3g}. Check the token ids, the embedding table and "
                           f"the embedding pipeline before anything else."))
    if candidates:
        candidates.sort(key=lambda c: c[0])
        block, kind, message = candidates[0]
        verdict = {"ok": False, "kind": kind, "block": block, "name": block_name(block),
                   "message": message}
    elif logits_nonfinite or logit_err > diverged:
        verdict = {"ok": False, "kind": "logits", "block": blocks, "name": "final norm and lm_head",
                   "message": f"every layer agrees but the logits do not (relative error "
                              f"{logit_err:.3g}, {logits_nonfinite} non-finite): the defect is in "
                              f"the final RMSNorm or the lm_head projection."}
    else:
        note = ""
        if first_warned is not None:
            note = (f" Error passes {warn:g} from the {block_name(first_warned)} on — within "
                    f"tolerance, but larger than a correct FP16 engine was measured to show.")
        verdict = {"ok": True, "kind": "agrees", "block": None, "name": None,
                   "message": f"every layer and the logits agree with the reference within "
                              f"tolerance; top-1 {'agrees' if logits['top1_agrees'] else 'differs'}, "
                              f"top-5 overlap {logits['top5_overlap']}/5.{note}"}

    return {"blocks": rows, "logits": logits, "verdict": verdict,
            "thresholds": {"diverged": diverged, "warn": warn,
                           "embedding_diverged": EMBEDDING_DIVERGED,
                           "channels_diverged": CHANNELS_DIVERGED}}
