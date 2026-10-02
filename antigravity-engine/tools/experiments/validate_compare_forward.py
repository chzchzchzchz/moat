"""Does tools/compare_forward.py name the right layer on a real model?

Its decision logic is unit-tested on synthetic arrays (tests/test_forward_compare.py). This
checks the rest against TinyLlama-1.1B-Chat: that reference_forward's hooks line up with the
engine's block layout, and that the calibrated thresholds hold on a real model's errors.

The engine is replaced by a CPU stand-in built from the same weights:
  clean   FP16 storage AND FP16 accumulation in every linear layer, 8 identical channels —
          a correct FP16 engine, as emulated in tools/experiments/fp16_accumulation.py.
          compare_forward must say it AGREES.
  faults  the same with one engine-style bug injected; compare_forward must name its layer:
          o_proj transposed in layer 7, SwiGLU's gate and up swapped in layer 12, a NaN
          appearing in layer 15's output, and one channel corrupted from layer 3 onward.

Usage:  python3 tools/experiments/validate_compare_forward.py <TinyLlama dir>

RESULT (2026-10-02), every case right:

  clean, FP16 storage + FP16 accumulation  -> AGREES              worst block error 0.0076
  o_proj transposed in layer 7             -> diverged, layer 7   worst block error 0.6920
  SwiGLU gate/up swapped in layer 12       -> diverged, layer 12  worst block error 0.3084
  NaN appears in layer 15's output         -> non-finite, layer 15
  channel 5 corrupted from layer 3         -> channels, layer 3

The clean case is the one that tests the threshold: a correct FP16 engine sits at 0.76%, a
thirteenth of the 10% limit, while the two weight faults sit at 31% and 69%.
"""
import gc
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
from compare_forward import DEFAULT_PROMPT, reference_forward, run  # noqa: E402

torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

D = Path(sys.argv[1])
CHANNELS = 8
tok = AutoTokenizer.from_pretrained(str(D))
ids = tok.encode(DEFAULT_PROMPT, add_special_tokens=False)     # as the engine tokenizes

def fp16(x):
    return x.half().float()

def fp16_accum_linear(x, w):
    """Running FP16 sum over 8-term blocks, one block at a time. A first version built every
    block's partial sum at once — [K/8, positions, N], 1.5 GB for the LM head alone — and with
    two fp32 models resident the out-of-memory killer took it at 11.4 GB."""
    K = x.shape[-1]
    xs = fp16(x).reshape(*x.shape[:-1], K // 8, 8)
    ws = fp16(w).reshape(w.shape[0], K // 8, 8)
    acc = None
    for b in range(K // 8):
        part = fp16(xs[..., b, :] @ ws[:, b, :].T)
        acc = part if acc is None else fp16(acc + part)
    return acc


# One stand-in model for every case; each fault is applied and undone around its case.
MODEL = AutoModelForCausalLM.from_pretrained(str(D), dtype=torch.float32).eval()


def apply_fault(fault):
    """Inject a weight fault and return a function that undoes it."""
    if fault == "o_proj_transposed":
        w = MODEL.model.layers[7].self_attn.o_proj.weight
        saved = w.data.clone()
        w.data = w.data.t().contiguous()
        return lambda: setattr(w, "data", saved)
    if fault == "swiglu_swapped":
        mlp = MODEL.model.layers[12].mlp
        g, u = mlp.gate_proj.weight.data.clone(), mlp.up_proj.weight.data.clone()
        mlp.gate_proj.weight.data, mlp.up_proj.weight.data = u.clone(), g.clone()
        def undo():
            mlp.gate_proj.weight.data, mlp.up_proj.weight.data = g, u
        return undo
    return lambda: None


def stand_in_engine(accumulate_fp16, fault=None):
    """An 'engine' made from the reference weights: FP16 numerics, optional injected fault."""
    model = MODEL
    undo_fault = apply_fault(fault)
    orig = torch.nn.Linear.forward
    if accumulate_fp16:
        torch.nn.Linear.forward = lambda self, x: fp16_accum_linear(x, self.weight)
    else:
        torch.nn.Linear.forward = lambda self, x: fp16(orig(self, fp16(x)))
    captured = []

    def grab(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        h = fp16(h)
        if fault == "nan_at_15" and len(captured) == 16:      # block 16 = layer 15's output
            h = h.clone()
            h[0, -1, 100] = float("nan")
        captured.append(h[0, -1].detach().float().numpy())
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    hooks = [model.model.embed_tokens.register_forward_hook(grab)]
    hooks += [layer.register_forward_hook(grab) for layer in model.model.layers]

    def engine_fn(prompt_ids):
        captured.clear()
        with torch.no_grad():
            logits = fp16(model(input_ids=torch.tensor([prompt_ids]),
                                logits_to_keep=1).logits[0, -1]).numpy()
        hidden = np.repeat(np.stack(captured)[:, None, :], CHANNELS, axis=1).copy()
        if fault == "channel_5_from_layer_3":
            rng = np.random.default_rng(0)
            for b in range(4, hidden.shape[0]):               # block 4 = layer 3's output
                hidden[b, 5] += rng.normal(0, 0.05 * np.abs(hidden[b, 5]).mean(), hidden.shape[2])
        return hidden.astype(np.float32), np.repeat(logits[None, :], CHANNELS, axis=0)

    def restore():
        torch.nn.Linear.forward = orig
        for h in hooks:
            h.remove()
        undo_fault()
    return engine_fn, restore


cases = [
    ("clean, FP16 storage + FP16 accumulation", True, None, None),
    ("o_proj transposed in layer 7", False, "o_proj_transposed", "layer 7"),
    ("SwiGLU gate/up swapped in layer 12", False, "swiglu_swapped", "layer 12"),
    ("NaN appears in layer 15's output", False, "nan_at_15", "layer 15"),
    ("channel 5 corrupted from layer 3", False, "channel_5_from_layer_3", "layer 3"),
]
# The reference once: its own fp32 model is loaded, used and released here, rather than
# alongside the stand-in for every case.
REF = reference_forward(D, ids)
gc.collect()
ref = lambda _i: REF  # noqa: E731
failures = 0
for label, acc16, fault, expect in cases:
    engine_fn, restore = stand_in_engine(acc16, fault)
    try:
        report = run(engine_fn, ref, ids)
    finally:
        restore()
    v = report["verdict"]
    worst = max(r["rel_err"] for r in report["blocks"] if r["rel_err"] != float("inf"))
    got = "AGREES" if v["ok"] else f"{v['kind']} at {v['name']}"
    want = "AGREES" if expect is None else expect
    right = v["ok"] if expect is None else (not v["ok"] and v["name"] == expect)
    failures += 0 if right else 1
    print(f"{'OK  ' if right else 'FAIL'} {label:44} -> {got:28} (expected {want}; "
          f"worst finite block error {worst:.4f})", flush=True)
print("\nALL CASES RIGHT" if failures == 0 else f"\n{failures} CASE(S) WRONG")
sys.exit(1 if failures else 0)
