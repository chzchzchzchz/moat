#!/usr/bin/env python3
"""
Where does the engine's forward pass go wrong? Compare it with transformers, layer by layer.

gsm8k_full_checkpoint.json in this repository is the engine emitting non-finite logits on 413
of 587 GSM8K problems — they show as 给, TinyLlama's last token id, because of how the old
sampler handled NaN under Apple's libc++ — and word salad on the rest. Every input to the
forward pass has been ruled out on the CPU (see NEXT_ON_HARDWARE.md); what is left is the
forward pass itself, and this finds the layer.

It runs one prompt through the engine's debug entry point, which returns every layer's output
for every channel and the logits, and through transformers in fp32 on the CPU with the same
token ids, then names the earliest of:

  - the first block holding non-finite values,
  - the first block where channels given identical input stop agreeing (a batch-indexing
    defect, visible without any reference),
  - the first block that departs from the reference by more than a correct FP16 engine was
    measured to (see src/forward_compare.py for the calibration).

    python3 tools/compare_forward.py --model-dir models/bench \\
        --dylib build/lib/libantigravity_engine.dylib

Needs a Mac for the engine, and torch and transformers for the reference. Exit codes: 0 the
engine agrees with the reference, 1 it does not (the report says where), 2 it could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from engine_paths import find_dylib, resolve_weights  # noqa: E402
from forward_compare import compare_forward  # noqa: E402

DEFAULT_PROMPT = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
                  "morning and bakes muffins for her friends every day with four. How many "
                  "eggs does she have left?")

EngineFn = Callable[[List[int]], Tuple[np.ndarray, np.ndarray]]


def reference_forward(model_dir: Path, ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """transformers in fp32: [n_layers + 1, hidden] at the last position, and its logits.

    Hooks rather than output_hidden_states, because the latter's last entry has the final norm
    applied in some versions and not in others, and the engine's block L is the raw output of
    the last layer. Block 0 is the embedding lookup, block l + 1 the output of decoder layer l —
    the layout AntigravityEngineDebugForward returns.
    """
    import torch                                                  # noqa: PLC0415
    from transformers import AutoModelForCausalLM                 # noqa: PLC0415

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    except AttributeError:
        # Multimodal checkpoints carry a nested text_config that the causal-LM class cannot read.
        from transformers import AutoModelForImageTextToText      # noqa: PLC0415
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), dtype=torch.float32).eval()
    captured: List[np.ndarray] = []

    def grab(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h[0, -1].detach().float().cpu().numpy())

    # Multimodal checkpoints (Qwen3.5) nest the text decoder under model.language_model.
    text = getattr(model.model, "language_model", model.model)
    hooks = [text.embed_tokens.register_forward_hook(grab)]
    hooks += [layer.register_forward_hook(grab) for layer in text.layers]
    try:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1].float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()
    expected = len(text.layers) + 1
    if len(captured) != expected:
        raise RuntimeError(f"captured {len(captured)} blocks, expected {expected}")
    return np.stack(captured), logits


def run(engine_fn: EngineFn, ref_fn: Callable[[List[int]], Tuple[np.ndarray, np.ndarray]],
        ids: List[int]) -> Dict:
    engine_hidden, engine_logits = engine_fn(ids)
    ref_hidden, ref_logits = ref_fn(ids)
    return compare_forward(engine_hidden, ref_hidden, engine_logits, ref_logits)


def print_report(report: Dict, ids: List[int]) -> None:
    print(f"\nprompt tokens {len(ids)}; comparing the last position\n")
    print(f"  {'block':<11} {'rel err':>9} {'cosine':>8} {'non-finite':>11} "
          f"{'ch spread':>10} {'max|eng|':>10} {'max|ref|':>10}")
    for row in report["blocks"]:
        err = "inf" if row["rel_err"] == float("inf") else f"{row['rel_err']:.4f}"
        print(f"  {row['name']:<11} {err:>9} {row['cosine']:>8.4f} {row['nonfinite']:>11} "
              f"{row['channel_spread']:>10.2e} {row['max_abs_engine']:>10.3g} "
              f"{row['max_abs_ref']:>10.3g}")
    lg = report["logits"]
    print(f"\n  logits: rel err {lg['rel_err']:.4f}, {lg['nonfinite']} non-finite, top-1 "
          f"{'agrees' if lg['top1_agrees'] else 'DIFFERS'}, top-5 overlap {lg['top5_overlap']}/5")
    print(f"          engine top-5 {lg['engine_top5']}   reference top-5 {lg['ref_top5']}")
    v = report["verdict"]
    print("\n" + ("AGREES: " if v["ok"] else f"FIRST PROBLEM — {v['name']}: ") + v["message"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True,
                    help="weights, tokenizer.json and config.json, as run_quality_benchmark.sh "
                         "downloads them")
    ap.add_argument("--dylib", default=None)
    ap.add_argument("--channels", type=int, default=8,
                    help="the GSM8K run used 8; the batched kernels only run with more than one")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--int4", action="store_true")
    ap.add_argument("--out", default="forward_compare.json")
    args = ap.parse_args()

    if args.int4:
        import os  # noqa: PLC0415
        os.environ["ANTIGRAVITY_INT4"] = "1"

    model_dir = Path(args.model_dir).expanduser().resolve()
    try:
        weights = resolve_weights(model_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not (model_dir / "config.json").is_file():
        print(f"{model_dir}/config.json is missing; transformers needs it for the reference",
              file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401,PLC0415
        import transformers  # noqa: F401,PLC0415
    except ImportError:
        print("the reference needs torch and transformers: python3 -m pip install torch "
              "transformers", file=sys.stderr)
        return 2

    from native_bridge import NativeMetalEngine   # noqa: PLC0415
    from tokenizer import LlamaTokenizer          # noqa: PLC0415

    # The engine's own tokenizer, so the ids are the ones generation sees; the reference is
    # given the same ids, so tokenization cannot differ between the two sides.
    ids = LlamaTokenizer(str(model_dir / "tokenizer.json")).encode(args.prompt)
    found = find_dylib(args.dylib)
    engine = NativeMetalEngine(n_channels=args.channels,
                               dylib_path=str(found) if found else None)
    try:
        if not engine.load_weights(str(weights)):
            print(f"engine failed to load {weights}", file=sys.stderr)
            return 2
        try:
            report = run(engine.debug_forward, lambda i: reference_forward(model_dir, i), ids)
        except (RuntimeError, ValueError) as exc:
            print(f"could not compare: {exc}", file=sys.stderr)
            return 2
    finally:
        engine.destroy()

    print_report(report, ids)
    report["prompt"] = args.prompt
    report["prompt_ids"] = ids
    report["config"] = {"channels": args.channels, "int4": bool(args.int4),
                        "weights": str(weights)}
    Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {args.out}")
    return 0 if report["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
