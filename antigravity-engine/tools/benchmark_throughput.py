#!/usr/bin/env python3
"""
Project Antigravity — throughput benchmark, native Metal engine vs PyTorch MPS.

Why this exists
---------------
The repository quoted ~5.3 tok/s (native) and ~27.0 tok/s (PyTorch MPS) in its
READMEs, but no machine-generated artifact produced either figure. The artifacts
that do exist disagree with each other by four orders of magnitude:

    benchmark_metrics.json        5.14 tok/s
    benchmark_real_weights.json   5.29 tok/s
    benchmark_results_v2.json   855.62 tok/s
    metal_hardware_proof.md  30,327.6 tok/s   (a raw GEMM microbenchmark,
                                               presented beside llama.cpp and MLX
                                               as if it were generation throughput)

Optimising against numbers like that is optimising against noise. This harness
measures both backends on one machine, with one prompt set, one token budget,
and writes a JSON artifact recording what hardware it ran on and what it did.

Decode is memory-bandwidth bound: every token streams the whole weight set. So
the report also states the roofline implied by the measured weight footprint,
which is what tells you whether a backend is near the hardware limit or leaving
most of it unused.

Usage:
    python tools/benchmark_throughput.py --model models/tinyllama \\
        --tokens 64 --repeats 3 --out benchmark_throughput.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def hardware_profile() -> dict:
    """Record the machine, so a number is never again quoted without its context."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if sys.platform == "darwin":
        for key, cmd in [
            ("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
        ]:
            try:
                info[key] = subprocess.check_output(cmd, text=True).strip()
            except Exception as exc:
                info[key] = f"unavailable: {exc}"
    return info


def measure(fn, repeats: int) -> dict:
    """Run fn() repeats times; report every sample, not just the best one."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n_tokens = fn()
        dt = time.perf_counter() - t0
        samples.append({"tokens": n_tokens, "seconds": dt,
                        "tokens_per_sec": n_tokens / dt if dt > 0 else 0.0})
    tps = sorted(s["tokens_per_sec"] for s in samples)
    return {
        "samples": samples,
        "median_tokens_per_sec": tps[len(tps) // 2],
        "min_tokens_per_sec": tps[0],
        "max_tokens_per_sec": tps[-1],
    }


def bench_native(model_dir: str, prompt_ids, max_tokens: int, repeats: int) -> dict:
    from native_bridge import NativeMetalEngine
    engine = NativeMetalEngine(n_channels=1)
    weights = os.path.join(model_dir, "model.safetensors")
    if not engine.load_weights(weights):
        raise RuntimeError(f"native engine failed to load {weights}")

    def once():
        ch, _lp, _ttft, _total = engine.generate(
            prompt_token_ids=prompt_ids, max_new_tokens=max_tokens,
            temperature=0.0, top_p=1.0,
        )
        return sum(len(c) for c in ch)

    once()  # warm-up: first call pays pipeline and allocation costs
    result = measure(once, repeats)
    result["weights_mb"] = os.path.getsize(weights) / (1024 * 1024)
    engine.destroy()
    return result


def bench_torch_mps(model_dir: str, prompt_ids, max_tokens: int, repeats: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16).to(device).eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    def once():
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
        if device == "mps":
            torch.mps.synchronize()  # generate() is async; time it honestly
        return out.shape[1] - ids.shape[1]

    once()
    result = measure(once, repeats)
    result["device"] = device
    result["weights_mb"] = sum(
        p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/tinyllama")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmark_throughput.json")
    ap.add_argument("--bandwidth-gbs", type=float, default=None,
                    help="Measured memory bandwidth, to report roofline utilisation.")
    args = ap.parse_args()

    prompt_ids = [1, 15043, 29892, 1125, 29892, 29871, 313, 29906]
    report = {
        "hardware": hardware_profile(),
        "model_dir": args.model,
        "max_new_tokens": args.tokens,
        "repeats": args.repeats,
        "prompt_token_ids": prompt_ids,
        "backends": {},
        "errors": {},
    }

    for name, fn in [("native_metal", bench_native), ("pytorch_mps", bench_torch_mps)]:
        try:
            report["backends"][name] = fn(args.model, prompt_ids, args.tokens, args.repeats)
            r = report["backends"][name]
            print(f"{name:>14}: {r['median_tokens_per_sec']:7.2f} tok/s median "
                  f"({r['min_tokens_per_sec']:.2f}-{r['max_tokens_per_sec']:.2f}), "
                  f"weights {r['weights_mb']:.0f} MB")
        except Exception as exc:
            # A backend that cannot run is recorded, never silently omitted.
            report["errors"][name] = f"{type(exc).__name__}: {exc}"
            print(f"{name:>14}: unavailable — {type(exc).__name__}: {exc}")

    if args.bandwidth_gbs:
        report["roofline"] = {}
        for name, r in report["backends"].items():
            gb = r["weights_mb"] / 1024.0
            ceiling = args.bandwidth_gbs / gb if gb else 0.0
            report["roofline"][name] = {
                "bandwidth_gbs": args.bandwidth_gbs,
                "ceiling_tokens_per_sec": ceiling,
                "fraction_of_roofline": (r["median_tokens_per_sec"] / ceiling) if ceiling else 0.0,
            }
            print(f"{name:>14}: {report['roofline'][name]['fraction_of_roofline']:.1%} of roofline")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")
    return 0 if report["backends"] else 1


if __name__ == "__main__":
    sys.exit(main())
