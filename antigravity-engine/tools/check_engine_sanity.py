#!/usr/bin/env python3
"""
Does the engine's output depend on its input? Answer in under a minute.

This check exists because the repository contains gsm8k_full_checkpoint.json: 587
GSM8K problems produced by this engine, in which one character accounts for 100% of
the degenerate generations, accuracy is 0.3% at every channel count, and the run
completed and wrote every row. Nothing stopped it. A forward pass that ignores its
input is the one failure that no amount of grading can detect, because every answer
is simply wrong and the verdict reads as a weak model.

Three prompts that share almost no tokens must not produce the same output. That is a
weaker claim than "the engine is correct" and a much stronger one than any accuracy
figure: a model can be bad, but it cannot be indifferent to its input.

Varying output is necessary but not sufficient, and the same artifact shows why: its 205
problems that were NOT one repeated character are word salad — "keseflectoractressample…",
", l pelo\nusername, Iah Speh of the same" — which varies with its input and is not a
repeated character, so a check of variety alone passes it. The second half of this check
asks for meaning: four prompts whose continuation any working model knows, decoded almost
greedily, at least three of which must come back right. Those four were chosen by running
them through two real models fed exactly as this engine is (no BOS token, no chat template):

    prompt                         TinyLlama-1.1B-Chat         Qwen2.5-0.5B-Instruct
    The capital of France is       "Paris, which is also..."   " Paris. It was founded..."
    1, 2, 3, 4, 5,                 "6, 7, 8, 9,"               " 6, 7, 8, 9,"
    Monday, Tuesday, Wednesday,    "Thursday, Friday, ..."     " and Thursday are each..."
    Water freezes at 0 degrees     "Celsius, and the ..."      " Celsius and boils at..."

Two other candidates were dropped because Qwen answered them as multiple-choice questions.
Three of four rather than four, so one argmax flipped by FP16 arithmetic cannot fail a
healthy engine; word salad gets none.

Run this before spending an hour on a benchmark.

    python3 tools/check_engine_sanity.py --model-dir models/bench \\
        --dylib build/lib/libantigravity_engine.dylib

Exit codes: 0 usable, 1 the engine is not producing input-dependent output,
2 it could not be run at all.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from engine_paths import find_dylib, resolve_weights  # noqa: E402

PROMPTS = [
    "What is 2 plus 2?",
    "Describe the colour of the ocean at sunset in one sentence.",
    "List three prime numbers larger than fifty.",
]


def _first_number_is(expected: str):
    def check(text: str) -> bool:
        match = re.search(r"-?\d+", text)
        return bool(match) and match.group(0) == expected
    return check


def _contains(word: str):
    return lambda text: word in text.lower()


# Prompts with a continuation a working model knows. See the module docstring for how they
# were chosen and measured. "6" is checked as the first number rather than as a substring,
# because a 6 turns up in word salad by chance; Paris, Thursday and Celsius do not.
KNOWN_ANSWERS = [
    ("The capital of France is", "Paris", _contains("paris")),
    ("1, 2, 3, 4, 5,", "6 as the next number", _first_number_is("6")),
    ("Monday, Tuesday, Wednesday,", "Thursday", _contains("thursday")),
    ("Water freezes at 0 degrees", "Celsius", _contains("celsius")),
]
KNOWN_ANSWERS_REQUIRED = 3
KNOWN_ANSWER_TOKENS = 12
# Near-greedy: the engine clamps temperature at 1e-6, and at 1e-3 a logit gap of 0.01 is
# already a factor of e^10, so this is the model's argmax in practice. top_p = 1.0 keeps
# nucleus truncation out of it.
NEAR_GREEDY_TEMPERATURE = 0.001


def dominant_char_share(text: str) -> float:
    """Share of the most common non-space character — 1.0 means one repeated token."""
    stripped = "".join(text.split())
    if not stripped:
        return 1.0
    return collections.Counter(stripped).most_common(1)[0][1] / len(stripped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dylib", default=None)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--int4", action="store_true")
    args = ap.parse_args()

    if args.int4:
        import os
        os.environ["ANTIGRAVITY_INT4"] = "1"

    model_dir = Path(args.model_dir).expanduser().resolve()
    try:
        weights = resolve_weights(model_dir)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    from native_bridge import NativeMetalEngine   # noqa: PLC0415
    from tokenizer import LlamaTokenizer          # noqa: PLC0415

    found = find_dylib(args.dylib)
    tokenizer = LlamaTokenizer(str(model_dir / "tokenizer.json"))
    engine = NativeMetalEngine(n_channels=args.channels,
                               dylib_path=str(found) if found else None)
    if not engine.load_weights(str(weights)):
        print(f"engine failed to load {weights}", file=sys.stderr)
        return 2

    outputs, failures = [], []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        channels, _logprobs, _ttft, _total = engine.generate(
            ids, max_new_tokens=args.max_tokens, temperature=0.7, top_p=0.9)
        text = tokenizer.decode(channels[0])
        outputs.append((prompt, channels[0], text))
        print(f"\nprompt   {prompt}")
        print(f"tokens   {channels[0][:12]}{' ...' if len(channels[0]) > 12 else ''}")
        print(f"text     {text[:160]!r}")

    known = []
    for prompt, expected, is_right in KNOWN_ANSWERS:
        ids = tokenizer.encode(prompt)
        channels, _logprobs, _ttft, _total = engine.generate(
            ids, max_new_tokens=KNOWN_ANSWER_TOKENS,
            temperature=NEAR_GREEDY_TEMPERATURE, top_p=1.0)
        text = tokenizer.decode(channels[0])
        right = is_right(text)
        known.append((prompt, expected, text, right))
        print(f"\nknown    {prompt!r} -> expect {expected}")
        print(f"text     {text[:80]!r}  {'right' if right else 'WRONG'}")

    engine.destroy()

    # 1. Different prompts must give different token sequences.
    sequences = {tuple(tokens) for _p, tokens, _t in outputs}
    if len(sequences) == 1:
        failures.append("all three prompts produced the IDENTICAL token sequence — the "
                        "forward pass is not reading its input")
    elif len(sequences) < len(outputs):
        failures.append(f"only {len(sequences)} distinct sequences from {len(outputs)} "
                        f"very different prompts")

    # 2. No output may be one token repeated.
    for prompt, tokens, text in outputs:
        share = dominant_char_share(text)
        if share > 0.5:
            failures.append(f"output for {prompt!r} is {share * 100:.0f}% a single "
                            f"repeated character")
        if len(set(tokens)) == 1 and len(tokens) > 3:
            failures.append(f"output for {prompt!r} is one token id repeated "
                            f"{len(tokens)} times")

    # 3. Anything generated at all.
    if any(len(tokens) == 0 for _p, tokens, _t in outputs):
        failures.append("at least one prompt generated no tokens")

    # 4. Meaning, not just variety.
    correct = sum(1 for *_rest, right in known if right)
    if correct < KNOWN_ANSWERS_REQUIRED:
        wrong = ", ".join(f"{p!r}" for p, _e, _t, right in known if not right)
        failures.append(f"only {correct} of {len(known)} prompts with a known continuation "
                        f"came back right (need {KNOWN_ANSWERS_REQUIRED}; wrong: {wrong}) — "
                        f"output that varies with its input but makes no sense, which is "
                        f"what the non-degenerate rows of gsm8k_full_checkpoint.json are")

    print()
    if failures:
        print("ENGINE IS NOT USABLE — do not benchmark it in this state:")
        for f in failures:
            print(f"  - {f}")
        print("\nAn accuracy run against this would report a low score and read as a weak")
        print("model, which is what gsm8k_full_checkpoint.json in this repository is.")
        return 1

    print(f"Engine output varies with its input, is not degenerate, and completed "
          f"{correct} of {len(known)} known continuations.")
    print("This does not say the engine is correct — only that it is worth measuring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
