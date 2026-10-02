"""
Tests for tools/check_engine_sanity.py.

This one guards the others, so it has to work. It is the thing standing between a
forward pass that ignores its input and an hour of grading that reports the resulting
garbage as a weak model — which is what gsm8k_full_checkpoint.json in this repository
is. A guard that silently passes is worse than no guard, because it manufactures
confidence.

Driven against a stand-in engine, so it runs anywhere.
"""

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))


TEXT = 1000   # ids at or above this carry a character: chr(id - TEXT)


class FakeTokenizer:
    """Reversible, so the stand-in engine can tell which prompt it was given.

    Ids below TEXT decode to letters, so a scripted run of one small id decodes to one
    repeated character, as the scripted tests below rely on."""

    def __init__(self, path):
        pass

    def encode(self, text):
        return [TEXT + ord(c) for c in text]

    def decode(self, token_ids):
        return "".join(chr(t - TEXT) if t >= TEXT else chr(97 + (t % 26)) for t in token_ids)


def as_ids(text):
    return [TEXT + ord(c) for c in text]


# Continuations a working model gives for the check's known-answer prompts, as measured on
# TinyLlama and Qwen (see check_engine_sanity's docstring).
RIGHT = {
    "The capital of France is": " Paris, which is also the largest city",
    "1, 2, 3, 4, 5,": " 6, 7, 8, 9,",
    "Monday, Tuesday, Wednesday,": " Thursday, Friday, Saturday",
    "Water freezes at 0 degrees": " Celsius, and boils at 100",
}


class FakeEngine:
    """Answers the known-answer prompts from `known` (right, unless a test changes it) and
    the free prompts from `script`, in order."""

    def __init__(self, n_channels=2, **_):
        self.n_channels = n_channels
        self.script = []
        self.known = dict(RIGHT)
        self.free_calls = 0
        self.known_calls = []          # (prompt, temperature, top_p)
        self.destroyed = False

    def load_weights(self, path):
        return True

    def generate(self, ids, max_new_tokens=40, temperature=0.7, top_p=0.9):
        prompt = "".join(chr(t - TEXT) for t in ids)
        if prompt in self.known:
            self.known_calls.append((prompt, temperature, top_p))
            tokens = as_ids(self.known[prompt])
        else:
            tokens = self.script[self.free_calls % len(self.script)]
            self.free_calls += 1
        return [list(tokens)] * self.n_channels, [-1.0] * self.n_channels, 5.0, 50.0

    def destroy(self):
        self.destroyed = True


@pytest.fixture
def checker(monkeypatch, tmp_path):
    engine = FakeEngine()
    native = types.ModuleType("native_bridge")
    native.NativeMetalEngine = lambda n_channels=2, **kw: engine
    tok = types.ModuleType("tokenizer")
    tok.LlamaTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "native_bridge", native)
    monkeypatch.setitem(sys.modules, "tokenizer", tok)
    (tmp_path / "model.safetensors").write_bytes(b"")
    import check_engine_sanity
    return check_engine_sanity, engine, tmp_path


def run(module, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["check_engine_sanity", "--model-dir", str(tmp_path)])
    return module.main()


# ---------------------------------------------------------------------------

def test_healthy_varied_output_passes(checker, monkeypatch):
    module, engine, tmp_path = checker
    engine.script = [
        [5, 12, 7, 19, 2, 30],
        [8, 3, 25, 11, 40, 6],
        [14, 22, 1, 9, 33, 17],
    ]
    assert run(module, monkeypatch, tmp_path) == 0
    assert engine.destroyed


def test_identical_output_for_every_prompt_fails(checker, monkeypatch):
    """The failure that shipped: output independent of input."""
    module, engine, tmp_path = checker
    same = [5, 12, 7, 19, 2, 30]
    engine.script = [same, same, same]
    assert run(module, monkeypatch, tmp_path) == 1


def test_one_repeated_token_fails(checker, monkeypatch):
    """The 587-problem checkpoint's exact shape: one token, over and over."""
    module, engine, tmp_path = checker
    engine.script = [[7] * 20, [7] * 20, [7] * 20]
    assert run(module, monkeypatch, tmp_path) == 1


def test_a_single_repeated_token_fails_even_when_prompts_differ(checker, monkeypatch):
    # Different sequences, but each is degenerate on its own. Both checks matter.
    module, engine, tmp_path = checker
    engine.script = [[3] * 20, [4] * 20, [5] * 20]
    assert run(module, monkeypatch, tmp_path) == 1


def test_empty_generation_fails(checker, monkeypatch):
    module, engine, tmp_path = checker
    engine.script = [[], [8, 3, 25, 11], [14, 22, 1, 9]]
    assert run(module, monkeypatch, tmp_path) == 1


def test_two_of_three_prompts_colliding_fails(checker, monkeypatch):
    # Partial collapse is still a red flag; three unrelated prompts should not tie.
    module, engine, tmp_path = checker
    shared = [5, 12, 7, 19, 2, 30]
    engine.script = [shared, shared, [14, 22, 1, 9, 33, 17]]
    assert run(module, monkeypatch, tmp_path) == 1


def test_missing_weights_is_exit_two_not_a_pass(checker, monkeypatch, tmp_path):
    # A setup failure must not be reported as the engine being fine.
    module, engine, _ = checker
    engine.script = [[1, 2, 3]] * 3
    empty = tmp_path / "nothing"
    empty.mkdir()
    monkeypatch.setattr(sys, "argv", ["check_engine_sanity", "--model-dir", str(empty)])
    assert module.main() == 2


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_high", [
    ("aaaaaaaaaaaaaaaaaaaa", True),
    ("给给给给给给给给给给给给", True),
    ("a a a a a a a a a a", True),          # whitespace must not dilute the share
    ("the quick brown fox jumps over", False),
    ("", True),                              # nothing generated counts as degenerate
])
def test_dominant_char_share(checker, text, expected_high):
    module, _engine, _tmp = checker
    share = module.dominant_char_share(text)
    assert (share > 0.5) is expected_high, f"{text!r} -> {share}"


# ---------------------------------------------------------------------------
# Meaning, not just variety
#
# The artifact's 205 problems that were NOT one repeated character are word salad. It varies
# with its input and is not a repeated character, so the variety checks above pass it. These
# are real outputs from gsm8k_full_checkpoint.json.
# ---------------------------------------------------------------------------

SALAD = [
    ", l pelo\nusername, Iah Speh of the same -  but\ncomcome of thenvisedly",
    "keseflectoractressampleveytheistory <\n2 end6lain comen J Postorm betme",
    "\n\nthe < bimes forimage still mathematical m andimm and millions Again\n",
    "ing and have a long term contract for for \n $ oh nos Long",
]


def test_word_salad_from_the_artifact_fails(checker, monkeypatch, capsys):
    module, engine, tmp_path = checker
    engine.script = [as_ids(SALAD[0]), as_ids(SALAD[1]), as_ids(SALAD[2])]
    engine.known = {p: SALAD[i] for i, p in enumerate(RIGHT)}

    assert run(module, monkeypatch, tmp_path) == 1
    out = capsys.readouterr().out
    assert "known continuation" in out
    # And it is the known-answer check that catches it: the variety checks alone pass this,
    # which is the whole reason it was added.
    assert "IDENTICAL" not in out and "repeated" not in out


def test_three_of_four_known_answers_is_enough(checker, monkeypatch):
    # One argmax flipped by FP16 arithmetic must not fail a healthy engine.
    module, engine, tmp_path = checker
    engine.script = [as_ids("two plus two is four"), as_ids("a warm orange glow"),
                     as_ids("53, 59 and 61")]
    engine.known["Water freezes at 0 degrees"] = " Fahrenheit, they said"
    assert run(module, monkeypatch, tmp_path) == 0


def test_two_of_four_known_answers_fails(checker, monkeypatch):
    module, engine, tmp_path = checker
    engine.script = [as_ids("two plus two is four"), as_ids("a warm orange glow"),
                     as_ids("53, 59 and 61")]
    engine.known["Water freezes at 0 degrees"] = " Fahrenheit"
    engine.known["The capital of France is"] = " Lyon"
    assert run(module, monkeypatch, tmp_path) == 1


def test_known_answers_are_decoded_near_greedily(checker, monkeypatch):
    # The answers were measured greedily; sampling at 0.7 could miss them on a healthy engine.
    module, engine, tmp_path = checker
    engine.script = [as_ids("four"), as_ids("orange"), as_ids("53")]
    run(module, monkeypatch, tmp_path)
    assert len(engine.known_calls) == 4, "every known-answer prompt must be asked"
    for prompt, temperature, top_p in engine.known_calls:
        assert temperature <= 0.01, f"{prompt!r} sampled at temperature {temperature}"
        assert top_p == 1.0, f"{prompt!r} truncated with top_p {top_p}"


def test_answers_match_case_insensitively(checker, monkeypatch):
    module, engine, tmp_path = checker
    engine.script = [as_ids("four"), as_ids("orange"), as_ids("53")]
    engine.known["The capital of France is"] = " PARIS!"
    engine.known["Monday, Tuesday, Wednesday,"] = " THURSDAY"
    assert run(module, monkeypatch, tmp_path) == 0


@pytest.mark.parametrize("continuation,right", [
    (" 6, 7, 8, 9,", True),
    ("6", True),
    (" and then 6", True),
    (" 16 eggs", False),          # 16 is not 6, though it contains a 6
    (" 3 apples and 6 pears", False),   # the FIRST number has to be 6
    (" no numbers at all", False),
])
def test_the_counting_prompt_checks_the_first_number(checker, continuation, right):
    module, _engine, _tmp = checker
    check = [c for p, _e, c in module.KNOWN_ANSWERS if p.startswith("1, 2")][0]
    assert check(continuation) is right
