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


class FakeTokenizer:
    def __init__(self, path):
        pass

    def encode(self, text):
        # Distinct ids per prompt, so a real engine would see different inputs.
        return [len(text), ord(text[0])]

    def decode(self, token_ids):
        # Map ids to letters so repeated ids decode to a repeated character.
        return "".join(chr(97 + (t % 26)) for t in token_ids)


class FakeEngine:
    """Returns whatever token sequences the test scripted, one per prompt."""

    def __init__(self, n_channels=2, **_):
        self.n_channels = n_channels
        self.script = []
        self.calls = 0
        self.destroyed = False

    def load_weights(self, path):
        return True

    def generate(self, ids, max_new_tokens=40, temperature=0.7, top_p=0.9):
        tokens = self.script[self.calls % len(self.script)]
        self.calls += 1
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
