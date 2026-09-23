"""
Tests for tools/benchmark_quality.py using a stand-in engine and tokenizer.

The harness needs Apple Silicon, a built dylib and model weights, so none of it
can run in CI as written. What can run is everything between the engine call and
the artifact — dataset loading, answer extraction from decoded text, how a
channel that answered nothing is scored, what happens when a problem throws, and
the exit codes. Those are where a quiet mistake would corrupt the number without
anything failing, so they are worth covering even though the GPU is not here.
"""

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Round-trips through a token list whose first element is the answer."""

    def __init__(self, path):
        self.path = path

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_ids):
        if not token_ids:
            return "I could not work this out."
        return f"Some reasoning here.\n#### {token_ids[0]}"


class FakeEngine:
    """Returns, per problem, the per-channel answers the test scripted."""

    destroyed = False

    def __init__(self, n_channels=8, **_):
        self.n_channels = n_channels
        self.calls = 0
        self.script = []        # list of per-problem lists of answers (or None)
        self.raise_on = set()

    def load_weights(self, path):
        return True

    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.7, top_p=0.9):
        index = self.calls
        self.calls += 1
        if index in self.raise_on:
            raise RuntimeError(f"scripted engine failure on problem {index}")
        answers = self.script[index]
        tokens = [[] if a is None else [a] for a in answers]
        logprobs = [-1.0] * len(answers)
        return tokens, logprobs, 10.0, 100.0

    def destroy(self):
        FakeEngine.destroyed = True


@pytest.fixture
def harness(monkeypatch):
    """Install the stand-ins where the harness imports them from, and hand back
    the module plus the single engine instance it will construct."""
    engine = FakeEngine()

    native = types.ModuleType("native_bridge")
    native.NativeMetalEngine = lambda n_channels=8, **kw: (
        setattr(engine, "n_channels", n_channels) or engine
    )
    tok = types.ModuleType("tokenizer")
    tok.LlamaTokenizer = FakeTokenizer

    monkeypatch.setitem(sys.modules, "native_bridge", native)
    monkeypatch.setitem(sys.modules, "tokenizer", tok)

    import benchmark_quality
    return benchmark_quality, engine


def write_dataset(tmp_path, golds):
    path = tmp_path / "gsm8k.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i, gold in enumerate(golds):
            handle.write(json.dumps({
                "question": f"problem {i}?",
                "answer": f"working out\n#### {gold}",
            }) + "\n")
    return path


def run(module, monkeypatch, tmp_path, dataset, extra=()):
    out = tmp_path / "result.json"
    argv = ["benchmark_quality", "--model-dir", str(tmp_path),
            "--dataset", str(dataset), "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    code = module.main()
    return code, (json.loads(out.read_text()) if out.exists() else None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_writes_a_complete_artifact(harness, monkeypatch, tmp_path):
    module, engine = harness
    golds = list(range(1, 21))
    dataset = write_dataset(tmp_path, golds)
    # Channel 0 wrong on every problem; the other 7 agree on the gold answer.
    engine.script = [[0] + [g] * 7 for g in golds]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "20"])

    assert code == 0
    assert result["comparison"]["n_problems"] == 20
    assert len(result["records"]) == 20
    assert result["comparison"]["baseline"]["correct"] == 0
    assert result["comparison"]["candidate"]["correct"] == 20
    assert result["comparison"]["significant"]
    assert result["errors"] == []
    # Context that makes the number checkable later.
    assert result["config"]["channels"] == 8
    assert result["config"]["model_dir"] and result["config"]["dataset"]
    assert result["hardware"]["platform"]
    assert result["generated_at"].endswith("+00:00")
    assert engine.destroyed


def test_a_channel_that_answered_nothing_scores_wrong_not_skipped(harness, monkeypatch, tmp_path):
    module, engine = harness
    dataset = write_dataset(tmp_path, [7] * 4)
    # Channel 0 silent every time; the rest are right.
    engine.script = [[None] + [7] * 7 for _ in range(4)]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "4"])

    assert code == 0
    assert result["comparison"]["n_problems"] == 4          # none dropped
    assert result["comparison"]["baseline"]["correct"] == 0
    assert result["comparison"]["candidate"]["correct"] == 4
    assert result["records"][0]["baseline_answer"] is None


def test_all_channels_silent_is_wrong_for_both_conditions(harness, monkeypatch, tmp_path):
    module, engine = harness
    dataset = write_dataset(tmp_path, [3, 3])
    engine.script = [[None] * 8, [None] * 8]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "2"])

    assert code == 0
    assert result["comparison"]["baseline"]["correct"] == 0
    assert result["comparison"]["candidate"]["correct"] == 0
    assert result["records"][0]["selected_answer"] is None


def test_a_failing_problem_is_recorded_and_the_run_exits_nonzero(harness, monkeypatch, tmp_path):
    module, engine = harness
    golds = [1, 2, 3, 4, 5]
    dataset = write_dataset(tmp_path, golds)
    engine.script = [[g] * 8 for g in golds]
    engine.raise_on = {2}

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "5"])

    # The other four still graded, but the run does not report itself as clean.
    assert code == 1
    assert result["comparison"]["n_problems"] == 4
    assert len(result["errors"]) == 1
    assert "scripted engine failure" in result["errors"][0]["error"]


def test_every_problem_failing_produces_no_artifact_and_fails(harness, monkeypatch, tmp_path):
    module, engine = harness
    dataset = write_dataset(tmp_path, [1, 2])
    engine.script = [[1] * 8, [2] * 8]
    engine.raise_on = {0, 1}

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "2"])

    assert code == 1
    assert result is None       # nothing written: there is no number to report


def test_a_null_result_carries_the_sample_size_it_would_have_needed(harness, monkeypatch, tmp_path):
    module, engine = harness
    golds = [1, 2, 3, 4, 5]
    dataset = write_dataset(tmp_path, golds)
    # One problem's difference out of five — the shape of the existing artifact.
    engine.script = [[0] + [g] * 7 if i == 0 else [g] * 8 for i, g in enumerate(golds)]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "5"])

    assert code == 0
    assert not result["comparison"]["significant"]
    assert result["power_note"]["problems_run"] == 5
    assert result["power_note"]["problems_needed_for_5_point_effect"] > 5


def test_one_channel_is_refused_because_the_comparison_would_be_vacuous(harness, monkeypatch, tmp_path):
    module, engine = harness
    dataset = write_dataset(tmp_path, [1, 2])
    engine.script = [[1], [2]]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--channels", "1"])

    assert code == 2
    assert result is None


def test_a_missing_dataset_is_refused(harness, monkeypatch, tmp_path):
    module, _ = harness
    code, result = run(module, monkeypatch, tmp_path, tmp_path / "nope.jsonl")
    assert code == 2
    assert result is None


def test_problems_without_a_gold_answer_are_dropped_not_counted_wrong(harness, monkeypatch, tmp_path):
    module, engine = harness
    path = tmp_path / "mixed.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"question": "a?", "answer": "no marker here"}) + "\n")
        handle.write(json.dumps({"question": "b?", "answer": "work\n#### 5"}) + "\n")
    engine.script = [[5] * 8]

    code, result = run(module, monkeypatch, tmp_path, path, ["--limit", "10"])

    assert code == 0
    assert result["comparison"]["n_problems"] == 1
    assert result["comparison"]["candidate"]["correct"] == 1


def test_limit_caps_the_number_of_problems(harness, monkeypatch, tmp_path):
    module, engine = harness
    golds = list(range(1, 51))
    dataset = write_dataset(tmp_path, golds)
    engine.script = [[g] * 8 for g in golds]

    code, result = run(module, monkeypatch, tmp_path, dataset, ["--limit", "6"])

    assert code == 0
    assert result["comparison"]["n_problems"] == 6
