"""Tests for checkpointing and resume in tools/benchmark_quality_reference.py.

The tool used to write its output once, at the very end. A 50-problem run at 8 samples and
768 tokens takes about two hours on CPU, and when the container holding one was reclaimed at
problem 14, all fourteen were lost — while a complete-looking artifact from an earlier,
shorter run sat at the output path, ready to be mistaken for it.

These tests cover the two things that make resume trustworthy rather than merely convenient:
that an interrupted run really does continue instead of redoing work, and that it refuses to
continue into records made with different settings. The second matters more. Silently mixing
a 400-token measurement with a 768-token one would produce a single artifact describing
neither, which is this repository's characteristic failure — a plausible number with nothing
able to object.

torch and transformers are stubbed, so the run costs milliseconds and the sampling is
deterministic without a model.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))


class FakeSeq(list):
    """A generated sequence: sliceable, and convertible back to a list of ids."""

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        return FakeSeq(result) if isinstance(item, slice) else result

    def tolist(self):
        return list(self)


class FakeInputIds:
    def __init__(self, length):
        self.shape = (1, length)


EOS = 99


class FakeTokenizer:
    """Emits "The answer is <gold>." for every sample, then EOS."""

    eos_token_id = EOS

    def __init__(self):
        self.answer_for_call = []

    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors=None):
        # The question text is carried through so generate() can answer it correctly.
        self.last_text = text
        return {"input_ids": FakeInputIds(4)}

    def decode(self, ids, skip_special_tokens=True):
        # ids are [marker, digit..., EOS]; rebuild the number they encode.
        digits = "".join(str(i) for i in ids if 0 <= i <= 9)
        return f"The answer is {digits}."


class FakeModel:
    #: problems whose generate() call should kill the process. KeyboardInterrupt, not a
    #: plain Exception: the tool catches Exception per problem, records it in `errors` and
    #: carries on, so an ordinary failure is NOT how a run dies. What killed the real run
    #: was the container being reclaimed — a SIGKILL, with nothing caught and nothing
    #: written after it. A BaseException escaping the loop is the closest stand-in.
    kill_on_gold = set()
    #: every gold value generate() was called for, in order — proof of what was redone.
    calls = []

    @classmethod
    def from_pretrained(cls, *_a, **_k):
        return cls()

    def eval(self):
        return self

    def generate(self, input_ids=None, num_return_sequences=1, **_kw):
        gold = int(FakeModel.current_gold)
        FakeModel.calls.append(gold)
        if gold in FakeModel.kill_on_gold:
            raise KeyboardInterrupt(f"simulated container reclaim on gold {gold}")
        digits = [int(c) for c in str(gold)]
        return [FakeSeq([0, 0, 0, 0] + digits + [EOS]) for _ in range(num_return_sequences)]


@pytest.fixture
def reference(monkeypatch):
    """Install the stubs and return the tool's module."""
    torch = types.ModuleType("torch")
    torch.float32 = "float32"
    torch.manual_seed = lambda _seed: None

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_a):
            return False

    torch.no_grad = _NoGrad
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    transformers.AutoModelForCausalLM = FakeModel

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    import benchmark_quality_reference as mod  # noqa: PLC0415

    FakeModel.kill_on_gold = set()
    FakeModel.calls = []

    # The tool formats the prompt from problem["question"]; carry the gold through it so the
    # stub can answer, and so each problem is distinguishable in FakeModel.calls.
    original = mod.PROMPT

    def fake_format(question):
        FakeModel.current_gold = question
        return question

    monkeypatch.setattr(mod, "PROMPT", types.SimpleNamespace(format=lambda question: fake_format(question)))
    yield mod
    mod.PROMPT = original


def write_dataset(tmp_path, golds):
    path = tmp_path / "set.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for gold in golds:
            f.write(json.dumps({"question": str(gold), "answer": f"#### {gold}"}) + "\n")
    return path


#: what run() returns as the exit code when the process was killed rather than returning.
KILLED = "killed"


def run(mod, tmp_path, dataset, extra=()):
    """Run the tool. Returns (exit code, artifact path), or (KILLED, path) if it was killed."""
    out = tmp_path / "artifact.json"
    argv = ["prog", "--dataset", str(dataset), "--limit", "3", "--samples", "2",
            "--max-tokens", "16", "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        code = mod.main()
    except KeyboardInterrupt:
        code = KILLED
    finally:
        sys.argv = old
    return code, out


def test_a_checkpoint_is_written_after_every_problem(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    code, out = run(reference, tmp_path, dataset)
    assert code == 0
    # On success the checkpoint is removed, because the full artifact is now on disk.
    assert not Path(str(out) + ".partial").exists()
    assert json.loads(out.read_text())["comparison"]["n_problems"] == 3


def test_an_interrupted_run_keeps_the_problems_it_finished(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}          # dies on the third problem
    code, out = run(reference, tmp_path, dataset)
    assert code == KILLED
    assert not out.exists(), "a killed run writes no artifact, so the checkpoint is all there is"
    partial = Path(str(out) + ".partial")
    assert partial.exists(), "the first two problems must survive the failure"
    saved = json.loads(partial.read_text())
    assert [r["problem_index"] for r in saved["records"]] == [0, 1]
    assert saved["fingerprint"]["max_new_tokens"] == 16


def test_resuming_does_not_redo_finished_problems(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}
    code, _ = run(reference, tmp_path, dataset)
    assert code == KILLED
    assert FakeModel.calls == [11, 22, 33], "it reached the third problem before dying"

    # Second leg: the failure is gone, and the finished problems must not be regenerated.
    FakeModel.kill_on_gold = set()
    FakeModel.calls = []
    code, out = run(reference, tmp_path, dataset)
    assert code == 0
    assert FakeModel.calls == [33], (
        f"only the unfinished problem should be generated, got {FakeModel.calls}")
    artifact = json.loads(out.read_text())
    assert artifact["comparison"]["n_problems"] == 3
    assert [r["problem_index"] for r in artifact["records"]] == [0, 1, 2]


def test_elapsed_time_carries_across_a_resume(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}
    run(reference, tmp_path, dataset)
    partial = Path(str(tmp_path / "artifact.json") + ".partial")

    # The stubs run in microseconds, so comparing the second leg against the first proves
    # nothing — both are ~0 and the assertion passes whether or not the time carries. Write
    # a first leg that plainly took two hours, so the final artifact has to include it.
    saved = json.loads(partial.read_text())
    saved["elapsed_seconds"] = 7200.0
    partial.write_text(json.dumps(saved), encoding="utf-8")

    FakeModel.kill_on_gold = set()
    code, out = run(reference, tmp_path, dataset)
    assert code == 0
    # An artifact whose elapsed_seconds describes only the last leg reports a two-hour run
    # as a two-second one, which is a claim about throughput, not just a cosmetic field.
    assert json.loads(out.read_text())["elapsed_seconds"] >= 7200.0


def test_elapsed_time_accumulates_across_two_resumes(reference, tmp_path):
    """The checkpoint's own elapsed field has to carry too, not just the final artifact's.

    A run killed twice writes a second checkpoint from the first one's total. If only the
    final artifact accumulated, the first leg's hours would be dropped the moment a second
    kill happened — and a long run is exactly the kind that gets killed more than once.
    """
    dataset = write_dataset(tmp_path, [11, 22, 33, 44])
    partial = Path(str(tmp_path / "artifact.json") + ".partial")

    # Leg 1: dies on the third problem.
    FakeModel.kill_on_gold = {33}
    code, _ = run(reference, tmp_path, dataset, extra=("--limit", "4"))
    assert code == KILLED
    saved = json.loads(partial.read_text())
    assert [r["problem_index"] for r in saved["records"]] == [0, 1]

    # Pretend leg 1 took two hours.
    saved["elapsed_seconds"] = 7200.0
    partial.write_text(json.dumps(saved), encoding="utf-8")

    # Leg 2: gets past the third problem, then dies on the fourth.
    FakeModel.kill_on_gold = {44}
    code, _ = run(reference, tmp_path, dataset, extra=("--limit", "4"))
    assert code == KILLED
    saved = json.loads(partial.read_text())
    assert [r["problem_index"] for r in saved["records"]] == [0, 1, 2]
    assert saved["elapsed_seconds"] >= 7200.0, (
        "the second checkpoint must include the first leg's time, or a run killed twice "
        "forgets how long it has been going")

    # Leg 3: finishes.
    FakeModel.kill_on_gold = set()
    code, out = run(reference, tmp_path, dataset, extra=("--limit", "4"))
    assert code == 0
    artifact = json.loads(out.read_text())
    assert artifact["comparison"]["n_problems"] == 4
    assert artifact["elapsed_seconds"] >= 7200.0


def test_a_checkpoint_from_different_settings_is_refused(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}
    run(reference, tmp_path, dataset)

    FakeModel.kill_on_gold = set()
    # Same everything but the token budget: resuming would mix two measurements.
    code, out = run(reference, tmp_path, dataset, extra=("--max-tokens", "999"))
    assert code == 1, "a mismatched checkpoint must refuse, not merge"
    assert not out.exists(), "no artifact may be written from a refused resume"


@pytest.mark.parametrize("flag,value", [
    ("--samples", "4"),
    ("--seed", "7"),
    ("--temperature", "0.1"),
    ("--limit", "2"),
])
def test_every_setting_in_the_fingerprint_refuses_a_mismatch(reference, tmp_path, flag, value):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}
    run(reference, tmp_path, dataset)
    FakeModel.kill_on_gold = set()
    code, _ = run(reference, tmp_path, dataset, extra=(flag, value))
    assert code == 1, f"{flag} differing must refuse the checkpoint"


def test_no_resume_starts_over(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    FakeModel.kill_on_gold = {33}
    run(reference, tmp_path, dataset)

    FakeModel.kill_on_gold = set()
    FakeModel.calls = []
    code, out = run(reference, tmp_path, dataset, extra=("--no-resume",))
    assert code == 0
    assert FakeModel.calls == [11, 22, 33], "--no-resume must regenerate everything"
    assert json.loads(out.read_text())["comparison"]["n_problems"] == 3


def test_an_unreadable_checkpoint_starts_over_rather_than_crashing(reference, tmp_path):
    dataset = write_dataset(tmp_path, [11, 22, 33])
    partial = tmp_path / "artifact.json.partial"
    partial.write_text("{not json", encoding="utf-8")
    code, out = run(reference, tmp_path, dataset)
    assert code == 0
    assert json.loads(out.read_text())["comparison"]["n_problems"] == 3
