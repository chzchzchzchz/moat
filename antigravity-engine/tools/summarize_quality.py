#!/usr/bin/env python3
"""
Print the headline of a quality_gsm8k.json in a few lines.

Kept as a file rather than inlined in the workflow so the summary can be run
against any artifact afterwards, and so the workflow has no indented heredoc —
which does not terminate and silently swallows the rest of the step.

    python3 tools/summarize_quality.py quality_gsm8k.json
"""

import json
import sys
from pathlib import Path


def main(argv) -> int:
    path = Path(argv[1] if len(argv) > 1 else "quality_gsm8k.json")
    if not path.is_file():
        print(f"no artifact at {path}; the run did not get far enough to write one")
        return 1

    d = json.loads(path.read_text(encoding="utf-8"))
    c = d["comparison"]
    hw = d.get("hardware", {})

    print(f"hardware   {hw.get('chip') or hw.get('platform', 'unknown')}")
    print(f"weights    {d['config'].get('weights_file', '?')}")
    print(f"int4       {d['config'].get('int4_weights')}")
    print(f"channels   {d['config'].get('channels')}")
    print(f"graded     {c['n_problems']} problems in {d.get('elapsed_seconds', 0):.0f}s")
    for name in ("baseline", "candidate"):
        s = c[name]
        low, high = s["ci95"]
        print(f"{name:10} {s['correct']}/{c['n_problems']} = {s['accuracy'] * 100:.1f}%"
              f"  [{low * 100:.1f}, {high * 100:.1f}]")
    p = c["paired"]
    print(f"disagreed  candidate-only {p['only_candidate_correct']}, "
          f"baseline-only {p['only_baseline_correct']}, "
          f"same {p['both_or_neither']}")
    print(f"verdict    {c['verdict']}")
    if "power_note" in d:
        note = d["power_note"]
        disc = note.get("observed_discordance")
        if disc is not None:
            print(f"power      at the {disc * 100:.0f}% discordance this run showed, "
                  f"under {note.get('test', 'the paired test')}:")
            for effect in (5, 10, 15):
                need = note.get(f"problems_needed_for_{effect}_point_effect")
                if need is not None:
                    shown = f"{need} problems" if isinstance(need, int) else str(need)
                    print(f"             a {effect}-point effect needs {shown}")
            print(f"             this run had {note['problems_run']}")
        else:
            print(f"power      this run had {note['problems_run']} problems")
    if d.get("errors"):
        print(f"errors     {len(d['errors'])} problem(s) failed and were excluded")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
