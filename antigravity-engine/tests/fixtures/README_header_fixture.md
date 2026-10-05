# `qwen2.5-0.5b-instruct.header.json`

The JSON header of `Qwen/Qwen2.5-0.5B-Instruct`'s `model.safetensors`, exactly as the
bytes appear in the file after the 8-byte length field. 32,280 bytes, 291 keys: 290
tensors plus `__metadata__`. No weight data — the header is only offsets, shapes and
dtypes, so this is metadata about a public checkpoint and nothing more.

It is here because every other test of the header parser uses a header I wrote myself,
and a parser that only ever sees its author's synthetic input is not tested against
reality. This one has the properties that matter and that I would not have thought to
synthesise: 290 tensors rather than a handful, `__metadata__` as the first key, real
names 40 characters long, and offsets that tile the file exactly with no gaps.

## `qwen2.5-0.5b-instruct.expected.tsv`

What Python's `json` module reports for the same bytes: one line per tensor, sorted by
name, as `name<TAB>dtype<TAB>comma-separated shape<TAB>offset_start<TAB>offset_end`.
Produced by:

```python
import json
j = json.load(open('qwen2.5-0.5b-instruct.header.json'))
for k in sorted(k for k in j if k != '__metadata__'):
    v = j[k]
    print(f"{k}\t{v['dtype']}\t{','.join(str(d) for d in v['shape'])}"
          f"\t{v['data_offsets'][0]}\t{v['data_offsets'][1]}")
```

`tests/test_safetensors_header.cpp` parses the header with
`antigravity::parseSafetensorsHeader` and requires every field of every tensor to match
this file. That makes it a cross-implementation check: the C++ parser has to agree with a
real JSON parser on a real header, not merely with my expectations.

The file's `data_start` is `8 + 32280 = 32288` and its total size is `988097824`, which is
exactly `data_start` plus the largest `offset_end`. Those two numbers are in the test, so
the bounds check is exercised against a real checkpoint's real geometry, including the
inclusive upper boundary.


# `tinyllama-1.1b-chat-v1.0.header.json`

The JSON header of `TinyLlama/TinyLlama-1.1B-Chat-v1.0`'s `model.safetensors` — the
checkpoint `run_full_gsm8k.py` names, and therefore the one that produced
`gsm8k_full_checkpoint.json`, the 587-problem artifact in this repository in which 413
problems are a single repeated character. 23,088 bytes, 202 keys: 201 tensors plus
`__metadata__`. Header bytes only, fetched with an HTTP Range request; no weight data.

It is here because it is the most relevant test input this repository can have. Every
defect fixed in the weight-loading path on this branch was found by reading the code, and
"could this have caused the degenerate run?" is then a question about this specific header.
With the header in the tests, it is a measurement instead of an argument.

What it says:

| property | value | consequence |
|---|---|---|
| dtypes | **all 201 tensors BF16** | the F32 misread cannot have applied |
| `__metadata__` | `{"format":"pt"}`, first key | flat, so the old first-brace skip survives it |
| nested object in metadata | none | the metadata misparse cannot have applied |
| `}` inside a metadata value | none | nor can the string-literal variant |
| shape vs `data_offsets` span | consistent for all 201 | nothing read past a tensor |
| implied file size | 2,200,119,864 | `data_start` + largest `offset_end`, exactly |

So **none of the header-parser or dtype defects fixed on this branch explains that run.**
They are real, and each one fails by producing a model that loads and runs, but this
checkpoint takes none of their paths. The cause of the degenerate run is still unknown, and
item 1 of `NEXT_ON_HARDWARE.md` remains the first thing to do.
