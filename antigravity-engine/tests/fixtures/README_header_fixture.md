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
