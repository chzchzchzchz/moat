#!/usr/bin/env python3
"""
Generate src/shader_sources.h from src/shaders/*.metal.

Why this exists: scripts/build_xcframework.sh packages only the static library.
Neither the .metal sources nor the compiled .metallib files end up inside the
framework, and MetalTransformerEngine looked for them by path relative to the
process working directory. On a device there is no such directory, so every
lookup failed and the engine fell back to a single inline kernel — leaving
rmsnorm, rope, attention, embedding and the rest as null pipelines.

Embedding the sources in the binary means a shader library can always be built,
whatever ships alongside the executable. A prebuilt .metallib stays the fast
path when one is present and current; this is the floor beneath it.

Run after editing any .metal file:
    python3 scripts/embed_shaders.py
CI regenerates the header and fails if it differs from what is committed.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHADER_DIR = REPO / "src" / "shaders"
OUTPUT = REPO / "src" / "shader_sources.h"
DELIM = "AGMETAL"  # a raw-string delimiter may be at most 16 characters

HEADER = f"""#pragma once
//
// GENERATED FILE — do not edit by hand.
// Regenerate with: python3 scripts/embed_shaders.py
//
// Metal shader sources, embedded so the engine can always build a shader
// library even when no .metal or .metallib file ships beside the executable —
// which is the case inside AntigravityEngine.xcframework, where only the static
// library is packaged. See scripts/embed_shaders.py for the full reason.
//
#include <cstddef>

namespace antigravity {{
namespace shaders {{

struct EmbeddedShader {{
    const char* name;      // file stem, e.g. "batched_gemm"
    const char* source;    // the complete .metal source
}};

"""

FOOTER_TEMPLATE = """
inline const EmbeddedShader* all(size_t* count) {{
    static const EmbeddedShader kShaders[] = {{
{entries}
    }};
    if (count) *count = sizeof(kShaders) / sizeof(kShaders[0]);
    return kShaders;
}}

// Look up an embedded source by file stem ("batched_gemm"), or nullptr.
inline const char* find(const char* name) {{
    if (!name) return nullptr;
    size_t n = 0;
    const EmbeddedShader* list = all(&n);
    for (size_t i = 0; i < n; i++) {{
        const char* a = list[i].name;
        const char* b = name;
        while (*a && *a == *b) {{ a++; b++; }}
        if (*a == '\\0' && *b == '\\0') return list[i].source;
    }}
    return nullptr;
}}

}}  // namespace shaders
}}  // namespace antigravity
"""


def identifier(stem: str) -> str:
    return "k" + "".join(part.capitalize() for part in stem.split("_")) + "Source"


def main() -> int:
    files = sorted(SHADER_DIR.glob("*.metal"))
    if not files:
        print(f"no .metal files under {SHADER_DIR}", file=sys.stderr)
        return 1

    body, entries = [], []
    for path in files:
        source = path.read_text(encoding="utf-8")
        # A raw string literal ends at the delimiter; if the source contained it,
        # the generated header would not compile. Refuse rather than emit that.
        if f'){DELIM}"' in source:
            print(f"{path.name} contains the raw-string delimiter ){DELIM}\"", file=sys.stderr)
            return 1
        name = identifier(path.stem)
        body.append(f'// ---- {path.name} ({len(source)} bytes) ----\n'
                    f'inline const char* const {name} = R"{DELIM}({source}){DELIM}";\n')
        entries.append(f'        {{ "{path.stem}", {name} }},')

    OUTPUT.write_text(
        HEADER + "\n".join(body) + FOOTER_TEMPLATE.format(entries="\n".join(entries)),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(REPO)} from {len(files)} shader(s): "
          + ", ".join(f.name for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
