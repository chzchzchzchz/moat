"""No Metal kernel may tree-reduce over a runtime thread count with a halve-by-shift loop.

    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1)
        if (tid < s) shared[tid] = op(shared[tid], shared[tid + s]);

combines every partial only when the thread count is a power of two. This is the cause
of the degenerate GSM8K run (quality_gsm8k_full_checkpoint.json): softmax_kernel was
written this way and dispatched with min(cur_seq_len, 256) threads, so 247 of the first
256 positions normalised attention with a max and a sum that skipped keys. Emulated on
TinyLlama (tools/experiments/old_softmax_reduction.py) it turns correct step-by-step
answers into word salad and into a run of NaN logits — the two signatures in that
artifact. RMSNorm had the same loop (fixed in 24c3bae, unreachable at the time).

The current kernels reduce with simd_max/simd_sum or with a rounded-up halving loop; this
test keeps the old form from coming back in any shader.
"""
import pathlib
import re

SHADERS = pathlib.Path(__file__).resolve().parents[1] / "src" / "shaders"

# for (uint s = <expr> / 2; s > 0; s >>= 1)  — also matches `>> 1` initialisers and `/= 2` steps.
HALVING_LOOP = re.compile(
    r"for\s*\(\s*(?:uint|int|ushort|uint32_t)?\s*(\w+)\s*=\s*([^;]+?)\s*(?:/\s*2u?|>>\s*1u?)\s*;"
    r"\s*\1\s*>\s*0u?\s*;\s*\1\s*(?:>>=\s*1u?|/=\s*2u?)\s*\)"
)

# A literal power of two (e.g. 256 / 2) is fine: the loop is correct for that count.
POWER_OF_TWO_LITERAL = re.compile(r"^\(?\s*(\d+)u?\s*\)?$")

# The loop as it was in softmax_kernel at 3c5382e, the binary the degenerate run loaded.
OLD_SOFTMAX = """
    for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
        if (tid < s) {
            max_shared[tid] = max(max_shared[tid], max_shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def unsafe_loops(source: str) -> list:
    found = []
    for match in HALVING_LOOP.finditer(strip_comments(source)):
        literal = POWER_OF_TWO_LITERAL.match(match.group(2).strip())
        if literal:
            n = int(literal.group(1))
            if n > 0 and n & (n - 1) == 0:
                continue
        found.append(match.group(0))
    return found


def covered(threads: int) -> int:
    """How many partials the halve-by-shift loop actually folds into shared[0]."""
    sets = [{t} for t in range(threads)]
    s = threads // 2
    while s > 0:
        for t in range(s):
            sets[t] |= sets[t + s]
        s >>= 1
    return len(sets[0])


def test_old_loop_drops_partials_for_non_powers_of_two():
    # Why the detector exists: the loop is right for 9 of the 256 thread counts the old
    # softmax dispatch used, and wrong for the rest.
    wrong = [n for n in range(1, 257) if covered(n) < n]
    assert len(wrong) == 247
    assert covered(256) == 256 and covered(255) == 128 and covered(6) == 4


def test_detector_flags_the_old_softmax_loop():
    assert len(unsafe_loops(OLD_SOFTMAX)) == 1


def test_detector_allows_power_of_two_literals_and_comments():
    assert unsafe_loops("for (uint s = 256 / 2; s > 0; s >>= 1) {}") == []
    assert unsafe_loops("// for (uint s = n / 2; s > 0; s >>= 1)") == []
    assert len(unsafe_loops("for (uint s = 96 / 2; s > 0; s >>= 1) {}")) == 1


def test_no_shader_uses_the_power_of_two_only_reduction():
    sources = sorted(SHADERS.glob("*.metal"))
    assert sources, f"no shaders found under {SHADERS}"
    offenders = {p.name: unsafe_loops(p.read_text(encoding="utf-8")) for p in sources}
    offenders = {name: loops for name, loops in offenders.items() if loops}
    assert not offenders, offenders
