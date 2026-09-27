#!/usr/bin/env bash
#
# Measure GSM8K accuracy end to end on the native Metal engine, on this Mac.
#
# This is the one step of the engine's story that no machine in CI can do.
# GitHub-hosted macOS runners are arm64 VMs without GPU passthrough —
# MTLCreateSystemDefaultDevice() returns nil there, measured rather than assumed,
# so the engine cannot run at all. Any Apple Silicon Mac can.
#
# What it measures, from one generation call per problem so the comparison is
# paired and costs no more than best-of-N alone:
#
#   baseline   channel 0 on its own   — one sample, the engine without the idea
#   candidate  majority vote over N   — self-consistency across all N channels
#
# The result is a JSON artifact carrying the hardware, the weights file, every
# per-problem record, both conditions' Wilson intervals and an exact paired test.
# It will not report a lift it cannot distinguish from chance, and on a null result
# it prints the sample size that would have been needed.
#
# Usage:
#   scripts/run_quality_benchmark.sh                  # 40 problems, 8 channels
#   PROBLEMS=200 scripts/run_quality_benchmark.sh     # a run that can resolve 10 points
#   INT4=1 scripts/run_quality_benchmark.sh           # 4-bit weights
#
# Everything is overridable by environment variable; see the defaults below.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PROBLEMS="${PROBLEMS:-40}"
CHANNELS="${CHANNELS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
INT4="${INT4:-0}"
MODEL_DIR="${MODEL_DIR:-models/bench}"
MODEL_REPO="${MODEL_REPO:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DATASET="${DATASET:-${PROJECT_ROOT}/../gsm8k_test_set.jsonl}"
OUT="${OUT:-quality_gsm8k.json}"
PYTHON="${PYTHON:-python3}"

say() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------------------
say "Machine"
sysctl -n machdep.cpu.brand_string
sw_vers -productVersion
printf 'memory: %s GB\n' "$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"

# ---------------------------------------------------------------------------
# Check the GPU before downloading 2.2 GB, and verify the INT4 kernels against the
# host reference while we are here — a kernel fault should be reported as itself,
# not as a bad accuracy number an hour later.
say "Metal device and INT4 kernels"
mkdir -p bin build/lib
clang++ -std=c++17 -fobjc-arc -x objective-c++ -Isrc \
  tests/test_metal_int4_gpu.mm \
  -framework Metal -framework Foundation \
  -o bin/test_metal_int4_gpu
if ! ./bin/test_metal_int4_gpu | tee metal_probe.txt; then
  echo "The INT4 kernels disagree with the host reference on this GPU. Stopping:" >&2
  echo "an accuracy number measured through a faulty kernel would be meaningless." >&2
  exit 1
fi
if grep -q SKIPPED metal_probe.txt; then
  echo "No Metal device on this machine, so the engine cannot run here." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
say "Building the engine dylib"
clang -O3 -c src/monocypher.c -target arm64-apple-macos12.0 -Isrc -o build/monocypher.o
clang -O3 -c src/monocypher-ed25519.c -target arm64-apple-macos12.0 -Isrc -o build/monocypher_ed.o
clang++ -std=c++17 -x objective-c++ -O3 -dynamiclib \
  -target arm64-apple-macos12.0 \
  -install_name "@rpath/libantigravity_engine.dylib" \
  -framework Metal -framework Foundation \
  src/antigravity_c_api.cpp \
  src/transformer_engine.mm \
  src/antigravity_engine_c.cpp \
  src/gguf_reader.cpp \
  src/config_parser.cpp \
  src/license_verifier.cpp \
  -x none build/monocypher.o build/monocypher_ed.o \
  -Isrc \
  -o build/lib/libantigravity_engine.dylib
ls -lh build/lib/libantigravity_engine.dylib

# The shaders are embedded in the binary (src/shader_sources.h), so these are only
# a fast path: when present the engine skips a startup compile, which keeps that
# cost out of the TTFT figures.
say "Compiling the shader libraries"
for name in batched_gemm transformer_ops; do
  xcrun -sdk macosx metal -Werror -c "src/shaders/${name}.metal" -o "build/${name}.air"
  xcrun -sdk macosx metallib "build/${name}.air" -o "src/shaders/${name}.metallib"
  echo "  src/shaders/${name}.metallib"
done

# ---------------------------------------------------------------------------
say "Weights"
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
  mkdir -p "${MODEL_DIR}"
  base="https://huggingface.co/${MODEL_REPO}/resolve/main"
  echo "downloading ${MODEL_REPO} into ${MODEL_DIR} (this is a few GB)"
  curl -fL --retry 3 --retry-delay 5 -o "${MODEL_DIR}/tokenizer.json" "$base/tokenizer.json"
  curl -fL --retry 3 --retry-delay 5 -o "${MODEL_DIR}/tokenizer_config.json" "$base/tokenizer_config.json" || true
  curl -fL --retry 3 --retry-delay 5 -o "${MODEL_DIR}/config.json" "$base/config.json" || true
  curl -fL --retry 3 --retry-delay 5 -o "${MODEL_DIR}/model.safetensors" "$base/model.safetensors"
else
  echo "reusing ${MODEL_DIR}/model.safetensors"
fi
ls -lh "${MODEL_DIR}"

"${PYTHON}" -c 'import tokenizers' 2>/dev/null || "${PYTHON}" -m pip install --quiet tokenizers

# ---------------------------------------------------------------------------
# Before spending an hour grading, establish that the engine reads its input at all.
# gsm8k_full_checkpoint.json in this repository is 587 problems of output that does
# not depend on the prompt, graded to 0.3% and written out as a result. An accuracy
# run cannot detect that — every answer is simply wrong and it reads as a weak model.
int4_flag=()
[ "${INT4}" = "1" ] && int4_flag=(--int4)

say "Does the engine's output depend on its input?"
if ! PYTHONPATH=src "${PYTHON}" tools/check_engine_sanity.py \
      --model-dir "${MODEL_DIR}" \
      --dylib build/lib/libantigravity_engine.dylib \
      "${int4_flag[@]}"; then
  echo "Stopping: grading an engine in this state would produce a number that looks" >&2
  echo "like a weak model rather than the fault it is." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
say "Measuring GSM8K accuracy"
set +e
PYTHONPATH=src "${PYTHON}" tools/benchmark_quality.py \
  --model-dir "${MODEL_DIR}" \
  --dataset "${DATASET}" \
  --dylib build/lib/libantigravity_engine.dylib \
  --limit "${PROBLEMS}" \
  --channels "${CHANNELS}" \
  --max-tokens "${MAX_TOKENS}" \
  "${int4_flag[@]}" \
  --out "${OUT}"
quality_status=$?
set -e

# ---------------------------------------------------------------------------
say "Measuring throughput on the same machine"
# Same machine, same session: the accuracy and the tok/s then share one hardware
# profile, which is exactly what the repository's four contradictory committed
# benchmark artifacts lack.
PYTHONPATH=src "${PYTHON}" tools/benchmark_throughput.py \
  --model "${MODEL_DIR}" \
  --tokens 64 \
  --repeats 3 \
  --out throughput.json || echo "throughput run failed; the accuracy result still stands"

# ---------------------------------------------------------------------------
say "Result"
"${PYTHON}" tools/summarize_quality.py "${OUT}" || true
echo
echo "artifacts: ${OUT}, throughput.json, metal_probe.txt"
echo "Commit them, or attach them — they carry the hardware and every per-problem record,"
echo "which is what makes the number checkable rather than quotable."

# A non-zero exit from the harness means some problems failed and were excluded.
# Propagate it: a partial run must not read as a clean one.
exit "${quality_status}"
