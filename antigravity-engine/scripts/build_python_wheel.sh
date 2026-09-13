#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_DIR="${PROJECT_ROOT}/scripts/python_package"
DIST_DIR="${PROJECT_ROOT}/dist"

echo "================================================================="
echo "Building Antigravity Engine Platform Wheel (macOS Apple Silicon)"
echo "================================================================="

# Clean prior build artifacts to avoid nested junk in wheels
rm -rf "${PKG_DIR}/build" "${PKG_DIR}"/*.egg-info "${PKG_DIR}/dist" "${DIST_DIR}"/*.whl

mkdir -p "${PKG_DIR}/antigravity_engine/lib" "${PKG_DIR}/antigravity_engine/shaders" "${DIST_DIR}"

# 1. Sync python sources into packaging directory
cp "${PROJECT_ROOT}/src/native_bridge.py" "${PKG_DIR}/antigravity_engine/native_bridge.py"
cp "${PROJECT_ROOT}/tools/license_keygen.py" "${PKG_DIR}/antigravity_engine/license_keygen.py"

# 2. Compile Shaders
echo "[1/4] Ensuring Metal Shaders are compiled..."
xcrun -sdk macosx metal -c "${PROJECT_ROOT}/src/shaders/batched_gemm.metal" -o "${PKG_DIR}/antigravity_engine/shaders/batched_gemm.air"
xcrun -sdk macosx metallib "${PKG_DIR}/antigravity_engine/shaders/batched_gemm.air" -o "${PKG_DIR}/antigravity_engine/shaders/batched_gemm.metallib"
rm -f "${PKG_DIR}/antigravity_engine/shaders/batched_gemm.air"

xcrun -sdk macosx metal -c "${PROJECT_ROOT}/src/shaders/transformer_ops.metal" -o "${PKG_DIR}/antigravity_engine/shaders/transformer_ops.air"
xcrun -sdk macosx metallib "${PKG_DIR}/antigravity_engine/shaders/transformer_ops.air" -o "${PKG_DIR}/antigravity_engine/shaders/transformer_ops.metallib"
rm -f "${PKG_DIR}/antigravity_engine/shaders/transformer_ops.air"

# 3. Compile native dynamic library
echo "[2/4] Compiling native dynamic library..."
clang -O3 -c "${PROJECT_ROOT}/src/monocypher.c" -target arm64-apple-macos12.0 -I"${PROJECT_ROOT}/src" -o "${PKG_DIR}/monocypher.o"
clang -O3 -c "${PROJECT_ROOT}/src/monocypher-ed25519.c" -target arm64-apple-macos12.0 -I"${PROJECT_ROOT}/src" -o "${PKG_DIR}/monocypher_ed.o"

DYLIB_OUT="${PKG_DIR}/antigravity_engine/lib/libantigravity_engine.dylib"

clang++ -std=c++17 -x objective-c++ -O3 -dynamiclib \
  -target arm64-apple-macos12.0 \
  -install_name "@rpath/libantigravity_engine.dylib" \
  -framework Metal -framework Foundation \
  "${PROJECT_ROOT}/src/antigravity_c_api.cpp" \
  "${PROJECT_ROOT}/src/transformer_engine.mm" \
  "${PROJECT_ROOT}/src/antigravity_engine_c.cpp" \
  "${PROJECT_ROOT}/src/gguf_reader.cpp" \
  "${PROJECT_ROOT}/src/config_parser.cpp" \
  "${PROJECT_ROOT}/src/license_verifier.cpp" \
  -x none "${PKG_DIR}/monocypher.o" "${PKG_DIR}/monocypher_ed.o" \
  -I"${PROJECT_ROOT}/src" \
  -o "${DYLIB_OUT}"

rm -f "${PKG_DIR}/monocypher.o" "${PKG_DIR}/monocypher_ed.o"

# Keep project root dylib updated as well
cp "${DYLIB_OUT}" "${PROJECT_ROOT}/libantigravity_engine.dylib"
cp "${DYLIB_OUT}" "${PROJECT_ROOT}/src/libantigravity_engine.dylib"

# 4. Build Platform-Specific Wheel
echo "[3/4] Building platform-specific .whl archive..."
cd "${PKG_DIR}"
python3 setup.py bdist_wheel --dist-dir "${DIST_DIR}"
cd "${PROJECT_ROOT}"

# 5. Checksum & Install Verification
echo "[4/4] Verifying Distribution Wheel..."
for whl in "${DIST_DIR}"/*.whl; do
    echo "  Wheel: ${whl}"
    shasum -a 256 "${whl}"
    echo "  Installing wheel into virtualenv..."
    pip3 install --force-reinstall --no-deps "${whl}"
done

echo "================================================================="
echo "✅ Python platform wheel successfully built and verified!"
echo "================================================================="
