#!/usr/bin/env bash
set -euo pipefail

# Directory paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_DIR="${PROJECT_ROOT}/frameworks"

echo "================================================================="
echo "Building Production Antigravity Engine .xcframework (macOS & iOS)"
echo "================================================================="

rm -rf "${BUILD_DIR}" "${OUTPUT_DIR}"
mkdir -p "${BUILD_DIR}/macos" "${BUILD_DIR}/ios" "${BUILD_DIR}/iossimulator" \
         "${BUILD_DIR}/module_cache/macos" "${BUILD_DIR}/module_cache/ios" "${BUILD_DIR}/module_cache/iossimulator" \
         "${OUTPUT_DIR}"

# 1. Compile Metal Compute Shaders to .metallib
echo "[1/6] Compiling Metal Shaders..."
xcrun -sdk macosx metal -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" -c "${PROJECT_ROOT}/src/shaders/batched_gemm.metal" -o "${BUILD_DIR}/batched_gemm.air"
xcrun -sdk macosx metallib "${BUILD_DIR}/batched_gemm.air" -o "${PROJECT_ROOT}/src/shaders/batched_gemm.metallib"

xcrun -sdk macosx metal -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" -c "${PROJECT_ROOT}/src/shaders/transformer_ops.metal" -o "${BUILD_DIR}/transformer_ops.air"
xcrun -sdk macosx metallib "${BUILD_DIR}/transformer_ops.air" -o "${PROJECT_ROOT}/src/shaders/transformer_ops.metallib"

# Helper function to compile a platform slice
compile_slice() {
    local PLATFORM="$1"
    local TARGET="$2"
    local SDK="$3"
    local OUT_DIR="${BUILD_DIR}/${PLATFORM}"
    local SDK_PATH
    SDK_PATH=$(xcrun --sdk "${SDK}" --show-sdk-path)

    echo "  -> Compiling ${PLATFORM} (${TARGET})..."

    # C sources
    clang -O3 -c "${PROJECT_ROOT}/src/monocypher.c" -target "${TARGET}" -isysroot "${SDK_PATH}" -I"${PROJECT_ROOT}/src" -o "${OUT_DIR}/monocypher.o"
    clang -O3 -c "${PROJECT_ROOT}/src/monocypher-ed25519.c" -target "${TARGET}" -isysroot "${SDK_PATH}" -I"${PROJECT_ROOT}/src" -o "${OUT_DIR}/monocypher_ed.o"

    # C++ and ObjC++ sources
    local CPP_FLAGS="-O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path=${BUILD_DIR}/module_cache/${PLATFORM} -target ${TARGET} -isysroot ${SDK_PATH} -I${PROJECT_ROOT}/src"
    
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/transformer_engine.mm" -o "${OUT_DIR}/transformer_engine.o"
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/antigravity_c_api.cpp" -o "${OUT_DIR}/antigravity_c_api.o"
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/antigravity_engine_c.cpp" -o "${OUT_DIR}/antigravity_engine_c.o"
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/gguf_reader.cpp" -o "${OUT_DIR}/gguf_reader.o"
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/config_parser.cpp" -o "${OUT_DIR}/config_parser.o"
    clang++ ${CPP_FLAGS} -c "${PROJECT_ROOT}/src/license_verifier.cpp" -o "${OUT_DIR}/license_verifier.o"

    ar rcs "${OUT_DIR}/libAntigravityEngine.a" \
        "${OUT_DIR}/monocypher.o" "${OUT_DIR}/monocypher_ed.o" \
        "${OUT_DIR}/transformer_engine.o" "${OUT_DIR}/antigravity_c_api.o" "${OUT_DIR}/antigravity_engine_c.o" \
        "${OUT_DIR}/gguf_reader.o" "${OUT_DIR}/config_parser.o" "${OUT_DIR}/license_verifier.o"
}

# 2. Compile for macOS arm64
echo "[2/6] Compiling static library for macOS arm64..."
compile_slice "macos" "arm64-apple-macos12.0" "macosx"

# 3. Compile for iOS arm64 Device
echo "[3/6] Compiling static library for iOS arm64 Device..."
compile_slice "ios" "arm64-apple-ios16.0" "iphoneos"

# 4. Compile for iOS Simulator arm64
echo "[4/6] Compiling static library for iOS Simulator arm64..."
compile_slice "iossimulator" "arm64-apple-ios16.0-simulator" "iphonesimulator"

# 5. Create .xcframework Bundle
echo "[5/6] Packaging .xcframework..."
xcodebuild -create-xcframework \
    -library "${BUILD_DIR}/macos/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -library "${BUILD_DIR}/ios/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -library "${BUILD_DIR}/iossimulator/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -output "${OUTPUT_DIR}/AntigravityEngine.xcframework"

# 6. Sign, Package ZIP & Compute SPM Checksum
echo "[6/6] Code Signing, Packaging & Generating SPM Checksum..."
SIGN_IDENTITY="${SIGN_IDENTITY:--}"
echo "  -> Signing with identity: ${SIGN_IDENTITY}"
codesign --force --timestamp=none --sign "${SIGN_IDENTITY}" "${OUTPUT_DIR}/AntigravityEngine.xcframework"

XCF_ZIP="${OUTPUT_DIR}/AntigravityEngine.xcframework.zip"
ditto -c -k --keepParent "${OUTPUT_DIR}/AntigravityEngine.xcframework" "${XCF_ZIP}"

CHECKSUM=$(swift package compute-checksum "${XCF_ZIP}")

echo "================================================================="
echo "✅ AntigravityEngine.xcframework successfully built and packaged!"
echo "   Framework: ${OUTPUT_DIR}/AntigravityEngine.xcframework"
echo "   Archive:   ${XCF_ZIP}"
echo "   SPM SHA256 Checksum: ${CHECKSUM}"
echo "================================================================="
