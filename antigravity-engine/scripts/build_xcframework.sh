#!/usr/bin/env bash
set -euo pipefail

# Directory paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
OUTPUT_DIR="${PROJECT_ROOT}/frameworks"

echo "================================================================="
echo "Building Antigravity Engine .xcframework (macOS & iOS)"
echo "================================================================="

rm -rf "${BUILD_DIR}" "${OUTPUT_DIR}"
mkdir -p "${BUILD_DIR}/macos" "${BUILD_DIR}/ios" "${BUILD_DIR}/iossimulator" "${BUILD_DIR}/module_cache/macos" "${BUILD_DIR}/module_cache/ios" "${BUILD_DIR}/module_cache/iossimulator" "${OUTPUT_DIR}"

# 1. Compile Metal Compute Shaders to .metallib
echo "[1/5] Compiling Metal Shaders..."
xcrun -sdk macosx metal -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" -c "${PROJECT_ROOT}/src/shaders/batched_gemm.metal" -o "${BUILD_DIR}/batched_gemm.air"
xcrun -sdk macosx metallib "${BUILD_DIR}/batched_gemm.air" -o "${PROJECT_ROOT}/src/shaders/batched_gemm.metallib"

xcrun -sdk macosx metal -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" -c "${PROJECT_ROOT}/src/shaders/transformer_ops.metal" -o "${BUILD_DIR}/transformer_ops.air"
xcrun -sdk macosx metallib "${BUILD_DIR}/transformer_ops.air" -o "${PROJECT_ROOT}/src/shaders/transformer_ops.metallib"

# 2. Compile C++ Engine Core for macOS arm64
echo "[2/5] Compiling static library for macOS arm64..."
clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" \
    -target arm64-apple-macos12.0 \
    -isysroot $(xcrun --sdk macosx --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/transformer_engine.mm" \
    -o "${BUILD_DIR}/macos/transformer_engine.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" \
    -target arm64-apple-macos12.0 \
    -isysroot $(xcrun --sdk macosx --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_c_api.cpp" \
    -o "${BUILD_DIR}/macos/antigravity_c_api.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/macos" \
    -target arm64-apple-macos12.0 \
    -isysroot $(xcrun --sdk macosx --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_engine_c.cpp" \
    -o "${BUILD_DIR}/macos/antigravity_engine_c.o"

ar rcs "${BUILD_DIR}/macos/libAntigravityEngine.a" "${BUILD_DIR}/macos/transformer_engine.o" "${BUILD_DIR}/macos/antigravity_c_api.o" "${BUILD_DIR}/macos/antigravity_engine_c.o"

# 3. Compile C++ Engine Core for iOS arm64 Device
echo "[3/5] Compiling static library for iOS arm64 Device..."
clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/ios" \
    -target arm64-apple-ios16.0 \
    -isysroot $(xcrun --sdk iphoneos --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/transformer_engine.mm" \
    -o "${BUILD_DIR}/ios/transformer_engine.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/ios" \
    -target arm64-apple-ios16.0 \
    -isysroot $(xcrun --sdk iphoneos --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_c_api.cpp" \
    -o "${BUILD_DIR}/ios/antigravity_c_api.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/ios" \
    -target arm64-apple-ios16.0 \
    -isysroot $(xcrun --sdk iphoneos --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_engine_c.cpp" \
    -o "${BUILD_DIR}/ios/antigravity_engine_c.o"

ar rcs "${BUILD_DIR}/ios/libAntigravityEngine.a" "${BUILD_DIR}/ios/transformer_engine.o" "${BUILD_DIR}/ios/antigravity_c_api.o" "${BUILD_DIR}/ios/antigravity_engine_c.o"

# 4. Compile C++ Engine Core for iOS Simulator arm64
echo "[4/5] Compiling static library for iOS Simulator arm64..."
clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/iossimulator" \
    -target arm64-apple-ios16.0-simulator \
    -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/transformer_engine.mm" \
    -o "${BUILD_DIR}/iossimulator/transformer_engine.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/iossimulator" \
    -target arm64-apple-ios16.0-simulator \
    -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_c_api.cpp" \
    -o "${BUILD_DIR}/iossimulator/antigravity_c_api.o"

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc -fmodules-cache-path="${BUILD_DIR}/module_cache/iossimulator" \
    -target arm64-apple-ios16.0-simulator \
    -isysroot $(xcrun --sdk iphonesimulator --show-sdk-path) \
    -I"${PROJECT_ROOT}/src" \
    -c "${PROJECT_ROOT}/src/antigravity_engine_c.cpp" \
    -o "${BUILD_DIR}/iossimulator/antigravity_engine_c.o"

ar rcs "${BUILD_DIR}/iossimulator/libAntigravityEngine.a" "${BUILD_DIR}/iossimulator/transformer_engine.o" "${BUILD_DIR}/iossimulator/antigravity_c_api.o" "${BUILD_DIR}/iossimulator/antigravity_engine_c.o"


# 5. Create .xcframework Bundle
echo "[5/5] Packaging .xcframework..."
xcodebuild -create-xcframework \
    -library "${BUILD_DIR}/macos/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -library "${BUILD_DIR}/ios/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -library "${BUILD_DIR}/iossimulator/libAntigravityEngine.a" -headers "${PROJECT_ROOT}/Sources/CAntigravityEngine/include" \
    -output "${OUTPUT_DIR}/AntigravityEngine.xcframework"

echo "================================================================="
echo "✅ AntigravityEngine.xcframework successfully built at:"
echo "   ${OUTPUT_DIR}/AntigravityEngine.xcframework"
echo "================================================================="
