#!/bin/bash
set -e
echo "🚀 Building Antigravity N=8 Engine + Qwen 3.5 Hybrid Shaders..."

# Compile Metal Shaders (bypassing sandbox caches if needed)
mkdir -p mcache
xcrun -sdk macosx metal -fmodules-cache-path=./mcache -c antigravity-engine/src/shaders/deltanet_forward.metal -o deltanet.air
xcrun -sdk macosx metallib deltanet.air -o antigravity-engine/src/shaders/deltanet.metallib

xcrun -sdk macosx metal -fmodules-cache-path=./mcache -c antigravity-engine/src/shaders/moe_gemm.metal -o moe.air
xcrun -sdk macosx metallib moe.air -o antigravity-engine/src/shaders/moe.metallib

# Compile C++ Engine
clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/transformer_engine.mm -o transformer_engine.o

clang++ -O3 -std=c++17 -x objective-c++ -fobjc-arc \
    -c antigravity-engine/src/antigravity_c_api.cpp -o antigravity_c_api.o

# Compile Swift App CLI
swiftc -O test_swift_app.swift \
    -import-objc-header antigravity-engine/src/antigravity_c_api.h \
    transformer_engine.o \
    antigravity_c_api.o \
    -framework Metal -framework Foundation -lc++ \
    -o swift_app_runner

echo "✅ Build Complete! Executing Hybrid Pipeline..."
./swift_app_runner
