#!/bin/bash
set -e
echo "Checking for glslc..."
if command -v glslc &> /dev/null; then
    echo "Compiling GLSL shaders to SPIR-V..."
    glslc src/shaders/batched_gemm.comp -o src/shaders/batched_gemm.spv
    glslc -DKERNEL_RMSNORM src/shaders/transformer_ops.comp -o src/shaders/rmsnorm.spv
else
    echo "glslc not found. Skipping SPIR-V compilation (expected on macOS without Vulkan SDK)."
fi

echo "Compiling C++ Vulkan test executable..."
clang++ -std=c++17 -O3 -fobjc-arc -x objective-c++ \
    src/test_vulkan_engine.cpp \
    src/antigravity_c_api.cpp \
    src/transformer_engine.mm \
    src/vulkan_transformer_engine.cpp \
    -o test_vulkan -framework Foundation -framework Metal

echo "Running Vulkan test..."
./test_vulkan
