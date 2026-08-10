#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#define METALLIB_PATH "antigravity-engine/src/shaders/batched_gemm.metallib"

struct SuperBlock {
    uint16_t scales[8];
    uint8_t  packed_nibbles[128];
};

// Helper: Convert float to FP16 (uint16_t) representation
uint16_t float_to_fp16(float f) {
    uint32_t x = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return (sign << 15);
    if (exp >= 31) return (sign << 15) | 0x7C00;
    return (sign << 15) | (exp << 10) | mant;
}

// Helper: Convert FP16 (uint16_t) representation to float
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) return 0.0f;
    uint32_t f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    return *reinterpret_cast<float*>(&f);
}

int main() {
    @autoreleasepool {
        std::cout << "=========================================================\n";
        std::cout << "Metal Compute Shader Micro-Unit Test Suite\n";
        std::cout << "=========================================================\n";

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        assert(device != nil && "Metal device required");

        NSString* libPath = @"src/shaders/batched_gemm.metallib";
        if (![[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            libPath = @"antigravity-engine/src/shaders/batched_gemm.metallib";
        }
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        assert(library != nil && "metallib must exist");

        id<MTLFunction> dequantFunc = [library newFunctionWithName:@"dequantize_superblocks_kernel"];
        id<MTLFunction> gemmFunc    = [library newFunctionWithName:@"batched_gemm_simdgroup"];

        id<MTLComputePipelineState> dequantPipeline = [device newComputePipelineStateWithFunction:dequantFunc error:&error];
        id<MTLComputePipelineState> gemmPipeline    = [device newComputePipelineStateWithFunction:gemmFunc error:&error];

        assert(dequantPipeline != nil && gemmPipeline != nil);

        // Test 1: Dequantization Kernel Parity Test
        std::cout << "Running Test 1: Dequantization Kernel Parity... ";
        {
            size_t n_superblocks = 1;
            id<MTLBuffer> buf_sb = [device newBufferWithLength:sizeof(SuperBlock) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_out = [device newBufferWithLength:256 * sizeof(uint16_t) options:MTLResourceStorageModeShared];

            SuperBlock* sb = (SuperBlock*)[buf_sb contents];
            for (int g = 0; g < 8; g++) sb->scales[g] = float_to_fp16(0.5f);
            for (int b = 0; b < 128; b++) sb->packed_nibbles[b] = 0x7F; // even: 7-8 = -1, odd: 15-8 = +7

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:dequantPipeline];
            [enc setBuffer:buf_sb offset:0 atIndex:0];
            [enc setBuffer:buf_out offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* res = (uint16_t*)[buf_out contents];
            float even_val = fp16_to_float(res[0]);
            float odd_val  = fp16_to_float(res[1]);

            // packed_byte = 0x7F: low nibble (even) = 0x0F = 15 -> (15-8)*0.5 = +3.5f
            //                     high nibble (odd) = 0x07 = 7  -> (7-8)*0.5  = -0.5f
            assert(std::abs(even_val - (3.5f)) < 0.05f && "Even element dequant mismatch");
            assert(std::abs(odd_val - (-0.5f)) < 0.05f && "Odd element dequant mismatch");
            std::cout << "PASSED ✅ (even: " << even_val << ", odd: " << odd_val << ")\n";
        }

        // Test 2: SIMD Matrix GEMM Mathematical Parity Test
        std::cout << "Running Test 2: SIMD Matrix GEMM Parity... ";
        {
            uint32_t N = 8, K = 16, M = 16;
            id<MTLBuffer> buf_A = [device newBufferWithLength:N * K * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_B = [device newBufferWithLength:K * M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_C = [device newBufferWithLength:N * M * sizeof(uint16_t) options:MTLResourceStorageModeShared];

            uint16_t* ptr_A = (uint16_t*)[buf_A contents];
            uint16_t* ptr_B = (uint16_t*)[buf_B contents];

            for (size_t i = 0; i < N * K; i++) ptr_A[i] = float_to_fp16(1.0f);
            for (size_t i = 0; i < K * M; i++) ptr_B[i] = float_to_fp16(0.5f);

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:gemmPipeline];
            [enc setBuffer:buf_A offset:0 atIndex:0];
            [enc setBuffer:buf_B offset:0 atIndex:1];
            [enc setBuffer:buf_C offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];

            MTLSize threadgroups = MTLSizeMake((M + 7) / 8, (N + 7) / 8, 1);
            MTLSize threadsPerTG = MTLSizeMake(32, 1, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* ptr_C = (uint16_t*)[buf_C contents];
            float first_val = fp16_to_float(ptr_C[0]);
            // Expected: dot product of 16 ones and 16 halves = 16 * 0.5 = 8.0
            assert(std::abs(first_val - 8.0f) < 0.1f && "GEMM matrix math mismatch");
            std::cout << "PASSED ✅ (C[0,0] = " << first_val << " == expected 8.0)\n";
        }

        // Test 3: Zero-NaN / Inf Safety Test
        std::cout << "Running Test 3: Zero-NaN / Inf Safety... ";
        {
            uint32_t N = 8, K = 32, M = 32;
            id<MTLBuffer> buf_A = [device newBufferWithLength:N * K * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_B = [device newBufferWithLength:K * M * sizeof(uint16_t) options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_C = [device newBufferWithLength:N * M * sizeof(uint16_t) options:MTLResourceStorageModeShared];

            uint16_t* ptr_A = (uint16_t*)[buf_A contents];
            uint16_t* ptr_B = (uint16_t*)[buf_B contents];

            for (size_t i = 0; i < N * K; i++) ptr_A[i] = float_to_fp16((float)(i % 10));
            for (size_t i = 0; i < K * M; i++) ptr_B[i] = float_to_fp16((float)(i % 5) * 0.1f);

            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:gemmPipeline];
            [enc setBuffer:buf_A offset:0 atIndex:0];
            [enc setBuffer:buf_B offset:0 atIndex:1];
            [enc setBuffer:buf_C offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];

            MTLSize threadgroups = MTLSizeMake((M + 7) / 8, (N + 7) / 8, 1);
            MTLSize threadsPerTG = MTLSizeMake(32, 1, 1);
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            uint16_t* ptr_C = (uint16_t*)[buf_C contents];
            for (size_t i = 0; i < N * M; i++) {
                float v = fp16_to_float(ptr_C[i]);
                assert(!std::isnan(v) && !std::isinf(v) && "NaN/Inf in Metal output");
            }
            std::cout << "PASSED ✅ (0 NaN / 0 Inf)\n";
        }

        std::cout << "=========================================================\n";
        std::cout << "ALL 3 METAL COMPUTE SHADER TESTS PASSED 100%! ✅\n";
        std::cout << "=========================================================\n";
    }
    return 0;
}
