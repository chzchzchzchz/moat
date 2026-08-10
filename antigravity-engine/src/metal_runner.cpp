#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

// Metal compiled library path
#define METALLIB_PATH "antigravity-engine/src/shaders/batched_gemm.metallib"

struct SuperBlock {
    uint16_t scales[8];            // FP16 scale factors
    uint8_t  packed_nibbles[128];  // 256 packed INT4 values
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    @autoreleasepool {
        std::cout << "=========================================================\n";
        std::cout << "Project Antigravity — Native Metal Compute Shader Runner\n";
        std::cout << "=========================================================\n";

        // 1. Get default Metal GPU device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Error: No Metal GPU device available!\n";
            return 1;
        }
        std::cout << "Metal GPU: " << [[device name] UTF8String] << "\n";

        // 2. Load compiled metallib using non-deprecated URL API
        NSString* libPath = [NSString stringWithUTF8String:METALLIB_PATH];
        NSURL* libURL = [NSURL fileURLWithPath:libPath];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:libURL error:&error];
        if (!library) {
            std::cerr << "Error loading metallib: " << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }

        // 3. Create pipeline states for kernels
        id<MTLFunction> dequantFunc = [library newFunctionWithName:@"dequantize_superblocks_kernel"];
        id<MTLFunction> gemmFunc    = [library newFunctionWithName:@"batched_gemm_simdgroup"];

        id<MTLComputePipelineState> dequantPipeline = [device newComputePipelineStateWithFunction:dequantFunc error:&error];
        id<MTLComputePipelineState> gemmPipeline    = [device newComputePipelineStateWithFunction:gemmFunc error:&error];

        if (!dequantPipeline || !gemmPipeline) {
            std::cerr << "Error creating compute pipelines!\n";
            return 1;
        }

        // 4. Test Matrix Dimensions
        uint32_t N = 8;    // Batch size (8 parallel reasoning traces)
        uint32_t K = 2048; // Input hidden dim
        uint32_t M = 2048; // Output hidden dim

        std::cout << "Matrix Shape: N=" << N << ", K=" << K << ", M=" << M << "\n";

        // 5. Create Page-Aligned Zero-Copy GPU Buffers (MTLResourceStorageModeShared)
        size_t activation_bytes = N * K * sizeof(uint16_t); // FP16
        size_t weight_bytes     = K * M * sizeof(uint16_t); // FP16
        size_t output_bytes     = N * M * sizeof(uint16_t); // FP16
        size_t n_superblocks    = (K * M) / 256;
        size_t superblock_bytes = n_superblocks * sizeof(SuperBlock);

        void* act_mem = nullptr;
        void* sb_mem = nullptr;
        void* w_mem = nullptr;
        void* out_mem = nullptr;

        posix_memalign(&act_mem, 4096, activation_bytes);
        posix_memalign(&sb_mem, 4096, superblock_bytes);
        posix_memalign(&w_mem, 4096, weight_bytes);
        posix_memalign(&out_mem, 4096, output_bytes);

        id<MTLBuffer> buf_activations = [device newBufferWithBytesNoCopy:act_mem length:activation_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_superblocks = [device newBufferWithBytesNoCopy:sb_mem length:superblock_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_weights     = [device newBufferWithBytesNoCopy:w_mem length:weight_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> buf_output      = [device newBufferWithBytesNoCopy:out_mem length:output_bytes options:MTLResourceStorageModeShared deallocator:nil];

        // 6. Initialize synthetic benchmark super-blocks & activations directly in zero-copy host pointers
        SuperBlock* sb_ptr = (SuperBlock*)[buf_superblocks contents];
        for (size_t i = 0; i < n_superblocks; i++) {
            for (int g = 0; g < 8; g++) {
                sb_ptr[i].scales[g] = 0x3800; // FP16 value ~0.5
            }
            for (int b = 0; b < 128; b++) {
                sb_ptr[i].packed_nibbles[b] = 0x77; // INT4 pair (+7, +7)
            }
        }

        uint16_t* act_ptr = (uint16_t*)[buf_activations contents];
        for (size_t i = 0; i < N * K; i++) {
            act_ptr[i] = 0x3C00; // FP16 value 1.0
        }

        // 7. Dispatch Kernel 1: Dequantize Super-Blocks
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        {
            id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

            [encoder setComputePipelineState:dequantPipeline];
            [encoder setBuffer:buf_superblocks offset:0 atIndex:0];
            [encoder setBuffer:buf_weights     offset:0 atIndex:1];

            MTLSize gridSize = MTLSizeMake(n_superblocks, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(std::min((size_t)256, n_superblocks), 1, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
        }

        std::cout << "✅ Dequantization Kernel executed successfully on Metal GPU!\n";

        // 8. Dispatch Kernel 2: Batched GEMM SIMD Group
        MTLSize threadgroups = MTLSizeMake((M + 7) / 8, (N + 7) / 8, 1);
        MTLSize threadsPerTG = MTLSizeMake(32, 1, 1); // 1 SIMD group (32 threads) per threadgroup

        // Benchmark Metal GEMM execution
        auto t0 = std::chrono::high_resolution_clock::now();
        int iterations = 100;
        for (int i = 0; i < iterations; i++) {
            id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:gemmPipeline];
            [enc setBuffer:buf_activations offset:0 atIndex:0];
            [enc setBuffer:buf_weights     offset:0 atIndex:1];
            [enc setBuffer:buf_output      offset:0 atIndex:2];
            [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&K length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&M length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];
            [cmdBuf commit];
            if (i == iterations - 1) {
                [cmdBuf waitUntilCompleted];
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms = total_ms / iterations;
        double per_token_ms = avg_ms / N;

        std::cout << "=========================================================\n";
        std::cout << "  Native Metal Compute Shader GEMM Benchmark\n";
        std::cout << "=========================================================\n";
        std::cout << "  Iterations:         " << iterations << "\n";
        std::cout << "  Avg Total Time:     " << avg_ms << " ms\n";
        std::cout << "  Per-Token Time:     " << per_token_ms << " ms/tok\n";
        std::cout << "  Tokens / Second:    " << (1000.0 / per_token_ms) << " tok/s\n";
        std::cout << "=========================================================\n";
        std::cout << "✅ SIMD Matrix Tile GEMM successfully executed and benchmarked!\n";

        free(act_mem);
        free(sb_mem);
        free(w_mem);
        free(out_mem);
    }
    return 0;
}

