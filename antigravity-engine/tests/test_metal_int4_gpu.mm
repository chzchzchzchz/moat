/*
 * Runs the INT4 kernels on an actual GPU and checks them against the host packer.
 *
 * Everything else verifying this path reproduces the kernels' arithmetic on the
 * CPU — tests/test_int4_gemv_math.cpp computes what gemv_int4_kernel *should*
 * compute. That catches layout and orientation mistakes, but it cannot catch a
 * mistake in the kernel source itself: a wrong barrier, a threadgroup array read
 * in the wrong address space, a lane split that leaves tile cells unwritten. Those
 * only appear when the shader runs.
 *
 * So this uploads real packed super-blocks, runs the kernels, and compares the
 * GPU's output against dequantSuperblockElement() on the host — the same
 * reference, now on the other side of the driver.
 *
 * If no Metal device exists (a runner without a GPU), it says so and exits 0
 * rather than failing: absence of hardware is not a defect in the code. It exits
 * non-zero only when a kernel runs and disagrees.
 *
 * Build and run from antigravity-engine/:
 *   clang++ -std=c++17 -fobjc-arc -x objective-c++ -Isrc \
 *     tests/test_metal_int4_gpu.mm -framework Metal -framework Foundation \
 *     -o bin/test_metal_int4_gpu && ./bin/test_metal_int4_gpu
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "superblock_pack.h"
#include "shader_sources.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace antigravity;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

static uint32_t g_state = 0x0BADC0DEu;
static float nextUnit() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float)(g_state >> 8) / (float)(1u << 24);
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            printf("no Metal device on this machine; nothing to verify here.\n");
            printf("SKIPPED (this is not a failure)\n");
            return 0;
        }
        printf("Metal device: %s\n", device.name.UTF8String);
        printf("  unified memory: %s | max threadgroup: %lu threads\n",
               device.hasUnifiedMemory ? "yes" : "no",
               (unsigned long)device.maxThreadsPerThreadgroup.width);

        NSError* err = nil;
        const char* source = shaders::find("batched_gemm");
        if (!source) { printf("[FAIL]  batched_gemm.metal is not embedded\n"); return 1; }

        id<MTLLibrary> lib =
            [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                 options:[[MTLCompileOptions alloc] init]
                                   error:&err];
        if (!lib) {
            printf("[FAIL]  embedded batched_gemm.metal does not compile: %s\n",
                   err ? err.localizedDescription.UTF8String : "unknown");
            return 1;
        }
        check(true, "the embedded shader source compiles on this device");

        id<MTLCommandQueue> queue = [device newCommandQueue];

        auto makePipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) { printf("[FAIL]  no kernel named %s\n", name.UTF8String); failures++; return nil; }
            NSError* e = nil;
            id<MTLComputePipelineState> pso =
                [device newComputePipelineStateWithFunction:fn error:&e];
            if (!pso) {
                printf("[FAIL]  pipeline for %s failed: %s\n", name.UTF8String,
                       e ? e.localizedDescription.UTF8String : "unknown");
                failures++;
            }
            return pso;
        };

        id<MTLComputePipelineState> gemvPso = makePipeline(@"gemv_int4_kernel");
        id<MTLComputePipelineState> gemmPso = makePipeline(@"fused_batched_gemm_int4");
        check(gemvPso != nil, "gemv_int4_kernel builds a pipeline");
        check(gemmPso != nil, "fused_batched_gemm_int4 builds a pipeline");
        if (!gemvPso || !gemmPso) {
            printf("\nTHERE WERE FAILURES\n");
            return 1;
        }

        // Shapes the engine uses, scaled down. K and N multiples of 8, K*N a
        // multiple of 256 so the weights pack exactly.
        const uint32_t K = 256, N = 64, M = 8;

        std::vector<uint16_t> B(K * N), X(M * K);
        for (auto& v : B) v = float_to_fp16_bits((nextUnit() - 0.5f) * 0.1f);
        for (auto& v : X) v = float_to_fp16_bits((nextUnit() - 0.5f) * 2.0f);

        std::vector<uint8_t> packed(superblockBytesFor(K * N));
        if (!packSuperblocks(B.data(), B.size(), packed.data())) {
            printf("[FAIL]  host packer refused the weight matrix\n");
            return 1;
        }

        id<MTLBuffer> bufB = [device newBufferWithBytes:packed.data()
                                                length:packed.size()
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [device newBufferWithBytes:X.data()
                                                length:X.size() * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];

        // ---- gemv_int4_kernel: the decode path (M = 1) ----------------------
        {
            id<MTLBuffer> bufY = [device newBufferWithLength:N * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:gemvPso];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufY offset:0 atIndex:2];
            uint32_t uK = K, uN = N;
            [enc setBytes:&uK length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&uN length:sizeof(uint32_t) atIndex:4];
            NSUInteger tg = std::min<NSUInteger>(N, gemvPso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(N, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check(cmd.status == MTLCommandBufferStatusCompleted,
                  "the gemv_int4_kernel command buffer completes");

            // Host reference, reading the same bytes with the same index arithmetic.
            const uint16_t* y = (const uint16_t*)[bufY contents];
            double worst_rel = 0.0;
            for (uint32_t col = 0; col < N; col++) {
                double expected = 0.0;
                for (uint32_t k = 0; k < K; k++) {
                    expected += (double)fp16_bits_to_float(X[k])
                              * dequantSuperblockElement(packed.data(), k * N + col);
                }
                const double got = fp16_bits_to_float(y[col]);
                const double scale = std::fabs(expected) > 1e-6 ? std::fabs(expected) : 1e-6;
                worst_rel = std::max(worst_rel, std::fabs(got - expected) / scale);
            }
            // The kernel accumulates in float and stores half, so the only loss is
            // the FP16 output: 11 significand bits, 2^-11 = 4.9e-4. Anything larger
            // is not rounding.
            printf("       gemv worst relative deviation from the host reference: %.2e\n",
                   worst_rel);
            check(worst_rel < 2e-3,
                  "GPU gemv_int4_kernel matches the host reference to FP16 output precision");
        }

        // ---- fused_batched_gemm_int4: the 8-channel path (M = 8) ------------
        {
            id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:gemmPso];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufB offset:0 atIndex:1];
            [enc setBuffer:bufC offset:0 atIndex:2];
            uint32_t uN = M, uK = K, uM = N;       // N_batch, K_dim, M_dim
            [enc setBytes:&uN length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&uK length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&uM length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            check(cmd.status == MTLCommandBufferStatusCompleted,
                  "the fused_batched_gemm_int4 command buffer completes");

            const uint16_t* C = (const uint16_t*)[bufC contents];
            double sum_sq_err = 0.0, sum_sq_ref = 0.0;
            bool any_nan = false, all_zero = true;
            for (uint32_t row = 0; row < M; row++) {
                for (uint32_t col = 0; col < N; col++) {
                    double expected = 0.0;
                    for (uint32_t k = 0; k < K; k++) {
                        expected += (double)fp16_bits_to_float(X[row * K + k])
                                  * dequantSuperblockElement(packed.data(), k * N + col);
                    }
                    const double got = fp16_bits_to_float(C[row * N + col]);
                    if (std::isnan(got)) any_nan = true;
                    if (got != 0.0) all_zero = false;
                    sum_sq_err += (got - expected) * (got - expected);
                    sum_sq_ref += expected * expected;
                }
            }
            const double rel_rms = std::sqrt(sum_sq_err / sum_sq_ref);
            printf("       fused GEMM relative RMS vs the host reference: %.2e\n", rel_rms);

            check(!any_nan, "no output element is NaN");
            check(!all_zero, "the kernel wrote something (a silent no-op would pass a tolerance test)");
            // This kernel accumulates in simdgroup_matrix<half>, so error compounds
            // over K/8 = 32 accumulation steps rather than being a single rounding.
            // 1% is loose for FP16 accumulation at this depth and still four orders
            // of magnitude tighter than an indexing or barrier mistake would give:
            // the transposed-packing control in test_int4_gemv_math.cpp measures 1.44.
            check(rel_rms < 1e-2,
                  "GPU fused_batched_gemm_int4 matches the host reference within FP16 accumulation error");

            // Every row must differ: one row leaking into another would mean the
            // 8x8 tile indexing is wrong, and an average-based check would hide it.
            bool rows_distinct = true;
            for (uint32_t r = 1; r < M; r++) {
                bool same = true;
                for (uint32_t col = 0; col < N; col++) {
                    if (C[r * N + col] != C[col]) { same = false; break; }
                }
                if (same) rows_distinct = false;
            }
            check(rows_distinct, "each of the 8 rows has its own result");
        }

        printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
        return failures == 0 ? 0 : 1;
    }
}
