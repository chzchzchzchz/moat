#include <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "No Metal device found" << std::endl;
        return 1;
    }
    
    NSString* source = @R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void rmsnorm_kernel(
            device const half* x [[buffer(0)]],
            device const half* weight [[buffer(1)]],
            device half* out [[buffer(2)]],
            constant uint& dim [[buffer(3)]],
            constant float& eps [[buffer(4)]],
            uint thread_position_in_threadgroup [[thread_position_in_threadgroup]],
            uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
            uint threads_per_threadgroup [[threads_per_threadgroup]]
        ) {
            uint batch_idx = threadgroup_position_in_grid;
            uint tid = thread_position_in_threadgroup;
            
            device const half* x_b = x + batch_idx * dim;
            device half* out_b = out + batch_idx * dim;
            
            threadgroup float sum_sq_shared[1024]; 
            
            float local_sum = 0.0;
            for (uint i = tid; i < dim; i += threads_per_threadgroup) {
                float val = (float)x_b[i];
                local_sum += val * val;
            }
            sum_sq_shared[tid] = local_sum;
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Reduction
            for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sum_sq_shared[tid] += sum_sq_shared[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            float mean_sq = sum_sq_shared[0] / (float)dim;
            float rsqrt_val = rsqrt(mean_sq + eps);
            
            for (uint i = tid; i < dim; i += threads_per_threadgroup) {
                out_b[i] = (half)(((float)x_b[i] * rsqrt_val) * (float)weight[i]);
            }
        }
    )";
    
    NSError* error = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
    if (!lib) {
        std::cerr << "Compile error: " << [[error localizedDescription] UTF8String] << std::endl;
        return 1;
    }
    
    id<MTLFunction> func = [lib newFunctionWithName:@"rmsnorm_kernel"];
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&error];
    
    uint32_t dim = 2048;
    float eps = 1e-5f;
    
    id<MTLBuffer> bufX = [device newBufferWithLength:dim*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufW = [device newBufferWithLength:dim*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufY = [device newBufferWithLength:dim*2 options:MTLResourceStorageModeShared];
    
    _Float16* x = (_Float16*)[bufX contents];
    _Float16* w = (_Float16*)[bufW contents];
    
    // Fill with embedding from PyTorch output
    x[0] = -0.001091; x[1] = 0.001930; x[2] = -0.001663; x[3] = 0.003768; x[4] = 0.001075;
    x[5] = 0.003784; x[6] = 0.000682; x[7] = -0.000172; x[8] = -0.001083; x[9] = -0.000896;
    for (int i=10; i<dim; i++) x[i] = 0.002f; // Dummy values for rest
    
    // Fill weight from PyTorch output
    w[0] = -0.00418; w[1] = 0.00631; w[2] = 0.0698; w[3] = -0.0294; w[4] = -0.00613;
    w[5] = -0.01519; w[6] = 0.00457; w[7] = 0.00415; w[8] = 0.00363; w[9] = 0.00720;
    for (int i=10; i<dim; i++) w[i] = 0.01f;
    
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufW offset:0 atIndex:1];
    [enc setBuffer:bufY offset:0 atIndex:2];
    [enc setBytes:&dim length:4 atIndex:3];
    [enc setBytes:&eps length:4 atIndex:4];
    
    MTLSize tg = MTLSizeMake(256, 1, 1);
    MTLSize grid = MTLSizeMake(1, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    _Float16* y = (_Float16*)[bufY contents];
    std::cout << "y[0] = " << (float)y[0] << std::endl;
    std::cout << "y[1] = " << (float)y[1] << std::endl;
    std::cout << "y[2] = " << (float)y[2] << std::endl;
    
    return 0;
}
