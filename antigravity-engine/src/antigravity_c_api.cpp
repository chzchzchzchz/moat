/* Project Antigravity — C++ Metal Engine Core (Metal Accelerated) */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "antigravity_c_api.h"
#include "transformer_engine.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <random>

struct AntigravityEngineContext {
    AntigravityConfig config;
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> gemmPipeline;
    id<MTLComputePipelineState> dequantPipeline;
    
    // Zero-copy Shared VRAM Buffers
    id<MTLBuffer> bufActivations;
    id<MTLBuffer> bufWeights;
    id<MTLBuffer> bufSuperblocks;
    id<MTLBuffer> bufOutput;
    
    uint64_t totalAllocatedBytes;
    
    // Native C++ Metal Transformer Engine
    MetalTransformerEngine* nativeEngine = nullptr;
};

extern "C" {

AntigravityEngineContext* AntigravityEngineCreate(const AntigravityConfig* config) {
    if (!config) return nullptr;

    AntigravityEngineContext* ctx = new AntigravityEngineContext();
    ctx->config = *config;
    ctx->totalAllocatedBytes = 0;

    if (config->use_metal_gpu) {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (ctx->device) {
            ctx->commandQueue = [ctx->device newCommandQueue];
            
            NSArray<NSString*>* candidatePaths = @[
                @"src/shaders/batched_gemm.metallib",
                @"./src/shaders/batched_gemm.metallib",
                @"antigravity-engine/src/shaders/batched_gemm.metallib",
                @"../src/shaders/batched_gemm.metallib"
            ];
            
            id<MTLLibrary> library = nil;
            NSError* err = nil;
            for (NSString* path in candidatePaths) {
                NSURL* url = [NSURL fileURLWithPath:path];
                library = [ctx->device newLibraryWithURL:url error:&err];
                if (library) break;
            }
            
            if (!library) {
                NSString* shaderSource = @"#include <metal_stdlib>\nusing namespace metal;\nstruct SuperBlock { half scales[8]; uchar packed_nibbles[128]; };\ninline half2 dequantize_nibble_pair(uchar packed_byte, half scale_even, half scale_odd) { int raw_even = int(packed_byte & 0x0F) - 8; int raw_odd = int((packed_byte >> 4) & 0x0F) - 8; return half2(static_cast<half>(raw_even) * scale_even, static_cast<half>(raw_odd) * scale_odd); }\nkernel void dequantize_superblocks_kernel(device const SuperBlock* superblocks [[buffer(0)]], device half* out_weights [[buffer(1)]], uint id [[thread_position_in_grid]]) { device const SuperBlock& sb = superblocks[id]; device half* out_ptr = out_weights + id * 256; for (int byte_idx = 0; byte_idx < 128; byte_idx++) { uchar packed_byte = sb.packed_nibbles[byte_idx]; int elem_even = byte_idx * 2; int elem_odd = elem_even + 1; half scale_even = sb.scales[elem_even / 32]; half scale_odd = sb.scales[elem_odd / 32]; half2 dequantized = dequantize_nibble_pair(packed_byte, scale_even, scale_odd); out_ptr[elem_even] = dequantized.x; out_ptr[elem_odd] = dequantized.y; } }\nkernel void batched_gemm_simdgroup(device const half* activations [[buffer(0)]], device const half* weights [[buffer(1)]], device half* output [[buffer(2)]], constant uint& N_batch [[buffer(3)]], constant uint& K_dim [[buffer(4)]], constant uint& M_dim [[buffer(5)]], uint2 group_id [[threadgroup_position_in_grid]]) { uint row_start = group_id.y * 8; uint col_start = group_id.x * 8; if (row_start >= N_batch || col_start >= M_dim) return; simdgroup_matrix<half, 8, 8> acc_matrix; acc_matrix = simdgroup_matrix<half, 8, 8>(0.0h); for (uint k = 0; k < K_dim; k += 8) { simdgroup_matrix<half, 8, 8> a_tile; simdgroup_matrix<half, 8, 8> b_tile; simdgroup_load(a_tile, activations + row_start * K_dim + k, K_dim); simdgroup_load(b_tile, weights + k * M_dim + col_start, M_dim); simdgroup_multiply_accumulate(acc_matrix, a_tile, b_tile, acc_matrix); } simdgroup_store(acc_matrix, output + row_start * M_dim + col_start, M_dim); }\n";
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                library = [ctx->device newLibraryWithSource:shaderSource options:opts error:&err];
            }
            
            if (library) {
                ctx->library = library;
                id<MTLFunction> gemmFunc = [library newFunctionWithName:@"batched_gemm_simdgroup"];
                if (gemmFunc) {
                    ctx->gemmPipeline = [ctx->device newComputePipelineStateWithFunction:gemmFunc error:&err];
                }
                id<MTLFunction> dequantFunc = [library newFunctionWithName:@"dequantize_superblocks_kernel"];
                if (dequantFunc) {
                    ctx->dequantPipeline = [ctx->device newComputePipelineStateWithFunction:dequantFunc error:&err];
                }
            }
        }
    }

    size_t actBytes = config->n_channels * config->hidden_dim * sizeof(uint16_t);
    size_t weightBytes = config->hidden_dim * config->vocab_size * sizeof(uint16_t);
    size_t outBytes = config->n_channels * config->vocab_size * sizeof(uint16_t);

    if (ctx->device) {
        ctx->bufActivations = [ctx->device newBufferWithLength:actBytes options:MTLResourceStorageModeShared];
        ctx->bufWeights     = [ctx->device newBufferWithLength:weightBytes options:MTLResourceStorageModeShared];
        ctx->bufOutput      = [ctx->device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];

        ctx->totalAllocatedBytes += (actBytes + weightBytes + outBytes);
    }

    return ctx;
}

void AntigravityEngineDestroy(AntigravityEngineContext* ctx) {
    if (ctx) {
        if (ctx->nativeEngine) {
            delete ctx->nativeEngine;
            ctx->nativeEngine = nullptr;
        }
        ctx->bufActivations = nil;
        ctx->bufWeights = nil;
        ctx->bufOutput = nil;
        ctx->gemmPipeline = nil;
        ctx->dequantPipeline = nil;
        ctx->library = nil;
        ctx->commandQueue = nil;
        ctx->device = nil;
        delete ctx;
    }
}

int32_t AntigravityEngineLoadModel(AntigravityEngineContext* ctx, const char* model_path) {
    if (!ctx || !model_path) return -1;

    if (!ctx->nativeEngine) {
        TransformerConfig t_cfg;
        t_cfg.n_channels = ctx->config.n_channels;
        t_cfg.vocab_size = ctx->config.vocab_size > 0 ? ctx->config.vocab_size : 32000;
        t_cfg.hidden_dim = ctx->config.hidden_dim > 0 ? ctx->config.hidden_dim : 2048;
        t_cfg.max_seq_len = ctx->config.max_seq_len > 0 ? ctx->config.max_seq_len : 2048;
        ctx->nativeEngine = new MetalTransformerEngine(t_cfg);
    }

    bool ok = ctx->nativeEngine->loadWeights(std::string(model_path));
    return ok ? 0 : -1;
}

int32_t AntigravityEngineGenerateRollouts(
    AntigravityEngineContext* ctx,
    const float* weight_matrix,
    int32_t max_steps,
    float temperature,
    int32_t* out_tokens,
    AntigravityRolloutResult* out_result
) {
    if (!ctx || !out_tokens || !weight_matrix || max_steps <= 0) return -1;

    auto start_time = std::chrono::high_resolution_clock::now();

    int N = ctx->config.n_channels;
    int V = ctx->config.vocab_size;
    int K = ctx->config.hidden_dim;

    if (temperature < 0.001f) temperature = 0.001f;

    // Convert weight_matrix to FP16 in zero-copy shared Metal VRAM buffer
    if (ctx->bufWeights) {
        uint16_t* w_ptr = (uint16_t*)[ctx->bufWeights contents];
        for (size_t i = 0; i < (size_t)(K * V); i++) {
            _Float16 h = (_Float16)weight_matrix[i];
            std::memcpy(&w_ptr[i], &h, sizeof(uint16_t));
        }
    }

    uint16_t* act_ptr = ctx->bufActivations ? (uint16_t*)[ctx->bufActivations contents] : nullptr;
    uint16_t* out_ptr = ctx->bufOutput ? (uint16_t*)[ctx->bufOutput contents] : nullptr;

    // Maintain per-channel hidden state activations initialized via PRNG
    std::vector<float> channel_states(N * K);
    for (int c = 0; c < N; c++) {
        std::mt19937 ch_rng(1337 + c * 997);
        std::normal_distribution<float> ch_dist(0.1f * (c + 1), 0.5f);
        for (int k = 0; k < K; k++) {
            channel_states[c * K + k] = ch_dist(ch_rng);
        }
    }

    int total_tokens = 0;

    static thread_local std::vector<std::mt19937> channel_rngs;
    if (channel_rngs.size() != (size_t)N) {
        channel_rngs.resize(N);
        for (int c = 0; c < N; c++) {
            channel_rngs[c].seed(1337 + c * 10007);
        }
    }

    for (int step = 0; step < max_steps; step++) {
        // Copy channel activations into zero-copy shared Metal GPU buffer
        if (act_ptr) {
            for (size_t i = 0; i < (size_t)(N * K); i++) {
                _Float16 h = (_Float16)channel_states[i];
                std::memcpy(&act_ptr[i], &h, sizeof(uint16_t));
            }
        }

        bool metal_success = false;

        if (ctx->device && ctx->commandQueue && ctx->gemmPipeline && ctx->bufActivations && ctx->bufWeights && ctx->bufOutput) {
            id<MTLCommandBuffer> cmdBuf = [ctx->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:ctx->gemmPipeline];
            [enc setBuffer:ctx->bufActivations offset:0 atIndex:0];
            [enc setBuffer:ctx->bufWeights offset:0 atIndex:1];
            [enc setBuffer:ctx->bufOutput offset:0 atIndex:2];

            uint32_t uintN = (uint32_t)N;
            uint32_t uintK = (uint32_t)K;
            uint32_t uintV = (uint32_t)V;

            [enc setBytes:&uintN length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&uintK length:sizeof(uint32_t) atIndex:4];
            [enc setBytes:&uintV length:sizeof(uint32_t) atIndex:5];

            MTLSize threadgroups = MTLSizeMake((V + 7) / 8, (N + 7) / 8, 1);
            MTLSize threadsPerTG = MTLSizeMake(32, 1, 1);

            [enc dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerTG];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            metal_success = true;
        }

        // Extract logits per channel and perform Softmax temperature categorical sampling
        for (int c = 0; c < N; c++) {
            std::vector<float> logits(V);

            if (metal_success && out_ptr) {
                for (int v = 0; v < V; v++) {
                    uint16_t raw_h = out_ptr[c * V + v];
                    _Float16 h;
                    std::memcpy(&h, &raw_h, sizeof(h));
                    logits[v] = (float)h;
                }
            } else {
                // CPU matrix multiply path fallback
                for (int v = 0; v < V; v++) {
                    float logit = 0.0f;
                    for (int k = 0; k < K; k++) {
                        logit += channel_states[c * K + k] * weight_matrix[k * V + v];
                    }
                    logits[v] = logit;
                }
            }

            float temp = (temperature > 0.001f) ? temperature : 0.001f;
            float max_val = -1e9f;
            for (int v = 0; v < V; v++) {
                logits[v] /= temp;
                if (logits[v] > max_val) max_val = logits[v];
            }

            float sum_exp = 0.0f;
            std::vector<float> probs(V);
            for (int v = 0; v < V; v++) {
                probs[v] = expf(logits[v] - max_val);
                sum_exp += probs[v];
            }
            for (int v = 0; v < V; v++) {
                probs[v] /= sum_exp;
            }

            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int sampled_tok = dist(channel_rngs[c]);

            out_tokens[c * max_steps + step] = sampled_tok;
            total_tokens++;

            // Update channel hidden state for autoregressive step
            for (int k = 0; k < K; k++) {
                channel_states[c * K + k] += weight_matrix[k * V + (sampled_tok % V)] * 0.01f;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Compute best candidate index based on candidate token diversity and entropy
    int best_channel = 0;
    float best_score = -1.0f;
    for (int c = 0; c < N; c++) {
        std::vector<int> token_counts(V, 0);
        for (int s = 0; s < max_steps; s++) {
            int tok = out_tokens[c * max_steps + s];
            if (tok >= 0 && tok < V) {
                token_counts[tok]++;
            }
        }
        float entropy = 0.0f;
        for (int v = 0; v < V; v++) {
            if (token_counts[v] > 0) {
                float p = (float)token_counts[v] / (float)max_steps;
                entropy -= p * logf(p + 1e-10f);
            }
        }
        float max_entropy = logf((float)std::min(max_steps, V));
        float score = (max_entropy > 0) ? entropy / max_entropy : 0.5f;

        if (score > best_score) {
            best_score = score;
            best_channel = c;
        }
    }

    if (out_result) {
        out_result->total_tokens_generated = total_tokens;
        out_result->active_channels = N;
        out_result->execution_wall_time_ms = wall_ms;
        out_result->throughput_tokens_per_sec = (wall_ms > 0) ? (total_tokens / (wall_ms / 1000.0)) : 0.0;
        out_result->best_candidate_channel_idx = best_channel;
        out_result->best_candidate_score = best_score;
    }

    return 0;
}

int32_t AntigravityEngineVerifyCandidates(
    AntigravityEngineContext* ctx,
    const int32_t* candidate_tokens,
    int32_t seq_len,
    float* out_scores
) {
    if (!ctx || !candidate_tokens || !out_scores || seq_len <= 0) return -1;

    int N = ctx->config.n_channels;
    int V = ctx->config.vocab_size;
    int best_channel = 0;
    float max_score = -1.0f;

    for (int c = 0; c < N; c++) {
        std::vector<int> counts(V, 0);
        int unique_toks = 0;
        for (int s = 0; s < seq_len; s++) {
            int32_t tok = candidate_tokens[c * seq_len + s];
            if (tok >= 0 && tok < V) {
                if (counts[tok] == 0) unique_toks++;
                counts[tok]++;
            }
        }
        float entropy = 0.0f;
        for (int v = 0; v < V; v++) {
            if (counts[v] > 0) {
                float p = (float)counts[v] / (float)seq_len;
                entropy -= p * logf(p + 1e-10f);
            }
        }
        float max_entropy = logf((float)std::min(seq_len, V));
        float norm_entropy = (max_entropy > 0.0f) ? (entropy / max_entropy) : 0.5f;
        float score = 0.5f * norm_entropy + 0.5f * ((float)unique_toks / (float)seq_len);
        if (score < 0.0f) score = 0.0f;
        if (score > 1.0f) score = 1.0f;

        out_scores[c] = score;
        if (score > max_score) {
            max_score = score;
            best_channel = c;
        }
    }

    return best_channel;
}

uint64_t AntigravityEngineGetAllocatedMemoryBytes(const AntigravityEngineContext* ctx) {
    if (!ctx) return 0;
    uint64_t total = ctx->totalAllocatedBytes;
    if (ctx->nativeEngine) {
        total += ctx->nativeEngine->getAllocatedBytes();
    }
    return total;
}

void AntigravityEngineSanitizeBuffers(AntigravityEngineContext* ctx) {
    if (!ctx) return;
    if (ctx->nativeEngine) {
        ctx->nativeEngine->sanitizeBuffers();
    }
    if (ctx->bufActivations) {
        memset([ctx->bufActivations contents], 0, [ctx->bufActivations length]);
    }
    if (ctx->bufWeights) {
        memset([ctx->bufWeights contents], 0, [ctx->bufWeights length]);
    }
    if (ctx->bufOutput) {
        memset([ctx->bufOutput contents], 0, [ctx->bufOutput length]);
    }
}

int32_t AntigravityEngineNativeGenerate(
    AntigravityEngineContext* ctx,
    const int32_t* prompt_tokens,
    int32_t prompt_len,
    int32_t max_new_tokens,
    float temperature,
    float top_p,
    int32_t* out_tokens,
    float* out_logprobs,
    int32_t* out_token_counts,
    double* out_ttft_ms,
    double* out_total_ms
) {
    if (!ctx || !prompt_tokens || !out_tokens || prompt_len <= 0 || max_new_tokens <= 0) return -2;
    if (!ctx->nativeEngine) return -1;

    // Delegate to the full 22-layer MetalTransformerEngine::generate()
    GenerationResult gen = ctx->nativeEngine->generate(
        prompt_tokens, prompt_len, max_new_tokens, temperature, top_p
    );

    int N = ctx->config.n_channels;

    // Flatten GenerationResult into C output buffers
    for (int c = 0; c < N; c++) {
        int n_toks = (int)gen.channel_tokens[c].size();
        if (out_token_counts) out_token_counts[c] = n_toks;
        if (out_logprobs)     out_logprobs[c] = gen.channel_logprobs[c];

        // Copy tokens into flat [n_channels * max_new_tokens] buffer
        for (int t = 0; t < max_new_tokens; t++) {
            if (t < n_toks) {
                out_tokens[c * max_new_tokens + t] = gen.channel_tokens[c][t];
            } else {
                out_tokens[c * max_new_tokens + t] = 0;  // pad
            }
        }
    }

    if (out_ttft_ms)  *out_ttft_ms  = gen.ttft_ms;
    if (out_total_ms) *out_total_ms = gen.total_ms;

    return 0;
}

int32_t AntigravityEngineNativeGenerateMultimodal(
    AntigravityEngineContext* ctx,
    const int32_t* text_tokens,
    int32_t text_len,
    const float* image_embeddings,
    int32_t n_image_patches,
    int32_t max_new_tokens,
    float temperature,
    float top_p,
    int32_t* out_tokens,
    float* out_logprobs,
    int32_t* out_token_counts,
    double* out_ttft_ms,
    double* out_total_ms
) {
    if (!ctx || !out_tokens || max_new_tokens <= 0) return -2;
    if (!ctx->nativeEngine) return -1;

    GenerationResult gen = ctx->nativeEngine->generateMultimodal(
        text_tokens, text_len, image_embeddings, n_image_patches, max_new_tokens, temperature, top_p
    );

    int N = ctx->config.n_channels;

    for (int c = 0; c < N; c++) {
        int n_toks = (int)gen.channel_tokens[c].size();
        if (out_token_counts) out_token_counts[c] = n_toks;
        if (out_logprobs)     out_logprobs[c] = gen.channel_logprobs[c];

        for (int t = 0; t < max_new_tokens; t++) {
            if (t < n_toks) {
                out_tokens[c * max_new_tokens + t] = gen.channel_tokens[c][t];
            } else {
                out_tokens[c * max_new_tokens + t] = 0;
            }
        }
    }

    if (out_ttft_ms)  *out_ttft_ms  = gen.ttft_ms;
    if (out_total_ms) *out_total_ms = gen.total_ms;

    return 0;
}

void AntigravityEngineUnloadWeights(AntigravityEngineContext* ctx) {
    if (!ctx) return;

    // Sanitize all GPU buffers (zeroes scratch + KV caches)
    if (ctx->nativeEngine) {
        ctx->nativeEngine->sanitizeBuffers();
    }

    // Release the GEMM activation/weight/output shared buffers
    if (ctx->bufActivations) {
        ctx->totalAllocatedBytes -= [ctx->bufActivations length];
        ctx->bufActivations = nil;
    }
    if (ctx->bufWeights) {
        ctx->totalAllocatedBytes -= [ctx->bufWeights length];
        ctx->bufWeights = nil;
    }
    if (ctx->bufOutput) {
        ctx->totalAllocatedBytes -= [ctx->bufOutput length];
        ctx->bufOutput = nil;
    }
    if (ctx->bufSuperblocks) {
        ctx->totalAllocatedBytes -= [ctx->bufSuperblocks length];
        ctx->bufSuperblocks = nil;
    }
}

bool AntigravityEngineHasWeights(const AntigravityEngineContext* ctx) {
    if (!ctx || !ctx->nativeEngine) return false;
    return ctx->nativeEngine->weightsLoaded_;
}

} // extern "C"


