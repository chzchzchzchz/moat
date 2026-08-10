#pragma once
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>
#include <cstdint>
#include <random>

// TinyLlama-1.1B architecture constants
struct TransformerConfig {
    int32_t hidden_dim = 2048;
    int32_t intermediate_dim = 5632;
    int32_t n_layers = 22;
    int32_t n_heads = 32;
    int32_t n_kv_heads = 4;
    int32_t head_dim = 64;
    int32_t vocab_size = 32000;
    int32_t max_seq_len = 2048;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int32_t n_channels = 8;  // parallel reasoning channels
};

struct GenerationResult {
    std::vector<std::vector<int32_t>> channel_tokens;  // [n_channels][seq_len]
    std::vector<float> channel_logprobs;  // [n_channels]
    double ttft_ms;   // Time to first token
    double tpot_ms;   // Time per output token (avg)
    double total_ms;  // Total wall time
    int32_t total_tokens;
    int32_t best_channel;
    float best_score;
};

class MetalTransformerEngine {
public:
    MetalTransformerEngine(const TransformerConfig& config);
    ~MetalTransformerEngine();

    // Load real model weights from Safetensors file into Metal GPU buffers
    bool loadWeights(const std::string& safetensors_path);
    
    // Run full autoregressive decode: prompt tokens in, N channels of generated tokens out
    GenerationResult generate(
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    );

    // Run multimodal autoregressive decode (text tokens + vision patch embeddings)
    GenerationResult generateMultimodal(
        const int32_t* text_tokens,
        int32_t text_len,
        const float* image_embeddings,
        int32_t n_patches,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    );
    
    uint64_t getAllocatedBytes() const;
    void sanitizeBuffers();

    // Public state — accessed by C API bridge
    bool weightsLoaded_;

private:
    // Metal device and pipelines
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> gemmLib_;
    id<MTLLibrary> opsLib_;
    
    // Compute pipelines for each kernel
    id<MTLComputePipelineState> gemmPipeline_;
    id<MTLComputePipelineState> gemvPipeline_;
    id<MTLComputePipelineState> rmsnormPipeline_;
    id<MTLComputePipelineState> ropePipeline_;
    id<MTLComputePipelineState> attnScoresPipeline_;
    id<MTLComputePipelineState> softmaxPipeline_;
    id<MTLComputePipelineState> attnValuePipeline_;
    id<MTLComputePipelineState> siluMulPipeline_;
    id<MTLComputePipelineState> residualPipeline_;
    id<MTLComputePipelineState> embedPipeline_;
    id<MTLComputePipelineState> kvAppendPipeline_;
    
    // Model weight buffers (one per layer)
    struct LayerWeights {
        id<MTLBuffer> input_norm;      // [hidden_dim]
        id<MTLBuffer> q_proj;          // [hidden_dim, hidden_dim] = [2048, 2048]
        id<MTLBuffer> k_proj;          // [n_kv_heads*head_dim, hidden_dim] = [256, 2048]
        id<MTLBuffer> v_proj;          // [256, 2048]
        id<MTLBuffer> o_proj;          // [2048, 2048]
        id<MTLBuffer> post_attn_norm;  // [hidden_dim]
        id<MTLBuffer> gate_proj;       // [intermediate_dim, hidden_dim] = [5632, 2048]
        id<MTLBuffer> up_proj;         // [5632, 2048]
        id<MTLBuffer> down_proj;       // [hidden_dim, intermediate_dim] = [2048, 5632]
    };
    
    std::vector<LayerWeights> layerWeights_;
    id<MTLBuffer> embedWeights_;      // [vocab_size, hidden_dim]
    id<MTLBuffer> finalNorm_;         // [hidden_dim]
    id<MTLBuffer> lmHead_;            // [vocab_size, hidden_dim]
    
    // KV cache buffers per layer per channel
    struct KVCache {
        id<MTLBuffer> k_cache;  // [n_kv_heads, max_seq_len, head_dim]
        id<MTLBuffer> v_cache;
    };
    std::vector<std::vector<KVCache>> kvCaches_;  // [n_layers][n_channels]
    
    // RoPE frequency buffers
    id<MTLBuffer> ropeFreqsCos_;  // [max_seq_len, head_dim/2]
    id<MTLBuffer> ropeFreqsSin_;  // [max_seq_len, head_dim/2]
    
    // Scratch buffers for intermediate results
    id<MTLBuffer> scratch1_;  // general purpose [n_channels, hidden_dim]
    id<MTLBuffer> scratch2_;
    id<MTLBuffer> scratch3_;
    id<MTLBuffer> scratchLogits_; // [n_channels, vocab_size]
    id<MTLBuffer> scratchAttn_;   // [n_channels, n_heads, max_seq_len]
    
    TransformerConfig config_;
    uint64_t allocatedBytes_;
    
    // Internal helpers
    void dispatchGEMM(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
    void dispatchRMSNorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> input, id<MTLBuffer> weight, id<MTLBuffer> output, uint32_t batch, uint32_t dim);
    void dispatchRoPE(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> q, id<MTLBuffer> k, uint32_t start_pos, uint32_t batch);
    void forwardLayer(id<MTLComputeCommandEncoder> enc, int layer_idx, id<MTLBuffer> input, id<MTLBuffer> output, int channel, uint32_t seq_pos);
    int32_t sampleToken(const _Float16* logits, int vocab_size, float temperature, float top_p, std::mt19937& rng);
    
    // Safetensors parser
    bool parseSafetensors(const std::string& path);
};
