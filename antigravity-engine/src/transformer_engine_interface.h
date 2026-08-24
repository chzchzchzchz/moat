#ifndef TRANSFORMER_ENGINE_INTERFACE_H
#define TRANSFORMER_ENGINE_INTERFACE_H

#include <vector>
#include <string>
#include <stdint.h>

struct GenerationResult {
    std::vector<std::vector<int32_t>> channel_tokens; // [n_channels][tokens]
    std::vector<float> channel_logprobs;              // [n_channels]
    double ttft_ms = 0;
    double tpot_ms = 0;
    double total_ms = 0;
    int total_tokens = 0;
    int best_channel = 0;
    float best_score = 0;
};

struct MCTSConfig {
    int chunk_tokens = 30;
    int num_chunks = 4;
    int branches_per_chunk = 3;
    float temperature = 0.8f;
    float top_p = 0.9f;
};

struct MCTSResult {
    std::vector<int32_t> best_tokens;
    float best_score = 0.0f;
    double total_ms = 0.0;
    int chunks_expanded = 0;
    int total_tokens_evaluated = 0;
};

class ITransformerEngine {
public:
    virtual ~ITransformerEngine() = default;

    virtual bool loadWeights(const std::string& safetensors_path) = 0;
    
    // For pure compute benchmarking
    virtual void allocateDummyWeights() = 0;
    
    virtual GenerationResult generate(
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    ) = 0;
    
    virtual GenerationResult generateSpeculative(
        ITransformerEngine* draft_engine,
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        int32_t max_new_tokens,
        int32_t k_draft,
        float temperature,
        float top_p
    ) = 0;
    
    virtual GenerationResult generateMultimodal(
        const int32_t* text_tokens,
        int32_t text_len,
        const float* image_embeddings,
        int32_t n_image_patches,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    ) = 0;

    virtual MCTSResult generateMCTS(
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        const MCTSConfig& cfg
    ) = 0;

    virtual void sanitizeBuffers() = 0;
    
    virtual uint64_t getAllocatedBytes() const = 0;
    
    bool weightsLoaded_ = false;
};

#endif // TRANSFORMER_ENGINE_INTERFACE_H
