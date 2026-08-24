#include "vulkan_transformer_engine.h"
#include <iostream>

VulkanTransformerEngine::VulkanTransformerEngine(const TransformerConfig& config) : config_(config) {
    std::cout << "[VulkanTransformerEngine] Initialized Stub" << std::endl;
}

VulkanTransformerEngine::~VulkanTransformerEngine() {}

bool VulkanTransformerEngine::loadWeights(const std::string& safetensors_path) {
    weightsLoaded_ = true;
    return true;
}

void VulkanTransformerEngine::allocateDummyWeights() {
    weightsLoaded_ = true;
}

GenerationResult VulkanTransformerEngine::generate(
    const int32_t* prompt_tokens, int32_t prompt_len, int32_t max_new_tokens, float temperature, float top_p) {
    GenerationResult res;
    res.channel_tokens.resize(config_.n_channels);
    res.channel_logprobs.resize(config_.n_channels, 0.0f);
    return res;
}

GenerationResult VulkanTransformerEngine::generateSpeculative(
    ITransformerEngine* draft_engine, const int32_t* prompt_tokens, int32_t prompt_len, int32_t max_new_tokens, int32_t k_draft, float temperature, float top_p) {
    GenerationResult res;
    res.channel_tokens.resize(config_.n_channels);
    res.channel_logprobs.resize(config_.n_channels, 0.0f);
    return res;
}

GenerationResult VulkanTransformerEngine::generateMultimodal(
    const int32_t* text_tokens, int32_t text_len, const float* image_embeddings, int32_t n_image_patches, int32_t max_new_tokens, float temperature, float top_p) {
    GenerationResult res;
    res.channel_tokens.resize(config_.n_channels);
    res.channel_logprobs.resize(config_.n_channels, 0.0f);
    return res;
}

MCTSResult VulkanTransformerEngine::generateMCTS(
    const int32_t* prompt_tokens, int32_t prompt_len, const MCTSConfig& cfg) {
    return MCTSResult();
}

void VulkanTransformerEngine::sanitizeBuffers() {}

uint64_t VulkanTransformerEngine::getAllocatedBytes() const {
    return allocatedBytes_;
}
