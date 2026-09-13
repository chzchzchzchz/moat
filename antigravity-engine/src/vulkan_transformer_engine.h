#pragma once
#if defined(USE_VULKAN) && __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#endif
#include "transformer_engine_interface.h"
#include "transformer_engine.h" // for TransformerConfig

// Cross-Platform Native Vulkan Engine implementation
class VulkanTransformerEngine : public ITransformerEngine {
public:
    VulkanTransformerEngine(const TransformerConfig& config);
    ~VulkanTransformerEngine() override;

    bool loadWeights(const std::string& safetensors_path) override;
    void allocateUnifiedMemoryMap() override;
    
    GenerationResult generate(
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    ) override;
    
    GenerationResult generateSpeculative(
        ITransformerEngine* draft_engine,
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        int32_t max_new_tokens,
        int32_t k_draft,
        float temperature,
        float top_p
    ) override;
    
    GenerationResult generateMultimodal(
        const int32_t* text_tokens,
        int32_t text_len,
        const float* image_embeddings,
        int32_t n_image_patches,
        int32_t max_new_tokens,
        float temperature,
        float top_p
    ) override;

    MCTSResult generateMCTS(
        const int32_t* prompt_tokens,
        int32_t prompt_len,
        const MCTSConfig& cfg
    ) override;

    void sanitizeBuffers() override;
    uint64_t getAllocatedBytes() const override;

private:
    TransformerConfig config_;
    uint64_t allocatedBytes_ = 0;
    
    // Vulkan objects would go here (VkInstance, VkDevice, etc.)
    // ...
};
