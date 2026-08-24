#include "antigravity_c_api.h"
#include <iostream>

int main() {
    std::cout << "🚀 Testing Vulkan Cross-Platform Engine..." << std::endl;
    
    AntigravityConfig config;
    config.use_metal_gpu = false; // Force Vulkan Engine
    config.n_channels = 1;
    config.hidden_dim = 256;
    
    AntigravityEngineContext* ctx = AntigravityEngineCreate(&config);
    if (!ctx) {
        std::cerr << "Failed to create context!" << std::endl;
        return 1;
    }
    
    // Test Load Model (creates Vulkan engine stub)
    int32_t res = AntigravityEngineLoadModel(ctx, "dummy_path.safetensors");
    if (res == 0) {
        std::cout << "✅ Vulkan Engine Stub Loaded successfully." << std::endl;
    } else {
        std::cerr << "❌ Failed to load model." << std::endl;
        return 1;
    }
    
    // Generate dummy token
    int32_t prompt[] = {1, 2, 3};
    int32_t out_tokens[32];
    
    res = AntigravityEngineNativeGenerate(ctx, prompt, 3, 32, 0.0f, 1.0f, out_tokens, nullptr, nullptr, nullptr, nullptr);
    if (res == 0) {
        std::cout << "✅ Vulkan Generate API successfully routed through abstraction layer." << std::endl;
    } else {
        std::cerr << "❌ Generate failed." << std::endl;
        return 1;
    }
    
    AntigravityEngineDestroy(ctx);
    std::cout << "✅ Cleaned up context." << std::endl;
    
    return 0;
}
