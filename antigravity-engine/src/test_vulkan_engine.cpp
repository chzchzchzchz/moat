#include "vulkan_transformer_engine.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "Testing Vulkan Transformer Engine Pipeline" << std::endl;
    std::cout << "===========================================" << std::endl;

    TransformerConfig config;
    config.n_channels = 8;
    config.vocab_size = 32000;
    config.hidden_dim = 2048;
    config.max_seq_len = 2048;

    VulkanTransformerEngine engine(config);

    // Test Load Model
    std::string model_path = "models/tinyllama/model.safetensors";
    bool loaded = engine.loadWeights(model_path);
    if (loaded) {
        std::cout << "✅ Vulkan Engine Loaded successfully." << std::endl;
    }

    // Generate tokens
    std::vector<int32_t> prompt = {1, 15043, 29892, 1125};
    GenerationResult res = engine.generate(prompt.data(), prompt.size(), 50, 0.7f, 0.9f);
    std::cout << "✅ Generated " << res.channel_tokens.size() << " channels." << std::endl;

    MCTSConfig mcts_cfg;
    mcts_cfg.n_channels = 8;
    MCTSResult mcts_res = engine.generateMCTS(prompt.data(), prompt.size(), mcts_cfg);
    std::cout << "✅ MCTS Tree Expansion complete." << std::endl;
    std::cout << "   Best Score: " << mcts_res.best_score << std::endl;
    std::cout << "   Chunks Expanded: " << mcts_res.chunks_expanded << std::endl;

    engine.sanitizeBuffers();
    return 0;
}
