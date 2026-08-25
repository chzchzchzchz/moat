#include "antigravity-engine/src/transformer_engine.h"
#include <iostream>
#include <iomanip>

int main() {
    TransformerConfig config;
    config.n_channels = 1;
    MetalTransformerEngine engine(config);
    if (!engine.loadWeights("models/tinyllama/model.safetensors")) {
        std::cerr << "Failed to load weights!" << std::endl;
        return 1;
    }
    
    int32_t prompt_tokens[] = {1, 450, 7483, 310, 3444, 338};
    int prompt_len = 6;
    
    GenerationResult res = engine.generate(prompt_tokens, prompt_len, 1, 0.001f, 1.0f);
    std::cout << "Generated token 0: " << res.channel_tokens[0][0] << std::endl;
    
    return 0;
}
