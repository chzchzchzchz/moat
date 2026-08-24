#include <iostream>
#include <vector>
#include <chrono>
#include "src/transformer_engine.h"

int main() {
    TransformerConfig config;
    config.n_layers = 1; // Test just 1 layer for speed
    config.n_channels = 1; 
    
    std::cout << "Initializing Engine..." << std::endl;
    MetalTransformerEngine engine(config);
    MetalTransformerEngine draft_engine(config);
    
    std::cout << "Running standard generate (q_len = 1)..." << std::endl;
    std::vector<int32_t> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        GenerationResult res = engine.generate(prompt.data(), prompt.size(), 2, 0.0f, 1.0f);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Standard generate successful! " 
                  << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
                  
        std::cout << "Running Speculative Decoding (parallel prefill q_len=4)..." << std::endl;
        
        start = std::chrono::high_resolution_clock::now();
        GenerationResult spec_res = engine.generateSpeculative(&draft_engine, prompt.data(), prompt.size(), 2, 4, 0.0f, 1.0f);
        end = std::chrono::high_resolution_clock::now();
        
        std::cout << "Speculative generate successful! " 
                  << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
                  
        std::cout << "SUCCESS: Parallel Prefill and Speculative Decoding pipelines execute without crashing!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
