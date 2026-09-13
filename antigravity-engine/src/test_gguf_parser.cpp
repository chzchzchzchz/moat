#include "transformer_engine.h"
#include <iostream>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

int main() {
    std::string models_dir = "/Users/MohssineChazi2/moat/models/";
    
    // Find a .gguf file
    std::string gguf_file;
    if (fs::exists(models_dir)) {
        for (const auto& entry : fs::recursive_directory_iterator(models_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                gguf_file = entry.path().string();
                break;
            }
        }
    }
    
    if (!gguf_file.empty()) {
        std::cout << "Testing GGUF parser with: " << gguf_file << "\n";
        try {
            TransformerConfig config = TransformerConfig::fromGGUF(gguf_file);
            config.print();
        } catch (const std::exception& e) {
            std::cerr << "GGUF Error: " << e.what() << "\n";
        }
    } else {
        std::cout << "No .gguf files found in " << models_dir << "\n";
    }
    
    std::cout << "\n";
    
    // Test Safetensors
    std::vector<std::string> safetensors_files = {
        models_dir + "tinyllama/model.safetensors",
        models_dir + "qwen/model_fp16.safetensors"
    };
    
    for (const auto& f : safetensors_files) {
        if (fs::exists(f)) {
            std::cout << "Testing Safetensors parser with: " << f << "\n";
            try {
                TransformerConfig config = TransformerConfig::fromSafetensors(f);
                config.print();
            } catch (const std::exception& e) {
                std::cerr << "Safetensors Error: " << e.what() << "\n";
            }
            std::cout << "\n";
        }
    }
    
    return 0;
}
