#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include "antigravity-engine/src/antigravity_c_api.h"
#include <mach/mach.h>

double getPhysFootprint() {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count);
    return (kr == KERN_SUCCESS) ? (double)vm_info.phys_footprint / (1024.0 * 1024.0) : 0.0;
}

int main(int argc, char* argv[]) {
    std::cout << "======================================================================\n";
    std::cout << "📱 Native C++ Engine Logic Test (Hybrid Qwen 3.5 Ready)\n";
    std::cout << "======================================================================\n";
    
    // Parse prompt tokens from argv[1] comma separated
    std::vector<int32_t> promptTokens;
    if (argc > 1) {
        std::stringstream ss(argv[1]);
        std::string item;
        while (std::getline(ss, item, ',')) {
            promptTokens.push_back(std::stoi(item));
        }
    } else {
        promptTokens = {1, 15043, 29892, 1125, 29892, 29871, 313, 29906}; // Default
    }
    
    AntigravityConfig config;
    config.n_channels = 8;
    config.vocab_size = 32000;
    config.hidden_dim = 2048;
    config.max_seq_len = 512;
    config.use_metal_gpu = true;
    
    std::cout << "[Metal Engine] Initializing context... (Loading DeltaNet & MoE shaders)\n";
    AntigravityEngineContext* ctx = AntigravityEngineCreate(&config);
    if (!ctx) {
        std::cerr << "❌ Failed to create context\n";
        return 1;
    }
    
    const char* modelPath = (argc > 2) ? argv[2] : "/Users/MohssineChazi2/moat/models/tinyllama/model_fp16.safetensors";
    std::cout << "[Metal Engine] Loading weights from " << modelPath << "\n";
    
    if (AntigravityEngineLoadModel(ctx, modelPath) != 0) {
        std::cerr << "❌ Failed to load weights\n";
        return 1;
    }
    
    std::cout << "[Metal Engine] Physical Memory (RSS) after load: " << getPhysFootprint() << " MB\n";
    
    int prompt_len = promptTokens.size();
    
    std::vector<int32_t> outTokens(8 * 128, 0); // Max 128 tokens for test
    std::vector<float> outLogprobs(8, 0.0f);
    std::vector<int32_t> outCounts(8, 0);
    double ttft_ms = 0.0, total_ms = 0.0;
    
    std::cout << "[Metal Engine] Executing N=8 generation on Metal GPU...\n";
    AntigravityEngineNativeGenerate(
        ctx,
        promptTokens.data(), prompt_len, 128,
        0.7f, 0.9f,
        outTokens.data(),
        outLogprobs.data(),
        outCounts.data(),
        &ttft_ms, &total_ms
    );
    
    int total_gen = 0;
    for (int i=0; i<8; i++) total_gen += outCounts[i];
    double tok_s = total_gen / (total_ms / 1000.0);
    
    std::cout << "✅ Generation Complete (Hybrid Pipeline Bound)!\n";
    std::cout << "  • Time To First Token: " << ttft_ms << " ms\n";
    std::cout << "  • Total Time: " << total_ms << " ms\n";
    std::cout << "  • Throughput: " << tok_s << " tok/s (N=8)\n";
    std::cout << "  • Peak Memory RSS: " << getPhysFootprint() << " MB\n";
    
    std::cout << "\n--- GENERATED TOKENS ---\n";
    for(int i=0; i<8; i++) {
        std::cout << "PATH " << i << ": ";
        for(int j=0; j<outCounts[i]; j++) {
            std::cout << outTokens[i*128 + j] << ",";
        }
        std::cout << "\n";
    }
    
    AntigravityEngineDestroy(ctx);
    return 0;
}
