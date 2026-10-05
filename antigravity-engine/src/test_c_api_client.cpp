/*
 * Project Antigravity — C++ API Integration Test Client
 */

#include "antigravity_c_api.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <cstdlib>

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  Project Antigravity — C++ / Swift Core SDK Test Client\n";
    std::cout << "=========================================================\n";

    AntigravityConfig config;
    config.n_channels = 8;
    config.vocab_size = 1000;
    config.hidden_dim = 256;
    config.max_seq_len = 2048;
    config.use_metal_gpu = true;

    AntigravityEngineContext* ctx = AntigravityEngineCreate(&config);
    assert(ctx != nullptr);
    std::vector<std::string> candidate_paths;
    if (const char* env_dir = std::getenv("ANTIGRAVITY_MODEL_DIR")) {
        if (env_dir[0] != '\0') {
            candidate_paths.push_back(std::string(env_dir) + "/tinyllama/model.safetensors");
            candidate_paths.push_back(std::string(env_dir) + "/model.safetensors");
        }
    }
    candidate_paths.push_back("../models/tinyllama/model.safetensors");
    candidate_paths.push_back("models/tinyllama/model.safetensors");

    int load_res = -1;
    for (const std::string& path_str : candidate_paths) {
        const char* path = path_str.c_str();
        std::cout << "Attempting to load weights from: " << path << "...\n";
        load_res = AntigravityEngineLoadModel(ctx, path);
        if (load_res == 0) break;
    }
    if (load_res != 0) {
        std::cerr << "FATAL ERROR: Failed to load safetensors model file!\n";
        return 1;
    }
    std::cout << "✅ Model weights loaded successfully into Metal VRAM.\n";

    uint64_t mem = AntigravityEngineGetAllocatedMemoryBytes(ctx);
    std::cout << "  • Zero-Copy Shared VRAM Allocated: " << mem / 1024 << " KB\n";

    std::vector<float> weights(config.hidden_dim * config.vocab_size, 0.01f);
    std::vector<int32_t> out_tokens(config.n_channels * 50, 0);
    AntigravityRolloutResult result;

    int ret = AntigravityEngineGenerateRollouts(ctx, weights.data(), 50, 0.7f, out_tokens.data(), &result);
    assert(ret == 0);

    std::cout << "✅ AntigravityEngineGenerateRollouts executed:\n";
    std::cout << "  • Total Tokens Generated: " << result.total_tokens_generated << "\n";
    std::cout << "  • Execution Wall Time:    " << result.execution_wall_time_ms << " ms\n";
    std::cout << "  • Rollout Throughput:    " << result.throughput_tokens_per_sec << " tok/s\n";

    std::vector<float> scores(config.n_channels, 0.0f);
    int best_channel = AntigravityEngineVerifyCandidates(ctx, out_tokens.data(), 50, scores.data());
    std::cout << "✅ AntigravityEngineVerifyCandidates executed:\n";
    std::cout << "  • Best Candidate Channel:  " << best_channel << "\n";
    std::cout << "  • Best Candidate Score:    " << scores[best_channel] << "\n";

    std::cout << "\nTesting Native C++ MCTS Tree Search with Process Reward Branch Pruning...\n";
    std::vector<int32_t> prompt = {1, 15043, 29892, 1125}; // "Prove that"
    AntigravityMCTSConfig mcts_cfg = {16, 3, 4, 0.8f, 0.9f};
    AntigravityMCTSResult mcts_res;
    std::vector<int32_t> mcts_out(128, 0);

    int mcts_ret = AntigravityEngineNativeMCTSGenerate(ctx, prompt.data(), (int32_t)prompt.size(), &mcts_cfg, mcts_out.data(), &mcts_res);
    assert(mcts_ret == 0);
    std::cout << "✅ AntigravityEngineNativeMCTSGenerate executed:\n";
    std::cout << "  • Winning Tokens Generated: " << mcts_res.total_tokens_generated << "\n";
    std::cout << "  • Evaluated Tokens Across Branches: " << mcts_res.total_tokens_evaluated << "\n";
    std::cout << "  • Chunks Expanded:          " << mcts_res.chunks_expanded << "\n";
    std::cout << "  • Best Tree Score:          " << mcts_res.best_score << "\n";
    std::cout << "  • Total Search Time:        " << mcts_res.execution_wall_time_ms << " ms\n";

    AntigravityEngineSanitizeBuffers(ctx);
    std::cout << "✅ AntigravityEngineSanitizeBuffers executed successfully.\n";

    AntigravityEngineDestroy(ctx);
    std::cout << "=========================================================\n";
    std::cout << "✅ All C++ API verification checks PASSED cleanly!\n";
    std::cout << "=========================================================\n";

    return 0;
}
