/*
 * Project Antigravity — C++ API Integration Test Client
 */

#include "antigravity_c_api.h"
#include <iostream>
#include <vector>
#include <cassert>

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
    std::cout << "✅ AntigravityEngine created successfully.\n";

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

    AntigravityEngineSanitizeBuffers(ctx);
    std::cout << "✅ AntigravityEngineSanitizeBuffers executed successfully.\n";

    AntigravityEngineDestroy(ctx);
    std::cout << "=========================================================\n";
    std::cout << "✅ All C++ API verification checks PASSED cleanly!\n";
    std::cout << "=========================================================\n";

    return 0;
}
