#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include "antigravity_c_api.h"
#include "gguf_reader.h"
#include "transformer_engine.h"

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   Antigravity Engine Pure C/C++ SDK Verification Suite   \n";
    std::cout << "=========================================================\n";

    // 1. Test Config creation
    AntigravityConfig config;
    config.n_channels = 4;
    config.vocab_size = 32000;
    config.hidden_dim = 2048;
    config.max_seq_len = 2048;
    config.use_metal_gpu = true;

    AntigravityEngineContext* ctx = AntigravityEngineCreate(&config);
    assert(ctx != nullptr && "Context creation failed");
    std::cout << "[PASS] 1. AntigravityEngineCreate successful\n";

    // 2. Test Initial Memory Allocation
    uint64_t initial_bytes = AntigravityEngineGetAllocatedMemoryBytes(ctx);
    assert(initial_bytes > 0 && "Compute scratchpad must allocate VRAM");
    std::cout << "[PASS] 2. Initial VRAM scratchpad: " << (initial_bytes / 1024 / 1024) << " MB\n";

    // 3. Test Invalid License Rejection
    int32_t invalid_ret = AntigravityEngineSetLicenseKey(ctx, "invalid.license.token");
    assert(invalid_ret != 0 && "Invalid license must be rejected");
    assert(!AntigravityEngineIsLicensed(ctx) && "Engine must report unlicensed");
    std::cout << "[PASS] 3. Invalid license token correctly rejected\n";

    // Path resolution helper
    auto resolve_path = [](const std::string& rel) -> std::string {
        if (FILE* f = fopen(rel.c_str(), "r")) { fclose(f); return rel; }
        std::string up = "../" + rel;
        if (FILE* f = fopen(up.c_str(), "r")) { fclose(f); return up; }
        return rel;
    };

    // 4. Test Safetensors Dynamic Config Inference
    std::string safetensors_path = resolve_path("models/tinyllama/model.safetensors");
    TransformerConfig t_cfg = TransformerConfig::fromSafetensors(safetensors_path);
    assert(t_cfg.n_layers == 22 && "TinyLlama must have 22 layers");
    assert(t_cfg.hidden_dim == 2048 && "TinyLlama hidden_dim must be 2048");
    assert(t_cfg.vocab_size == 32000 && "TinyLlama vocab_size must be 32000");
    std::cout << "[PASS] 4. Safetensors dynamic header parsing: 22 layers, 2048 dim, 32000 vocab\n";

    // 5. Test GGUF Header Parsing
    std::string gguf_path = resolve_path("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");
    GGUFReader reader(gguf_path);
    GGUFModelConfig gguf_cfg = reader.parse();
    assert(gguf_cfg.block_count == 22 && "GGUF block count must be 22");
    assert(gguf_cfg.embedding_length == 2048 && "GGUF embedding length must be 2048");
    std::cout << "[PASS] 5. GGUF dynamic metadata parsing: " << gguf_cfg.architecture 
              << " (" << gguf_cfg.block_count << " blocks, " 
              << gguf_cfg.embedding_length << " dim)\n";

    // 6. Test Model Weight Loading into Metal
    int32_t load_ret = AntigravityEngineLoadModel(ctx, safetensors_path.c_str());
    assert(load_ret == 0 && "Model loading must succeed");
    assert(AntigravityEngineHasWeights(ctx) && "HasWeights must be true");
    uint64_t loaded_bytes = AntigravityEngineGetAllocatedMemoryBytes(ctx);
    assert(loaded_bytes > initial_bytes + 500 * 1024 * 1024 && "Loaded weights must allocate VRAM");
    std::cout << "[PASS] 6. Model loaded into Metal VRAM: " << (loaded_bytes / 1024 / 1024) << " MB total\n";

    // 7. Test Native Parallel Generation across 4 Channels with Linguistic Meaning Check
    // Prompt: "The symptoms of clinical depression include"
    std::vector<int32_t> prompt = {1, 450, 25828, 4835, 310, 24899, 936, 316, 2590, 3160};
    int32_t max_new = 15;
    std::vector<int32_t> out_tokens(config.n_channels * max_new, 0);
    std::vector<float> out_logprobs(config.n_channels, 0.0f);
    std::vector<int32_t> out_counts(config.n_channels, 0);
    double ttft_ms = 0.0, total_ms = 0.0;

    int32_t gen_ret = AntigravityEngineNativeGenerate(
        ctx,
        prompt.data(),
        (int32_t)prompt.size(),
        max_new,
        0.1f, // Low temperature for deterministic evaluation
        0.9f,
        out_tokens.data(),
        out_logprobs.data(),
        out_counts.data(),
        &ttft_ms,
        &total_ms
    );
    assert(gen_ret == 0 && "Native generation must succeed");
    assert(ttft_ms > 0.0 && "TTFT must be non-zero");

    // Verify token validity and linguistic meaning across channels
    for (int c = 0; c < config.n_channels; c++) {
        assert(out_counts[c] > 0 && "Channel must generate tokens");
        int offset = c * max_new;
        for (int i = 0; i < out_counts[c]; i++) {
            int32_t tok = out_tokens[offset + i];
            assert(tok >= 0 && tok < config.vocab_size && "Token ID must be within vocabulary range");
        }
    }

    // Verify exact linguistic prediction from real weights:
    // "The symptoms of clinical depression include" -> " feelings" (21737), " of" (310), " sadness" (14610)
    assert(out_tokens[0] == 21737 && "First generated token must be 'feelings' (id 21737)");
    assert(out_tokens[1] == 310 && "Second generated token must be 'of' (id 310)");
    assert(out_tokens[2] == 14610 && "Third generated token must be 'sadness' (id 14610)");

    std::cout << "[PASS] 7. 4-Channel Parallel Metal Rollout verified with real linguistic tokens:\n";
    std::cout << "         Tokens: [" << out_tokens[0] << " (feelings), " 
              << out_tokens[1] << " (of), " << out_tokens[2] << " (sadness)...] "
              << "TTFT=" << ttft_ms << "ms, Total=" << total_ms << "ms\n";

    // 8. Test Verifier Scoring
    std::vector<float> scores(config.n_channels, 0.0f);
    int32_t best_idx = AntigravityEngineVerifyCandidates(
        ctx,
        out_tokens.data(),
        max_new,
        scores.data()
    );
    assert(best_idx >= 0 && best_idx < config.n_channels && "Best index must be within channels");
    assert(scores[best_idx] > 0.0f && scores[best_idx] <= 1.0f && "Winning score must be in (0, 1]");
    for (int c = 0; c < config.n_channels; c++) {
        assert(scores[c] >= 0.0f && scores[c] <= 1.0f && "Candidate score must be in [0, 1]");
    }

    std::cout << "[PASS] 8. Candidate Verification: winning channel=" << best_idx 
              << " score=" << scores[best_idx] << "\n";

    // 9. Test Unload & Resource Deallocation
    AntigravityEngineUnloadWeights(ctx);
    assert(!AntigravityEngineHasWeights(ctx) && "Weights must be unloaded");
    assert(AntigravityEngineGetAllocatedMemoryBytes(ctx) == 0 && "Memory must be 0 after unload");
    std::cout << "[PASS] 9. AntigravityEngineUnloadWeights freed all VRAM (0 bytes)\n";

    // 10. Destroy Context
    AntigravityEngineDestroy(ctx);
    std::cout << "[PASS] 10. AntigravityEngineDestroy cleanly released context\n";

    std::cout << "=========================================================\n";
    std::cout << "   ALL 10 C/C++ SDK VERIFICATION TESTS PASSED (100%)    \n";
    std::cout << "=========================================================\n";
    return 0;
}
