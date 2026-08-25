#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "src/transformer_engine.h"

void write_json(const std::string& path, 
                double std_tok_sec,
                double spec_k4_tok_sec,
                double mcts_eval_sec) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"hardware\": \"Apple M-Series Unified Memory\",\n";
    f << "  \"standard_autoregressive\": {\n";
    f << "    \"tokens_per_sec\": " << std_tok_sec << "\n";
    f << "  },\n";
    f << "  \"speculative_decoding_real\": {\n";
    f << "    \"tokens_per_sec\": " << spec_k4_tok_sec << "\n";
    f << "  },\n";
    f << "  \"mcts_parallel_expansion\": {\n";
    f << "    \"evaluations_per_sec\": " << mcts_eval_sec << "\n";
    f << "  }\n";
    f << "}\n";
    f.close();
}

int main() {
    std::cout << "🚀 Starting REAL Physical Edge Reasoning Benchmark Suite...\n";
    
    // TinyLlama-1.1B Architecture
    TransformerConfig config;
    config.n_layers = 22; 
    config.n_channels = 1; 
    
    std::cout << "Loading 2.2GB Safetensors into Metal Unified Memory...\n";
    MetalTransformerEngine target_engine(config);
    bool loaded = target_engine.loadWeights("../models/qwen3.5/model.safetensors");
    if (!loaded) {
        std::cerr << "❌ Failed to load target model weights!\n";
        return 1;
    }
    
    MetalTransformerEngine draft_engine(config);
    loaded = draft_engine.loadWeights("../models/qwen3.5/model.safetensors");
    if (!loaded) {
        std::cerr << "❌ Failed to load draft model weights!\n";
        return 1;
    }
    
    std::vector<int32_t> prompt = {1, 15, 23, 44, 12, 9, 3, 4, 1, 15, 23, 44, 12, 9, 3, 4}; // 16 tokens
    int32_t max_new_tokens = 32;
    
    std::cout << "\n[1/3] Running Standard Autoregressive Benchmark (Physical I/O bound)...\n";
    GenerationResult res_std = target_engine.generate(prompt.data(), prompt.size(), max_new_tokens, 0.0f, 1.0f);
    double std_tok_sec = (res_std.total_tokens / (res_std.total_ms / 1000.0));
    std::cout << "      -> TTFT: " << res_std.ttft_ms << " ms, TPOT: " << res_std.tpot_ms << " ms, Throughput: " << std_tok_sec << " tok/s\n";
    
    std::cout << "\n[2/3] Running Speculative Decoding Benchmark (Real Weights)...\n";
    GenerationResult res_spec = target_engine.generateSpeculative(&draft_engine, prompt.data(), prompt.size(), max_new_tokens, 4, 0.0f, 1.0f);
    double spec_k4_tok_sec = (res_spec.total_tokens / (res_spec.total_ms / 1000.0));
    std::cout << "      -> TTFT: " << res_spec.ttft_ms << " ms, TPOT: " << res_spec.tpot_ms << " ms, Throughput: " << spec_k4_tok_sec << " tok/s\n";
    
    std::cout << "\n[3/3] Running MCTS Benchmark (Batch=8, Bandwidth Saturation Test)...\n";
    config.n_channels = 8;
    MetalTransformerEngine mcts_engine(config);
    mcts_engine.loadWeights("../models/qwen3.5/model.safetensors");
    MCTSConfig mcts_cfg;
    mcts_cfg.chunk_tokens = 16;
    mcts_cfg.num_chunks = 2;
    mcts_cfg.branches_per_chunk = 8;
    MCTSResult res_mcts = mcts_engine.generateMCTS(prompt.data(), prompt.size(), mcts_cfg);
    double mcts_eval_sec = (res_mcts.total_tokens_evaluated / (res_mcts.total_ms / 1000.0));
    std::cout << "      -> MCTS Latency: " << res_mcts.total_ms << " ms, Tokens Evaluated: " << res_mcts.total_tokens_evaluated 
              << ", Throughput: " << mcts_eval_sec << " tok/s\n";
    
    write_json("benchmark_real_weights.json", std_tok_sec, spec_k4_tok_sec, mcts_eval_sec);
    
    std::cout << "\n✅ Physical Benchmarks complete. Dumping results to benchmark_real_weights.json...\n";
    
    return 0;
}
