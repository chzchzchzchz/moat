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
    f << "  \"speculative_decoding_worst_case\": {\n";
    f << "    \"tokens_per_sec\": " << spec_k4_tok_sec << "\n";
    f << "  },\n";
    f << "  \"mcts_parallel_expansion\": {\n";
    f << "    \"evaluations_per_sec\": " << mcts_eval_sec << "\n";
    f << "  }\n";
    f << "}\n";
    f.close();
}

int main() {
    TransformerConfig config;
    config.n_layers = 22; // TinyLlama/Qwen scale
    config.n_channels = 1; 
    
    MetalTransformerEngine target_engine(config);
    MetalTransformerEngine draft_engine(config);
    
    std::vector<int32_t> prompt = {1, 15, 23, 44, 12, 9, 3, 4, 1, 15, 23, 44, 12, 9, 3, 4}; // 16 tokens
    int32_t max_new_tokens = 32;
    
    // 1. Standard (Batch=1)
    target_engine.allocateDummyWeights();
    draft_engine.allocateDummyWeights();
    
    GenerationResult res_std = target_engine.generate(prompt.data(), prompt.size(), max_new_tokens, 0.0f, 1.0f);
    double std_tok_sec = (res_std.total_tokens / (res_std.total_ms / 1000.0));
    
    // 2. Speculative Decoding (Worst case with dummy random weights)
    GenerationResult res_spec = target_engine.generateSpeculative(&draft_engine, prompt.data(), prompt.size(), max_new_tokens, 4, 0.0f, 1.0f);
    double spec_k4_tok_sec = (res_spec.total_tokens / (res_spec.total_ms / 1000.0));
    
    // 3. MCTS (Batch=8)
    config.n_channels = 8;
    MetalTransformerEngine mcts_engine(config);
    mcts_engine.allocateDummyWeights();
    MCTSConfig mcts_cfg;
    mcts_cfg.chunk_tokens = 16;
    mcts_cfg.num_chunks = 2;
    mcts_cfg.branches_per_chunk = 8;
    MCTSResult res_mcts = mcts_engine.generateMCTS(prompt.data(), prompt.size(), mcts_cfg);
    double mcts_eval_sec = (res_mcts.total_tokens_evaluated / (res_mcts.total_ms / 1000.0));
    
    write_json("benchmark_metrics.json", std_tok_sec, spec_k4_tok_sec, mcts_eval_sec);
    
    std::cout << "Standard: " << std_tok_sec << " tok/s\n";
    std::cout << "Speculative (Worst Case): " << spec_k4_tok_sec << " tok/s\n";
    std::cout << "MCTS Evaluated: " << mcts_eval_sec << " tok/s\n";
    
    return 0;
}
