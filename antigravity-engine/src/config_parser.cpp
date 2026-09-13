#include "transformer_engine.h"
#include "gguf_reader.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

void TransformerConfig::print() const {
    std::cout << "--- Model Architecture ---\n";
    std::cout << "hidden_dim: " << hidden_dim << "\n";
    std::cout << "intermediate_dim: " << intermediate_dim << "\n";
    std::cout << "n_layers: " << n_layers << "\n";
    std::cout << "n_heads: " << n_heads << "\n";
    std::cout << "n_kv_heads: " << n_kv_heads << "\n";
    std::cout << "head_dim: " << head_dim << "\n";
    std::cout << "vocab_size: " << vocab_size << "\n";
    std::cout << "max_seq_len: " << max_seq_len << "\n";
    std::cout << "norm_eps: " << norm_eps << "\n";
    std::cout << "rope_theta: " << rope_theta << "\n";
    std::cout << "n_channels: " << n_channels << "\n";
    std::cout << "q_len_max: " << q_len_max << "\n";
    std::cout << "--------------------------\n";
}

TransformerConfig TransformerConfig::fromGGUF(const std::string& gguf_path) {
    GGUFReader reader(gguf_path);
    GGUFModelConfig parsed = reader.parse();
    
    TransformerConfig cfg;
    if (parsed.block_count > 0) cfg.n_layers = parsed.block_count;
    if (parsed.embedding_length > 0) cfg.hidden_dim = parsed.embedding_length;
    if (parsed.head_count > 0) cfg.n_heads = parsed.head_count;
    if (parsed.head_count_kv > 0) cfg.n_kv_heads = parsed.head_count_kv;
    if (parsed.feed_forward_length > 0) cfg.intermediate_dim = parsed.feed_forward_length;
    if (parsed.vocab_size > 0) cfg.vocab_size = parsed.vocab_size;
    if (parsed.context_length > 0) cfg.max_seq_len = parsed.context_length;
    if (parsed.rope_freq_base > 0.0f) cfg.rope_theta = parsed.rope_freq_base;
    if (parsed.norm_eps > 0.0f) cfg.norm_eps = parsed.norm_eps;
    if (parsed.head_dim() > 0) cfg.head_dim = parsed.head_dim();
    
    return cfg;
}

static std::vector<int> parseShape(const std::string& json, size_t offset) {
    std::vector<int> shape;
    size_t start = json.find('[', offset);
    if (start == std::string::npos) return shape;
    size_t end = json.find(']', start);
    if (end == std::string::npos) return shape;
    
    std::string arr = json.substr(start + 1, end - start - 1);
    size_t pos = 0;
    while (pos < arr.size()) {
        while (pos < arr.size() && (std::isspace(arr[pos]) || arr[pos] == ',')) pos++;
        if (pos >= arr.size()) break;
        size_t next = pos;
        while (next < arr.size() && std::isdigit(arr[next])) next++;
        if (next > pos) {
            shape.push_back(std::stoi(arr.substr(pos, next - pos)));
        }
        pos = next;
    }
    return shape;
}

TransformerConfig TransformerConfig::fromSafetensors(const std::string& safetensors_path) {
    std::ifstream file(safetensors_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open safetensors file: " + safetensors_path);
    }
    
    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    
    if (header_len > 100 * 1024 * 1024 || file.eof()) {
        throw std::runtime_error("Invalid safetensors header length");
    }
    
    std::string json(header_len, '\0');
    file.read(&json[0], header_len);
    
    TransformerConfig cfg;
    
    // Count layers
    int max_layer = -1;
    size_t pos = 0;
    while ((pos = json.find("\"model.layers.", pos)) != std::string::npos) {
        pos += 14;
        size_t dot = json.find('.', pos);
        if (dot != std::string::npos) {
            std::string num = json.substr(pos, dot - pos);
            try {
                int l = std::stoi(num);
                if (l > max_layer) max_layer = l;
            } catch (...) {}
        }
    }
    if (max_layer >= 0) cfg.n_layers = max_layer + 1;
    
    // Hidden dim & vocab size
    pos = json.find("\"model.embed_tokens.weight\"");
    if (pos != std::string::npos) {
        pos = json.find("\"shape\"", pos);
        if (pos != std::string::npos) {
            auto shape = parseShape(json, pos);
            if (shape.size() >= 2) {
                cfg.vocab_size = shape[0];
                cfg.hidden_dim = shape[1];
            }
        }
    }
    
    // n_heads
    pos = json.find("\"model.layers.0.self_attn.q_proj.weight\"");
    if (pos != std::string::npos) {
        pos = json.find("\"shape\"", pos);
        if (pos != std::string::npos) {
            auto shape = parseShape(json, pos);
            if (shape.size() >= 1 && cfg.head_dim > 0) {
                cfg.n_heads = shape[0] / cfg.head_dim;
            }
        }
    }
    
    // n_kv_heads
    pos = json.find("\"model.layers.0.self_attn.k_proj.weight\"");
    if (pos != std::string::npos) {
        pos = json.find("\"shape\"", pos);
        if (pos != std::string::npos) {
            auto shape = parseShape(json, pos);
            if (shape.size() >= 1 && cfg.head_dim > 0) {
                cfg.n_kv_heads = shape[0] / cfg.head_dim;
            }
        }
    }
    
    // intermediate_dim
    pos = json.find("\"model.layers.0.mlp.gate_proj.weight\"");
    if (pos != std::string::npos) {
        pos = json.find("\"shape\"", pos);
        if (pos != std::string::npos) {
            auto shape = parseShape(json, pos);
            if (shape.size() >= 1) {
                cfg.intermediate_dim = shape[0];
            }
        }
    }
    
    return cfg;
}
