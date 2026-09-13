#pragma once
#include <string>
#include <cstdint>
#include <map>
#include <vector>
#include <fstream>

struct GGUFModelConfig {
    std::string architecture;   // e.g. "llama", "qwen2", "mistral", "phi"
    uint32_t block_count = 0;         // n_layers
    uint32_t embedding_length = 0;    // hidden_dim
    uint32_t head_count = 0;          // n_heads (query)
    uint32_t head_count_kv = 0;       // n_kv_heads (for GQA)
    uint32_t feed_forward_length = 0; // intermediate_dim
    uint32_t context_length = 0;      // max_seq_len
    uint32_t vocab_size = 0;
    float rope_freq_base = 10000.0f;  // rope_theta
    float norm_eps = 1e-5f;
    uint64_t tensor_count = 0;
    
    // Derived
    uint32_t head_dim() const { return (head_count > 0) ? embedding_length / head_count : 0; }
    bool isValid() const { return block_count > 0 && embedding_length > 0 && head_count > 0; }
};

class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);
    
    /// Parse GGUF header and extract model architecture configuration
    GGUFModelConfig parse();
    
    /// Get raw metadata key-value pairs
    const std::map<std::string, std::string>& getStringMetadata() const;
    const std::map<std::string, uint32_t>& getUint32Metadata() const;
    const std::map<std::string, float>& getFloatMetadata() const;
    
private:
    std::ifstream file_;
    std::map<std::string, std::string> string_metadata_;
    std::map<std::string, uint32_t> uint32_metadata_;
    std::map<std::string, float> float_metadata_;
    
    std::string readString();
    void skipValue(uint32_t type);
    void skipArray();
};
