#include "gguf_reader.h"
#include <iostream>
#include <stdexcept>

GGUFReader::GGUFReader(const std::string& path) {
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open GGUF file: " + path);
    }
}

const std::map<std::string, std::string>& GGUFReader::getStringMetadata() const {
    return string_metadata_;
}

const std::map<std::string, uint32_t>& GGUFReader::getUint32Metadata() const {
    return uint32_metadata_;
}

const std::map<std::string, float>& GGUFReader::getFloatMetadata() const {
    return float_metadata_;
}

std::string GGUFReader::readString() {
    uint64_t len;
    file_.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (file_.eof()) throw std::runtime_error("Unexpected EOF while reading string length");
    std::string str(len, '\0');
    file_.read(&str[0], len);
    if (file_.eof()) throw std::runtime_error("Unexpected EOF while reading string content");
    return str;
}

void GGUFReader::skipValue(uint32_t type) {
    switch (type) {
        case 0: file_.seekg(1, std::ios::cur); break; // UINT8
        case 1: file_.seekg(1, std::ios::cur); break; // INT8
        case 2: file_.seekg(2, std::ios::cur); break; // UINT16
        case 3: file_.seekg(2, std::ios::cur); break; // INT16
        case 4: file_.seekg(4, std::ios::cur); break; // UINT32
        case 5: file_.seekg(4, std::ios::cur); break; // INT32
        case 6: file_.seekg(4, std::ios::cur); break; // FLOAT32
        case 7: file_.seekg(1, std::ios::cur); break; // BOOL
        case 8: readString(); break;                  // STRING
        case 9: skipArray(); break;                   // ARRAY
        case 10: file_.seekg(8, std::ios::cur); break; // UINT64
        case 11: file_.seekg(8, std::ios::cur); break; // INT64
        case 12: file_.seekg(8, std::ios::cur); break; // FLOAT64
        default: throw std::runtime_error("Unknown GGUF metadata value type: " + std::to_string(type));
    }
}

void GGUFReader::skipArray() {
    uint32_t type;
    uint64_t count;
    file_.read(reinterpret_cast<char*>(&type), sizeof(type));
    file_.read(reinterpret_cast<char*>(&count), sizeof(count));
    for (uint64_t i = 0; i < count; ++i) {
        skipValue(type);
    }
}

GGUFModelConfig GGUFReader::parse() {
    uint32_t magic;
    file_.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x46554747) { // "GGUF" in little-endian
        throw std::runtime_error("Invalid GGUF magic");
    }

    uint32_t version;
    file_.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 2 && version != 3) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version));
    }

    uint64_t tensor_count;
    uint64_t kv_count;
    file_.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    file_.read(reinterpret_cast<char*>(&kv_count), sizeof(kv_count));
    
    GGUFModelConfig config;
    config.tensor_count = tensor_count;

    for (uint64_t i = 0; i < kv_count; ++i) {
        if (file_.eof()) throw std::runtime_error("Unexpected EOF while reading KV pairs");
        std::string key = readString();
        
        uint32_t type;
        file_.read(reinterpret_cast<char*>(&type), sizeof(type));
        
        if (type == 8) { // STRING
            std::string val = readString();
            string_metadata_[key] = val;
            if (key == "general.architecture") {
                config.architecture = val;
            }
        } else if (type == 4) { // UINT32
            uint32_t val;
            file_.read(reinterpret_cast<char*>(&val), sizeof(val));
            uint32_metadata_[key] = val;
        } else if (type == 6) { // FLOAT32
            float val;
            file_.read(reinterpret_cast<char*>(&val), sizeof(val));
            float_metadata_[key] = val;
        } else if (type == 9 && key == "tokenizer.ggml.tokens") { // ARRAY
            uint32_t arr_type;
            uint64_t arr_count;
            file_.read(reinterpret_cast<char*>(&arr_type), sizeof(arr_type));
            file_.read(reinterpret_cast<char*>(&arr_count), sizeof(arr_count));
            config.vocab_size = arr_count;
            for (uint64_t j = 0; j < arr_count; ++j) {
                skipValue(arr_type);
            }
        } else {
            skipValue(type);
        }
    }
    
    std::string arch = config.architecture;
    if (!arch.empty()) {
        auto get_u32 = [&](const std::string& k, uint32_t& v) {
            if (uint32_metadata_.count(k)) v = uint32_metadata_[k];
        };
        auto get_f32 = [&](const std::string& k, float& v) {
            if (float_metadata_.count(k)) v = float_metadata_[k];
        };
        
        get_u32(arch + ".block_count", config.block_count);
        get_u32(arch + ".embedding_length", config.embedding_length);
        get_u32(arch + ".attention.head_count", config.head_count);
        get_u32(arch + ".attention.head_count_kv", config.head_count_kv);
        get_u32(arch + ".feed_forward_length", config.feed_forward_length);
        get_u32(arch + ".context_length", config.context_length);
        
        get_f32(arch + ".rope.freq_base", config.rope_freq_base);
        get_f32(arch + ".attention.layer_norm_rms_epsilon", config.norm_eps);
    }
    
    return config;
}
