/* Project Antigravity — Unified C API Engine Bridge */
#include "antigravity_engine_c.h"
#include "antigravity_c_api.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <algorithm>

namespace {

struct BPETokenizer {
    std::unordered_map<std::string, int32_t> vocab_map;
    std::vector<std::string> vocab_table;
    std::unordered_map<std::string, int> merge_ranks;
    int32_t bos_id = 1;
    int32_t eos_id = 2;
    int32_t unk_id = 0;
    bool is_loaded = false;

    static std::string unescape_json(const std::string& s) {
        std::string res;
        res.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[++i];
                if (c == '"') res.push_back('"');
                else if (c == '\\') res.push_back('\\');
                else if (c == '/') res.push_back('/');
                else if (c == 'b') res.push_back('\b');
                else if (c == 'f') res.push_back('\f');
                else if (c == 'n') res.push_back('\n');
                else if (c == 'r') res.push_back('\r');
                else if (c == 't') res.push_back('\t');
                else if (c == 'u' && i + 4 < s.size()) {
                    std::string hex_str = s.substr(i + 1, 4);
                    i += 4;
                    uint32_t cp = (uint32_t)std::stoul(hex_str, nullptr, 16);
                    if (cp < 0x80) {
                        res.push_back((char)cp);
                    } else if (cp < 0x800) {
                        res.push_back((char)(0xC0 | (cp >> 6)));
                        res.push_back((char)(0x80 | (cp & 0x3F)));
                    } else {
                        res.push_back((char)(0xE0 | (cp >> 12)));
                        res.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                        res.push_back((char)(0x80 | (cp & 0x3F)));
                    }
                } else {
                    res.push_back(c);
                }
            } else {
                res.push_back(s[i]);
            }
        }
        return res;
    }

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        // Parse vocab section
        size_t vocab_pos = content.find("\"vocab\"");
        if (vocab_pos == std::string::npos) return false;
        size_t open_brace = content.find('{', vocab_pos);
        if (open_brace == std::string::npos) return false;

        size_t pos = open_brace + 1;
        int brace_depth = 1;
        vocab_table.resize(32000, "");

        while (pos < content.size() && brace_depth > 0) {
            char ch = content[pos];
            if (ch == '}') {
                brace_depth--;
                pos++;
                continue;
            }
            if (ch == '{') {
                brace_depth++;
                pos++;
                continue;
            }
            if (ch == '"') {
                size_t key_start = pos + 1;
                size_t key_end = key_start;
                while (key_end < content.size()) {
                    if (content[key_end] == '"') {
                        size_t bs = 0;
                        size_t k = key_end;
                        while (k > key_start && content[k - 1] == '\\') {
                            bs++;
                            k--;
                        }
                        if (bs % 2 == 0) break;
                    }
                    key_end++;
                }
                std::string raw_key = content.substr(key_start, key_end - key_start);
                std::string key = unescape_json(raw_key);

                pos = content.find(':', key_end);
                if (pos == std::string::npos) break;
                pos++;
                while (pos < content.size() && isspace((unsigned char)content[pos])) pos++;
                size_t num_end = pos;
                while (num_end < content.size() && (isdigit((unsigned char)content[num_end]) || content[num_end] == '-')) num_end++;
                if (pos < num_end) {
                    int32_t val = (int32_t)std::stoi(content.substr(pos, num_end - pos));
                    vocab_map[key] = val;
                    if ((size_t)val >= vocab_table.size()) {
                        vocab_table.resize(val + 1, "");
                    }
                    vocab_table[val] = key;
                }
                pos = num_end;
                while (pos < content.size() && (content[pos] == ',' || isspace((unsigned char)content[pos]))) pos++;
                continue;
            }
            pos++;
        }

        // Parse merges section if present
        size_t merges_pos = content.find("\"merges\"");
        if (merges_pos != std::string::npos) {
            size_t open_bracket = content.find('[', merges_pos);
            if (open_bracket != std::string::npos) {
                size_t m_pos = open_bracket + 1;
                int rank = 0;
                while (m_pos < content.size()) {
                    if (content[m_pos] == ']') break;
                    if (content[m_pos] == '"') {
                        size_t str_start = m_pos + 1;
                        size_t str_end = str_start;
                        while (str_end < content.size()) {
                            if (content[str_end] == '"') {
                                size_t bs = 0;
                                size_t k = str_end;
                                while (k > str_start && content[k - 1] == '\\') {
                                    bs++;
                                    k--;
                                }
                                if (bs % 2 == 0) break;
                            }
                            str_end++;
                        }
                        std::string raw_merge = content.substr(str_start, str_end - str_start);
                        std::string merge_pair = unescape_json(raw_merge);
                        merge_ranks[merge_pair] = rank++;
                        m_pos = str_end + 1;
                        continue;
                    }
                    m_pos++;
                }
            }
        }

        if (vocab_map.find("<s>") != vocab_map.end()) bos_id = vocab_map["<s>"];
        if (vocab_map.find("</s>") != vocab_map.end()) eos_id = vocab_map["</s>"];
        if (vocab_map.find("<unk>") != vocab_map.end()) unk_id = vocab_map["<unk>"];

        is_loaded = true;
        return true;
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> tokens;
        if (text.empty()) {
            tokens.push_back(bos_id);
            return tokens;
        }

        tokens.push_back(bos_id);

        if (!is_loaded) {
            // Fallback word/character token IDs
            for (char c : text) {
                tokens.push_back((int32_t)(uint8_t)c);
            }
            return tokens;
        }

        std::string processed;
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] == ' ') {
                processed += "\xe2\x96\x81";
            } else {
                if (i == 0) processed += "\xe2\x96\x81";
                processed.push_back(text[i]);
            }
        }

        std::vector<std::string> pieces;
        size_t idx = 0;
        while (idx < processed.size()) {
            unsigned char c = (unsigned char)processed[idx];
            size_t len = 1;
            if ((c & 0x80) == 0) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;

            if (idx + len <= processed.size()) {
                std::string piece = processed.substr(idx, len);
                if (vocab_map.find(piece) != vocab_map.end()) {
                    pieces.push_back(piece);
                } else {
                    for (size_t b = 0; b < len; ++b) {
                        char hex_buf[16];
                        snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", (unsigned char)processed[idx + b]);
                        pieces.push_back(std::string(hex_buf));
                    }
                }
                idx += len;
            } else {
                pieces.push_back(std::string(1, processed[idx++]));
            }
        }

        if (!merge_ranks.empty()) {
            while (pieces.size() >= 2) {
                int best_rank = 1000000000;
                int best_idx = -1;
                for (size_t i = 0; i < pieces.size() - 1; ++i) {
                    std::string pair_str = pieces[i] + " " + pieces[i + 1];
                    auto it = merge_ranks.find(pair_str);
                    if (it != merge_ranks.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = (int)i;
                    }
                }
                if (best_idx == -1) break;
                pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1];
                pieces.erase(pieces.begin() + best_idx + 1);
            }
        }

        for (const auto& p : pieces) {
            auto it = vocab_map.find(p);
            if (it != vocab_map.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(unk_id);
            }
        }

        return tokens;
    }

    std::string decode(const std::vector<int32_t>& tokens) const {
        if (!is_loaded) {
            std::string res;
            for (int32_t tok : tokens) {
                if (tok >= 32 && tok <= 126) res.push_back((char)tok);
                else res += " ";
            }
            return res;
        }

        std::string res;
        for (int32_t tok : tokens) {
            if (tok < 0 || (size_t)tok >= vocab_table.size()) continue;
            const std::string& piece = vocab_table[tok];
            if (piece == "<s>" || piece == "</s>" || piece == "<unk>") continue;

            if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
                uint8_t byte_val = (uint8_t)std::stoul(piece.substr(3, 2), nullptr, 16);
                res.push_back((char)byte_val);
                continue;
            }

            std::string text_piece = piece;
            size_t underline_pos = 0;
            while ((underline_pos = text_piece.find("\xe2\x96\x81", underline_pos)) != std::string::npos) {
                text_piece.replace(underline_pos, 3, " ");
                underline_pos += 1;
            }
            res += text_piece;
        }
        if (!res.empty() && res[0] == ' ') {
            res = res.substr(1);
        }
        return res;
    }
};

} // namespace

struct AntigravityEngineInternal {
    antigravity_config_t config;
    std::string model_path;
    AntigravityEngineContext* ctx = nullptr;
    BPETokenizer tokenizer;
    std::vector<int32_t> last_generated_tokens;
    uint32_t last_n_channels = 0;
    uint32_t last_seq_len = 0;
};

extern "C" {

antigravity_engine_t antigravity_engine_create(const antigravity_config_t* config, const char* model_path) {
    if (!config || !model_path) return NULL;

    AntigravityEngineInternal* engine = new AntigravityEngineInternal();
    engine->config = *config;
    engine->model_path = std::string(model_path);

    // Locate and load BPE tokenizer
    std::vector<std::string> tok_search_paths;
    std::string mp(model_path);
    size_t last_slash = mp.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        tok_search_paths.push_back(mp.substr(0, last_slash) + "/tokenizer.json");
    }
    tok_search_paths.push_back(mp + "/tokenizer.json");
    tok_search_paths.push_back("models/tinyllama/tokenizer.json");
    tok_search_paths.push_back("models/qwen/tokenizer.json");
    tok_search_paths.push_back("/Users/MohssineChazi2/moat/models/tinyllama/tokenizer.json");
    tok_search_paths.push_back("/Users/MohssineChazi2/moat/models/qwen/tokenizer.json");

    for (const auto& path : tok_search_paths) {
        if (engine->tokenizer.load(path)) {
            break;
        }
    }

    AntigravityConfig api_config;
    api_config.n_channels = config->parallel_channels > 0 ? (int32_t)config->parallel_channels : 8;
    api_config.vocab_size = engine->tokenizer.is_loaded ? (int32_t)engine->tokenizer.vocab_map.size() : 32000;
    api_config.hidden_dim = 2048;
    api_config.max_seq_len = 2048;
    api_config.use_metal_gpu = true;

    engine->ctx = AntigravityEngineCreate(&api_config);
    if (!engine->ctx) {
        delete engine;
        return NULL;
    }

    if (model_path && strlen(model_path) > 0) {
        AntigravityEngineLoadModel(engine->ctx, model_path);
    }
    return engine;
}

void antigravity_engine_destroy(antigravity_engine_t engine) {
    if (engine) {
        if (engine->ctx) {
            AntigravityEngineDestroy(engine->ctx);
            engine->ctx = NULL;
        }
        delete engine;
    }
}

antigravity_rollout_result_t* antigravity_generate_rollouts(
    antigravity_engine_t engine,
    const char* prompt,
    uint32_t max_tokens,
    float temperature
) {
    if (!engine || !engine->ctx || !prompt) return NULL;

    uint32_t n_channels = engine->config.parallel_channels > 0 ? engine->config.parallel_channels : 8;

    // Proper BPE Tokenization using loaded tokenizer vocabulary
    std::vector<int32_t> prompt_tokens = engine->tokenizer.encode(prompt);
    std::vector<int32_t> out_tokens(n_channels * max_tokens, 0);

    AntigravityMCTSConfig mcts_cfg = { (int32_t)n_channels, 3, 4, temperature, 0.9f };
    AntigravityMCTSResult api_result;
    memset(&api_result, 0, sizeof(api_result));

    int ret = AntigravityEngineNativeMCTSGenerate(
        engine->ctx,
        prompt_tokens.data(),
        (int32_t)prompt_tokens.size(),
        &mcts_cfg,
        out_tokens.data(),
        &api_result
    );

    if (ret != 0) return NULL;

    // Cache generated token buffers directly for downstream verification
    engine->last_generated_tokens = out_tokens;
    engine->last_n_channels = n_channels;
    engine->last_seq_len = max_tokens;

    antigravity_rollout_result_t* res = new antigravity_rollout_result_t();
    res->candidate_count = n_channels;
    res->candidates = new antigravity_candidate_t[n_channels];

    std::string prompt_str = std::string(prompt);
    uint32_t best_idx = 0;
    float max_logprob = -1e9f;

    for (uint32_t c = 0; c < n_channels; c++) {
        // Extract channel tokens and decode into human-readable text
        std::vector<int32_t> chan_toks;
        chan_toks.reserve(max_tokens);
        for (uint32_t s = 0; s < max_tokens; s++) {
            chan_toks.push_back(out_tokens[c * max_tokens + s]);
        }

        std::string decoded_text = engine->tokenizer.decode(chan_toks);
        std::string trace = prompt_str + "\n[Channel " + std::to_string(c + 1) + " rollout]:\n" + decoded_text;

        // Compute candidate log-probability from search confidence without synthetic channel penalty
        float chan_logprob = (api_result.best_score != 0.0f) ? std::log(std::max(0.01f, std::min(1.0f, std::abs(api_result.best_score)))) : -1.0f;

        res->candidates[c].trace_text = strdup(trace.c_str());
        res->candidates[c].logprob = chan_logprob;
        res->candidates[c].token_count = max_tokens;

        if (chan_logprob > max_logprob) {
            max_logprob = chan_logprob;
            best_idx = c;
        }
    }

    res->best_candidate_index = best_idx;
    res->total_latency_ms = api_result.execution_wall_time_ms;
    res->token_savings_pct = 0.0f;
    res->reflection_triggered = false;

    return res;
}

antigravity_verification_result_t* antigravity_verify_candidates(
    antigravity_engine_t engine,
    const antigravity_rollout_result_t* rollouts
) {
    if (!engine || !engine->ctx || !rollouts || rollouts->candidate_count == 0) return NULL;

    uint32_t n_channels = rollouts->candidate_count;
    uint32_t seq_len = rollouts->candidates[0].token_count;

    std::vector<int32_t> candidate_tokens(n_channels * seq_len, 0);

    // Verify candidate tokens directly from token buffers
    if (engine->last_generated_tokens.size() >= n_channels * seq_len &&
        engine->last_n_channels == n_channels && engine->last_seq_len == seq_len) {
        candidate_tokens = engine->last_generated_tokens;
    } else {
        // Fallback: tokenize trace texts with BPE tokenizer
        for (uint32_t c = 0; c < n_channels; c++) {
            const char* trace = rollouts->candidates[c].trace_text;
            if (trace) {
                std::vector<int32_t> encoded = engine->tokenizer.encode(trace);
                for (size_t s = 0; s < seq_len; s++) {
                    if (s < encoded.size()) {
                        candidate_tokens[c * seq_len + s] = encoded[s];
                    } else {
                        candidate_tokens[c * seq_len + s] = engine->tokenizer.eos_id;
                    }
                }
            }
        }
    }

    std::vector<float> scores(n_channels, 0.0f);
    int32_t best_channel = AntigravityEngineVerifyCandidates(
        engine->ctx,
        candidate_tokens.data(),
        (int32_t)seq_len,
        scores.data()
    );

    uint32_t selected_c = 0;
    float best_score = scores[0];
    for (uint32_t c = 1; c < n_channels; c++) {
        if (scores[c] > best_score) {
            best_score = scores[c];
            selected_c = c;
        }
    }

    antigravity_verification_result_t* vres = new antigravity_verification_result_t();
    vres->selected_index = (best_channel >= 0 && (uint32_t)best_channel < n_channels) ? (uint32_t)best_channel : selected_c;
    vres->confidence_score = scores[vres->selected_index];

    std::string reasoning = "Verified candidate channel " + std::to_string(vres->selected_index) +
                            " with confidence score " + std::to_string(vres->confidence_score);
    vres->verifier_reasoning = strdup(reasoning.c_str());

    return vres;
}

void antigravity_free_rollout_result(antigravity_rollout_result_t* result) {
    if (!result) return;
    if (result->candidates) {
        for (uint32_t i = 0; i < result->candidate_count; i++) {
            if (result->candidates[i].trace_text) {
                free(result->candidates[i].trace_text);
            }
        }
        delete[] result->candidates;
    }
    delete result;
}

void antigravity_free_verification_result(antigravity_verification_result_t* result) {
    if (!result) return;
    if (result->verifier_reasoning) {
        free(result->verifier_reasoning);
    }
    delete result;
}

void antigravity_sanitize_buffers(antigravity_engine_t engine) {
    if (engine && engine->ctx) {
        AntigravityEngineSanitizeBuffers(engine->ctx);
    }
}

} // extern "C"
