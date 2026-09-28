#pragma once
//
// Bounds on the engine's generation arguments.
//
// These live in a Metal-free header for two reasons: the public C API and the
// engine itself both need to apply them, and they can then be tested on a machine
// with no GPU. The checks matter because the failure they prevent is a write past
// the end of a Metal buffer, which does not fault — it corrupts whatever is next
// in the allocation and produces plausible output.
//
// Two limits, both of which were previously unenforced anywhere:
//
//   q_len_max    sizes every scratch buffer (max_q_rows in reinitBuffersAndRoPE).
//                Speculative decode builds a chunk of k_draft + 1 tokens and
//                forwards it in one pass, so q_len > q_len_max overflows all six
//                of them, by an amount the caller chooses. k_draft arrives from
//                AntigravityEngineNativeGenerateSpeculative unvalidated — the
//                boundary checked prompt_len and max_new_tokens and not this.
//
//   max_seq_len  sizes each layer's KV cache. Decode writes at seq_pos, which
//                advances once per token from prompt_len, so a prompt plus
//                generation longer than the cache walks off the end of it.
//
#include <cstdint>
#include <vector>
#include <utility>

namespace antigravity {

// Why an argument was rejected. Returned rather than logged so the caller can
// map it onto its own error code and say something specific.
enum class LimitError {
    Ok = 0,
    DraftChunkNegative,     // k_draft < 0
    DraftChunkTooLong,      // k_draft + 1 > q_len_max
    PromptEmpty,            // prompt_len <= 0
    PromptTooLong,          // prompt_len alone exceeds the KV cache
    SequenceTooLong,        // prompt_len + max_new_tokens exceeds the KV cache
};

inline const char* describe(LimitError error) {
    switch (error) {
        case LimitError::Ok:                 return "ok";
        case LimitError::DraftChunkNegative: return "k_draft must not be negative";
        case LimitError::DraftChunkTooLong:
            return "k_draft + 1 exceeds q_len_max, which sizes the scratch buffers";
        case LimitError::PromptEmpty:        return "prompt_len must be positive";
        case LimitError::PromptTooLong:      return "prompt_len exceeds max_seq_len";
        case LimitError::SequenceTooLong:
            return "prompt_len + max_new_tokens exceeds max_seq_len, the KV cache length";
    }
    return "unknown";
}

// A speculative chunk is the current token plus k_draft drafted ones, and that
// whole chunk is forwarded in a single pass whose row count is bounded by
// q_len_max. Rejecting equals-the-limit is not needed: k_draft + 1 == q_len_max
// exactly fills the buffers.
inline LimitError checkDraftChunk(int32_t k_draft, int32_t q_len_max) {
    if (k_draft < 0) return LimitError::DraftChunkNegative;
    if (q_len_max <= 0) return LimitError::DraftChunkTooLong;
    // Compare in 64-bit: k_draft + 1 overflows int32_t at INT32_MAX, which would
    // otherwise wrap negative and pass.
    if ((int64_t)k_draft + 1 > (int64_t)q_len_max) return LimitError::DraftChunkTooLong;
    return LimitError::Ok;
}

// The KV cache holds max_seq_len positions per layer. Prefill consumes prompt_len
// of them and decode one per generated token.
inline LimitError checkSequence(int32_t prompt_len, int32_t max_new_tokens,
                                int32_t max_seq_len) {
    if (prompt_len <= 0) return LimitError::PromptEmpty;
    if (prompt_len > max_seq_len) return LimitError::PromptTooLong;
    if (max_new_tokens < 0) return LimitError::SequenceTooLong;
    if ((int64_t)prompt_len + (int64_t)max_new_tokens > (int64_t)max_seq_len) {
        return LimitError::SequenceTooLong;
    }
    return LimitError::Ok;
}

// How many tokens can still be generated after a prompt of prompt_len. Callers
// that would rather truncate than refuse use this instead of checkSequence.
inline int32_t remainingCapacity(int32_t prompt_len, int32_t max_seq_len) {
    if (prompt_len < 0 || max_seq_len <= 0) return 0;
    if (prompt_len >= max_seq_len) return 0;
    return max_seq_len - prompt_len;
}

// How many prompt tokens to encode into one command buffer during prefill.
//
// The prefill loop used to create a command buffer per prompt token and block on
// waitUntilCompleted each time, so a 100-token prompt cost 100 CPU-GPU round trips before a
// single output token appeared. Several tokens can share one command buffer: nothing in that
// loop reads a GPU result back, and a serial MTLComputeCommandEncoder keeps the dispatches
// in encoding order. See the comment at the loop in transformer_engine.mm.
//
// It is chunked rather than made one buffer for the whole prompt because a command buffer
// holds every dispatch encoded into it until committed. At roughly a dozen dispatches per
// layer per token, a 2048-token prompt over 22 layers would encode half a million dispatches
// before anything started executing.
//
// Lives here, with the other bounds, because getting it wrong means prefill silently skips
// prompt tokens — a partially ignored prompt, which produces confident output that does not
// follow from its input. That is the failure this engine's history is made of, so the value
// and the iteration over it are both tested. Returns at least 1 for any n_layers, including
// zero or negative.
inline int prefillChunkTokens(int n_layers) {
    const int dispatches_per_token = n_layers > 0 ? n_layers * 12 : 1;
    const int budget = 8192 / (dispatches_per_token > 0 ? dispatches_per_token : 1);
    if (budget < 1) return 1;
    return budget > 64 ? 64 : budget;
}

// The [start, end) ranges a chunked prefill loop should visit to cover [0, total) exactly.
// Returned rather than open-coded so the coverage property is testable.
inline std::vector<std::pair<int, int>> prefillChunks(int total, int chunk) {
    std::vector<std::pair<int, int>> ranges;
    if (total <= 0) return ranges;
    if (chunk < 1) chunk = 1;
    for (int start = 0; start < total; start += chunk) {
        const int end = (start + chunk < total) ? start + chunk : total;
        ranges.emplace_back(start, end);
    }
    return ranges;
}

}  // namespace antigravity
