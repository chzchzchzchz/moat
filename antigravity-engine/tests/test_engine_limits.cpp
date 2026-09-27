/*
 * Tests for src/engine_limits.h.
 *
 * Each of these bounds stands between a caller-supplied integer and a write past
 * the end of a Metal buffer. Nothing enforced them before:
 * AntigravityEngineNativeGenerateSpeculative validated prompt_len and
 * max_new_tokens and passed k_draft straight through, and nothing anywhere
 * compared prompt_len + max_new_tokens against the KV cache length.
 *
 * Build and run from antigravity-engine/:
 *   c++ -std=c++17 -Wall -Wextra -Isrc tests/test_engine_limits.cpp \
 *       -o bin/test_engine_limits && ./bin/test_engine_limits
 */

#include "engine_limits.h"

#include <climits>
#include <cstdio>
#include <cstring>
#include <initializer_list>

using namespace antigravity;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

int main() {
    const int32_t Q_LEN_MAX = 64;       // TransformerConfig::q_len_max
    const int32_t MAX_SEQ   = 2048;     // TransformerConfig::max_seq_len

    // ---- Draft chunk -------------------------------------------------------
    check(checkDraftChunk(0, Q_LEN_MAX) == LimitError::Ok,
          "k_draft = 0 is allowed (speculation off, chunk is the current token alone)");
    check(checkDraftChunk(4, Q_LEN_MAX) == LimitError::Ok,
          "a typical k_draft of 4 is allowed");
    check(checkDraftChunk(63, Q_LEN_MAX) == LimitError::Ok,
          "k_draft = 63 exactly fills q_len_max = 64 and is allowed");
    check(checkDraftChunk(64, Q_LEN_MAX) == LimitError::DraftChunkTooLong,
          "k_draft = 64 needs 65 rows and is refused");
    check(checkDraftChunk(1000, Q_LEN_MAX) == LimitError::DraftChunkTooLong,
          "a wildly oversized k_draft is refused");
    check(checkDraftChunk(-1, Q_LEN_MAX) == LimitError::DraftChunkNegative,
          "a negative k_draft is refused rather than silently doing nothing");

    // INT32_MAX + 1 wraps negative in int32 arithmetic, which would pass a naive
    // check. This is why the comparison is done in 64-bit.
    check(checkDraftChunk(INT_MAX, Q_LEN_MAX) == LimitError::DraftChunkTooLong,
          "k_draft = INT32_MAX is refused, not wrapped to a negative");

    check(checkDraftChunk(0, 0) == LimitError::DraftChunkTooLong,
          "a zero q_len_max leaves no room for even one row");

    // ---- Sequence length ---------------------------------------------------
    check(checkSequence(10, 100, MAX_SEQ) == LimitError::Ok,
          "a short prompt with room to generate is allowed");
    check(checkSequence(2048, 0, MAX_SEQ) == LimitError::Ok,
          "a prompt that exactly fills the cache with nothing generated is allowed");
    check(checkSequence(2048, 1, MAX_SEQ) == LimitError::SequenceTooLong,
          "one token past a full cache is refused");
    check(checkSequence(2049, 0, MAX_SEQ) == LimitError::PromptTooLong,
          "a prompt longer than the cache is refused, and named as the prompt");
    check(checkSequence(2000, 100, MAX_SEQ) == LimitError::SequenceTooLong,
          "a prompt that fits but leaves too little room is refused");
    check(checkSequence(0, 10, MAX_SEQ) == LimitError::PromptEmpty,
          "an empty prompt is refused");
    check(checkSequence(-5, 10, MAX_SEQ) == LimitError::PromptEmpty,
          "a negative prompt_len is refused");
    check(checkSequence(10, INT_MAX, MAX_SEQ) == LimitError::SequenceTooLong,
          "max_new_tokens = INT32_MAX is refused, not wrapped");
    check(checkSequence(INT_MAX, INT_MAX, MAX_SEQ) == LimitError::PromptTooLong,
          "both arguments at INT32_MAX is refused");

    // ---- Remaining capacity ------------------------------------------------
    check(remainingCapacity(100, MAX_SEQ) == 1948, "capacity left after a 100-token prompt");
    check(remainingCapacity(MAX_SEQ, MAX_SEQ) == 0, "a full cache leaves no capacity");
    check(remainingCapacity(MAX_SEQ + 1, MAX_SEQ) == 0,
          "an over-long prompt reports zero capacity, never a negative");
    check(remainingCapacity(-1, MAX_SEQ) == 0, "a negative prompt_len reports zero");
    check(remainingCapacity(10, 0) == 0, "a zero-length cache reports zero");

    // Consistency: anything checkSequence accepts must fit in remainingCapacity.
    bool consistent = true;
    for (int32_t p = 1; p <= 40; p++) {
        for (int32_t n = 0; n <= 40; n++) {
            if (checkSequence(p, n, 32) == LimitError::Ok && n > remainingCapacity(p, 32)) {
                consistent = false;
            }
        }
    }
    check(consistent, "checkSequence and remainingCapacity agree on every small case");

    // ---- The growing-prefix loop generateMCTS now relies on -----------------
    {
        // generateMCTS appends a chunk to its prefix each round, so the cache fills
        // part-way through the search. It asks for min(chunk_tokens, room) and stops
        // at room <= 0. Simulate that: it must terminate, never request more than
        // fits, and never exceed the cache.
        const int32_t cache = 128, chunk = 30, rounds = 100;
        int32_t prefix = 10;
        int32_t iterations = 0;
        bool never_overran = true, never_over_asked = true;

        for (int32_t r = 0; r < rounds; r++) {
            const int32_t room = remainingCapacity(prefix, cache);
            if (room <= 0) break;
            const int32_t ask = (chunk < room) ? chunk : room;
            if (ask > room) never_over_asked = false;
            if (checkSequence(prefix, ask, cache) != LimitError::Ok) never_over_asked = false;
            prefix += ask;
            if (prefix > cache) never_overran = false;
            iterations++;
        }

        check(iterations < rounds, "the search terminates before exhausting its rounds");
        check(never_over_asked, "every request fits the remaining cache");
        check(never_overran, "the prefix never grows past the cache");
        check(prefix == cache, "the loop fills the cache exactly, then stops");
    }
    {
        // A prompt already at or past the cache must stop the search immediately
        // rather than issuing a single doomed request.
        check(remainingCapacity(2048, 2048) == 0, "a prompt filling the cache leaves no round");
        check(remainingCapacity(9999, 2048) == 0, "an over-long prompt leaves no round");
    }

    // ---- Messages ----------------------------------------------------------
    check(std::strcmp(describe(LimitError::Ok), "ok") == 0, "describe(Ok) is \"ok\"");
    bool described = true;
    for (LimitError e : {LimitError::DraftChunkNegative, LimitError::DraftChunkTooLong,
                         LimitError::PromptEmpty, LimitError::PromptTooLong,
                         LimitError::SequenceTooLong}) {
        if (std::strcmp(describe(e), "unknown") == 0 || describe(e)[0] == '\0') described = false;
    }
    check(described, "every error names what was wrong, for the caller's log");

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
