#pragma once
//
// Token sampling, kept free of Metal so it can be tested on any machine.
//
// It was not, and it held a defect that produces exactly the failure this project
// kept hitting: silent, plausible-looking garbage.
//
// std::discrete_distribution given NaN weights returns index 0. The old softmax fed
// it NaN whenever a single logit was non-finite: NaN fails every ordering comparison,
// so it never became the running maximum; expf(NaN - max) is NaN; the sum is NaN; and
// after normalising, every weight is NaN. Measured on the host — one NaN among 1,999
// healthy logits returned token 0 on 400 of 400 draws, and so did one +Inf.
//
// The consequence is worse than a wrong token. A forward pass producing NaN would
// emit the same token forever, which reads as a model that has collapsed rather than
// as a fault to fix, and the run completes and writes results.
//
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace antigravity {

struct SamplingStats {
    uint64_t non_finite_logits = 0;   // discarded, not propagated
    uint64_t empty_distributions = 0; // calls where nothing was sampleable
};

// Sample one token id from `logits`.
//
// Non-finite logits are given zero probability rather than poisoning the rest. When
// every logit is non-finite there is nothing to sample: the call records it in
// `stats` and returns 0, and the caller is expected to surface that rather than treat
// the 0 as a decision.
//
// If `raw_logprob` is non-null it receives the chosen token's log-probability under the
// UN-tempered distribution, log softmax(logits)[token] — the quantity the engine
// accumulates into channel_logprobs and the C API ranks candidates by. It used to be
// computed by the engine in two further full passes over the vocabulary after this call
// returned, with a fresh expf per element, and without the non-finite handling this
// function has: one NaN or +Inf logit made the sum NaN, and because the value is
// accumulated with +=, that channel's total stayed NaN for the rest of the sequence. A
// NaN never wins `chan_logprob > max_logprob`, and if every channel is NaN the C API keeps
// whatever index it started from — the same silent default to 0 this function was
// hardened against.
//
// Computed here in the passes that already read every logit. Non-finite logits are
// excluded exactly as they are from sampling. When nothing is sampleable the value is
// -INFINITY: the channel produced no real token, so it should rank last, not first and
// not unpredictably.
template <typename Logit>
inline int32_t sampleTokenFromLogits(const Logit* logits, int vocab_size,
                                     float temperature, float top_p,
                                     std::mt19937& rng, SamplingStats& stats,
                                     float* raw_logprob = nullptr) {
    if (raw_logprob) *raw_logprob = -INFINITY;
    if (vocab_size <= 0) {
        stats.empty_distributions++;
        return 0;
    }

    std::vector<float> probs((size_t)vocab_size);
    const float inv_temp = 1.0f / std::max(temperature, 1e-6f);

    float max_logit = -INFINITY;
    float raw_max = -INFINITY;          // over the un-tempered logits, for raw_logprob
    int n_finite = 0;
    for (int i = 0; i < vocab_size; i++) {
        const float raw = (float)logits[i];
        const float scaled = raw * inv_temp;
        if (std::isfinite(scaled)) {
            probs[(size_t)i] = scaled;
            n_finite++;
            if (scaled > max_logit) max_logit = scaled;
            if (raw > raw_max) raw_max = raw;
        } else {
            probs[(size_t)i] = -INFINITY;
            stats.non_finite_logits++;
        }
    }

    if (n_finite == 0) {
        stats.empty_distributions++;
        return 0;
    }

    // At temperature exactly 1 the tempered and raw distributions are the same numbers, so
    // the raw exp-sum is the tempered one and costs nothing. Otherwise it needs its own exp
    // per element — there is no way to recover sum(exp(l)) from sum(exp(l/T)) — but it
    // shares this pass rather than taking two more of its own.
    const bool want_raw = raw_logprob != nullptr;
    const bool raw_is_tempered = (inv_temp == 1.0f);
    float sum_exp = 0.0f;
    float raw_sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        const bool finite = probs[(size_t)i] != -INFINITY;
        probs[(size_t)i] = finite ? std::exp(probs[(size_t)i] - max_logit) : 0.0f;
        sum_exp += probs[(size_t)i];
        if (want_raw && !raw_is_tempered && finite) {
            raw_sum += std::exp((float)logits[i] - raw_max);
        }
    }
    if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
        stats.empty_distributions++;
        return 0;
    }
    for (int i = 0; i < vocab_size; i++) probs[(size_t)i] /= sum_exp;

    if (top_p < 1.0f && top_p > 0.0f) {
        // The nucleus is the shortest prefix of the tokens, most probable first, whose
        // probabilities sum to at least top_p. This used to std::sort all vocab_size indices
        // to find it. Measured on 151,936 logits at temperature 0.7, that sort made a call
        // 5.6x slower than the same call at top_p = 1.0 — 18 ms against 3.2 ms per channel
        // per token on the machine that measured it — when the nucleus it was looking for is
        // typically a few dozen tokens.
        //
        // partial_sort orders only the first k. Start small and widen until the prefix is
        // long enough to reach top_p: each attempt is O(V log k), k grows geometrically, and
        // a flat distribution that needs everything ends at k = V, which is the full sort the
        // old code always paid for.
        //
        // The comparator is a TOTAL order — probability descending, then index ascending —
        // where the old one compared probability alone. That matters more than it looks:
        // logits come out of the GPU as FP16, which has few enough distinct values that
        // exactly tied probabilities are common, and std::sort orders ties however its
        // implementation likes. A tie straddling the cutoff therefore decided which token was
        // kept differently under libc++ and libstdc++. With the tie broken by index, the
        // nucleus is the same set whatever sorts it, which is also what lets the fast path be
        // tested bit-for-bit against a full sort.
        std::vector<int> order((size_t)vocab_size);
        std::iota(order.begin(), order.end(), 0);
        auto ranks_before = [&](int a, int b) {
            const float pa = probs[(size_t)a], pb = probs[(size_t)b];
            if (pa != pb) return pa > pb;
            return a < b;
        };

        int cutoff = vocab_size;
        int k = std::min(vocab_size, 64);
        while (true) {
            std::partial_sort(order.begin(), order.begin() + k, order.end(), ranks_before);
            float cumulative = 0.0f;
            bool reached = false;
            for (int i = 0; i < k; i++) {
                cumulative += probs[(size_t)order[(size_t)i]];
                if (cumulative >= top_p) { cutoff = i + 1; reached = true; break; }
            }
            if (reached || k == vocab_size) break;   // k == V and not reached: keep all
            k = (k > vocab_size / 4) ? vocab_size : k * 4;
        }
        // Everything outside the nucleus. Beyond k, partial_sort leaves the order
        // unspecified, which does not matter: this is a set operation.
        for (int i = cutoff; i < vocab_size; i++) probs[(size_t)order[(size_t)i]] = 0.0f;

        // Guarded for the same reason as above: a zero or non-finite total would put
        // NaN back into every weight and land us right back at token 0.
        float renorm = 0.0f;
        for (int i = 0; i < vocab_size; i++) renorm += probs[(size_t)i];
        if (renorm > 0.0f && std::isfinite(renorm)) {
            for (int i = 0; i < vocab_size; i++) probs[(size_t)i] /= renorm;
        }
    }

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    const int32_t token = (int32_t)dist(rng);

    if (want_raw) {
        const float total = raw_is_tempered ? sum_exp : raw_sum;
        const float chosen = (float)logits[token];
        // The drawn token always has a finite logit (non-finite ones were given zero
        // probability), and total is at least exp(0) = 1 from the maximum itself.
        *raw_logprob = (std::isfinite(chosen) && total > 0.0f && std::isfinite(total))
                     ? chosen - raw_max - std::log(total)
                     : -INFINITY;
    }
    return token;
}

// Sample every active channel's next token, letting the caller decide how to run the
// channels — one after another, or concurrently.
//
// After each decode step the engine samples its channels one at a time on one core. With the
// sampler at about 1.2 ms per channel at a 32k vocabulary and 4.8 ms at 151,936, that is 10 to
// 39 ms per step for 8 channels, serial, while every other core waits. The channels are
// independent by construction: channel c reads only its own slice of the logits, draws only
// from its own RNG, and writes only its own slot of the outputs here. So running them
// concurrently changes only when each is computed, never what it computes, and each channel's
// token is the one the serial loop would have drawn.
//
// What is NOT independent stays out of this function, deliberately:
//   - The engine's running counters (non-finite logits, empty distributions) would race if
//     every channel incremented them. Each channel gets its own SamplingStats here, and the
//     caller sums them afterwards.
//   - `channel_active` is a std::vector<bool> in the engine, whose elements are packed bits:
//     two threads writing different channels' flags write the same byte. This only READS
//     activity, through `is_active`; deactivating a channel on EOS is the caller's job, after.
//   - Pushing tokens into per-channel histories and the total-token count happen after, in
//     channel order, exactly as before.
//
// An inactive channel is skipped entirely, so its RNG does not advance — as in the loop this
// replaces, where `continue` came before the draw. Its outputs are left as they were.
//
// `parallel_for(n, fn)` must call fn(i) exactly once for each i in [0, n), in any order and on
// any threads, and return only when all calls have finished.
template <typename Logit, typename IsActive, typename ParallelFor>
inline void sampleChannels(const Logit* logits_base, int vocab_size, int n_channels,
                           IsActive&& is_active, float temperature, float top_p,
                           std::mt19937* rngs, int32_t* tokens_out, float* logprobs_out,
                           SamplingStats* stats_out, ParallelFor&& parallel_for) {
    auto one = [&](int c) {
        if (!is_active(c)) return;
        tokens_out[c] = sampleTokenFromLogits(
            logits_base + (size_t)c * (size_t)vocab_size, vocab_size, temperature, top_p,
            rngs[c], stats_out[c], &logprobs_out[c]);
    };
    parallel_for(n_channels, one);
}

// Reproducible seeding, opt-in.
//
// The engine seeds every channel's RNG from std::random_device, so no two runs of a benchmark
// are the same and a re-run after a fix cannot be compared with the run before it sample for
// sample — the paired comparison that makes small effects measurable at all. But
// random_device is there for a reason, recorded in generateMultimodal: seeds used to be fixed
// constants, so EVERY CALL produced identical rollouts. A fixed seed must not bring that back.
//
// So the seed for channel c of the k-th generation call is derived from (base, k, c). The
// whole sequence of calls in a run is reproducible from `base`, and successive calls still
// differ, because k changes. Used only when ANTIGRAVITY_SEED is set; otherwise the engine
// seeds exactly as it did.
//
// The three values are mixed to 64 bits with splitmix64 and fed to std::seed_seq as two
// words, rather than truncated to the 32-bit integer mt19937::seed takes: distinct
// (call, channel) pairs would collide on 32 bits after about 65,000 of them by the birthday
// bound, and a benchmark run makes that many.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline uint64_t channelSeedValue(uint64_t base, uint64_t call_index, uint32_t channel) {
    return splitmix64(splitmix64(splitmix64(base) ^ call_index) ^ (uint64_t)channel);
}

inline void seedChannelRng(std::mt19937& rng, uint64_t base, uint64_t call_index,
                           uint32_t channel) {
    const uint64_t v = channelSeedValue(base, call_index, channel);
    std::seed_seq seq{(uint32_t)(v & 0xFFFFFFFFu), (uint32_t)(v >> 32)};
    rng.seed(seq);
}

// Parse ANTIGRAVITY_SEED: a non-empty run of decimal digits that fits in 64 bits. Anything
// else — empty, signed, trailing characters, overflow — is rejected rather than read as a
// different number, because a seed that silently became some other seed would make a run
// look reproducible while reproducing nothing.
inline bool parseSeed(const char* text, uint64_t& out) {
    if (text == nullptr || *text == '\0') return false;
    uint64_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return false;
        const uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

}  // namespace antigravity
