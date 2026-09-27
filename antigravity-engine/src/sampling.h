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
template <typename Logit>
inline int32_t sampleTokenFromLogits(const Logit* logits, int vocab_size,
                                     float temperature, float top_p,
                                     std::mt19937& rng, SamplingStats& stats) {
    if (vocab_size <= 0) {
        stats.empty_distributions++;
        return 0;
    }

    std::vector<float> probs((size_t)vocab_size);
    const float inv_temp = 1.0f / std::max(temperature, 1e-6f);

    float max_logit = -INFINITY;
    int n_finite = 0;
    for (int i = 0; i < vocab_size; i++) {
        const float scaled = (float)logits[i] * inv_temp;
        if (std::isfinite(scaled)) {
            probs[(size_t)i] = scaled;
            n_finite++;
            if (scaled > max_logit) max_logit = scaled;
        } else {
            probs[(size_t)i] = -INFINITY;
            stats.non_finite_logits++;
        }
    }

    if (n_finite == 0) {
        stats.empty_distributions++;
        return 0;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[(size_t)i] = (probs[(size_t)i] == -INFINITY)
                         ? 0.0f : std::exp(probs[(size_t)i] - max_logit);
        sum_exp += probs[(size_t)i];
    }
    if (!(sum_exp > 0.0f) || !std::isfinite(sum_exp)) {
        stats.empty_distributions++;
        return 0;
    }
    for (int i = 0; i < vocab_size; i++) probs[(size_t)i] /= sum_exp;

    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<int> order((size_t)vocab_size);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) { return probs[(size_t)a] > probs[(size_t)b]; });

        float cumulative = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumulative += probs[(size_t)order[(size_t)i]];
            if (cumulative >= top_p) { cutoff = i + 1; break; }
        }
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
    return (int32_t)dist(rng);
}

}  // namespace antigravity
