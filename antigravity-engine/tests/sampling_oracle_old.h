#pragma once
#include "sampling.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// A frozen copy of sampleTokenFromLogits as it was before top-p stopped sorting the whole
// vocabulary, kept as the oracle the new one is compared against. Do not "fix" it: its job
// is to be the old behaviour, exactly.
namespace old_impl {
template <typename Logit>
inline int32_t oldSampleTokenFromLogits(const Logit* logits, int vocab_size,
                                     float temperature, float top_p,
                                     std::mt19937& rng, antigravity::SamplingStats& stats) {
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
}  // namespace old_impl
