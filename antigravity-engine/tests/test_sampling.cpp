/*
 * Tests for src/sampling.h.
 *
 * The bug these exist for: std::discrete_distribution given NaN weights returns index
 * 0, and the old softmax handed it NaN whenever a single logit was non-finite. One NaN
 * among 1,999 healthy logits returned token 0 on 400 of 400 draws, silently. A forward
 * pass producing NaN would therefore emit the same token forever and the run would
 * complete and write results — which is indistinguishable from a model that has
 * collapsed, and is the shape of the 587-problem checkpoint in this repository where
 * one character accounts for every degenerate generation.
 *
 * Build and run from antigravity-engine/:
 *   c++ -std=c++17 -Wall -Wextra -Isrc tests/test_sampling.cpp \
 *       -o bin/test_sampling && ./bin/test_sampling
 */

#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

using namespace antigravity;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

static std::vector<float> healthyLogits(int n, unsigned seed = 7) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> v((size_t)n);
    for (auto& x : v) x = nd(gen);
    return v;
}

struct Draws {
    std::set<int32_t> distinct;
    SamplingStats stats;
};

static Draws draw(const std::vector<float>& logits, int n = 400,
                  float temperature = 0.7f, float top_p = 0.9f) {
    Draws d;
    std::mt19937 rng(12345);
    for (int i = 0; i < n; i++) {
        d.distinct.insert(sampleTokenFromLogits(logits.data(), (int)logits.size(),
                                                temperature, top_p, rng, d.stats));
    }
    return d;
}

int main() {
    const int V = 2000;

    // ---- Healthy behaviour -------------------------------------------------
    {
        Draws d = draw(healthyLogits(V));
        check(d.distinct.size() > 20, "healthy logits produce a varied distribution");
        check(d.stats.non_finite_logits == 0 && d.stats.empty_distributions == 0,
              "healthy logits record no anomalies");
    }
    {
        std::vector<float> peaked((size_t)V, -10.0f);
        peaked[1234] = 20.0f;
        Draws d = draw(peaked);
        check(d.distinct.size() == 1 && *d.distinct.begin() == 1234,
              "a dominant logit is selected, so the sampler still concentrates");
    }
    {
        // Uniform logits must spread, which also proves the draws are not constant for
        // some unrelated reason.
        Draws d = draw(std::vector<float>((size_t)V, 0.0f));
        check(d.distinct.size() > 100, "uniform logits spread across many tokens");
    }

    // ---- The defect --------------------------------------------------------
    {
        std::vector<float> one_nan = healthyLogits(V);
        one_nan[500] = std::nanf("");
        Draws d = draw(one_nan);
        check(d.distinct.size() > 20,
              "ONE NaN no longer collapses the distribution to a single token");
        check(d.distinct.count(0) == 0 || d.distinct.size() > 1,
              "the result is not token 0 on every draw");
        check(d.stats.non_finite_logits == 400,
              "the discarded NaN is counted, once per call, so a run cannot look clean");
        check(d.stats.empty_distributions == 0,
              "one bad logit is not treated as a failed distribution");
    }
    {
        std::vector<float> one_inf = healthyLogits(V);
        one_inf[900] = INFINITY;
        Draws d = draw(one_inf);
        check(d.distinct.size() > 20, "one +Inf does not collapse the distribution");
        check(d.stats.non_finite_logits == 400, "the +Inf is counted");
    }
    {
        std::vector<float> one_ninf = healthyLogits(V);
        one_ninf[123] = -INFINITY;
        Draws d = draw(one_ninf);
        check(d.distinct.size() > 20, "one -Inf does not collapse the distribution");
        check(d.distinct.count(123) == 0, "a -Inf logit is never selected");
    }

    // ---- Total failure is reported, not disguised ---------------------------
    {
        Draws d = draw(std::vector<float>((size_t)V, std::nanf("")), 10);
        check(d.stats.empty_distributions == 10,
              "all-NaN logits are recorded as empty distributions every call");
    }
    {
        Draws d = draw(std::vector<float>((size_t)V, -INFINITY), 10);
        check(d.stats.empty_distributions == 10, "all -Inf is recorded as empty");
    }
    {
        SamplingStats stats;
        std::mt19937 rng(1);
        float none = 0.0f;
        check(sampleTokenFromLogits(&none, 0, 0.7f, 0.9f, rng, stats) == 0
              && stats.empty_distributions == 1,
              "a zero-length vocabulary is recorded rather than read out of bounds");
    }

    // ---- Parameters still do what they say ---------------------------------
    {
        // Near-greedy temperature must concentrate far more than a warm one.
        std::vector<float> logits = healthyLogits(V);
        check(draw(logits, 400, 0.01f, 1.0f).distinct.size()
              < draw(logits, 400, 2.0f, 1.0f).distinct.size(),
              "a low temperature concentrates relative to a high one");
        // A tight nucleus must admit fewer tokens than an open one.
        check(draw(logits, 400, 1.0f, 0.10f).distinct.size()
              <= draw(logits, 400, 1.0f, 1.0f).distinct.size(),
              "a tight top_p admits no more tokens than an open one");
    }
    {
        // Determinism: the same seed and logits must give the same sequence, or no
        // result from this engine is reproducible.
        std::vector<float> logits = healthyLogits(V);
        SamplingStats s1, s2;
        std::mt19937 r1(99), r2(99);
        bool same = true;
        for (int i = 0; i < 50; i++) {
            if (sampleTokenFromLogits(logits.data(), V, 0.7f, 0.9f, r1, s1)
                != sampleTokenFromLogits(logits.data(), V, 0.7f, 0.9f, r2, s2)) same = false;
        }
        check(same, "the same seed yields the same tokens");
    }

    // ---- reproducible seeding (ANTIGRAVITY_SEED) ----------------------------------------
    {
        // The same (base, call, channel) always gives the same stream.
        std::mt19937 a, b;
        antigravity::seedChannelRng(a, 42, 3, 5);
        antigravity::seedChannelRng(b, 42, 3, 5);
        check(a == b, "the same base, call and channel give the same RNG state");

        // The reason random_device was introduced: fixed seeds made every call produce
        // identical rollouts. Successive calls under one base seed must differ.
        std::mt19937 call0, call1;
        antigravity::seedChannelRng(call0, 42, 0, 0);
        antigravity::seedChannelRng(call1, 42, 1, 0);
        check(call0 != call1, "successive calls under one seed get different streams, so a "
                              "fixed seed does not bring back identical rollouts");

        std::mt19937 ch0, ch1;
        antigravity::seedChannelRng(ch0, 42, 0, 0);
        antigravity::seedChannelRng(ch1, 42, 0, 1);
        check(ch0 != ch1, "channels within one call get different streams");

        std::mt19937 base1, base2;
        antigravity::seedChannelRng(base1, 1, 0, 0);
        antigravity::seedChannelRng(base2, 2, 0, 0);
        check(base1 != base2, "different base seeds give different streams");

        // No collisions over a benchmark-sized grid. Checked on the 64-bit value the seed_seq
        // is built from; truncated to the 32 bits mt19937::seed takes, this many pairs would
        // be expected to collide (birthday bound near 65,000).
        std::set<uint64_t> seen;
        for (uint64_t call = 0; call < 20000; call++)
            for (uint32_t ch = 0; ch < 8; ch++)
                seen.insert(antigravity::channelSeedValue(7, call, ch));
        check(seen.size() == 20000u * 8u,
              "160,000 (call, channel) pairs get 160,000 distinct seeds");

        // Adjacent inputs must not give related streams: compare the first draws.
        std::set<uint32_t> firsts;
        for (uint32_t ch = 0; ch < 64; ch++) {
            std::mt19937 r;
            antigravity::seedChannelRng(r, 0, 0, ch);
            firsts.insert(r());
        }
        check(firsts.size() == 64, "64 adjacent channels' first draws are all distinct");
    }
    {
        uint64_t v = 99;
        check(antigravity::parseSeed("0", v) && v == 0, "seed 0 parses");
        check(antigravity::parseSeed("12345", v) && v == 12345, "a decimal seed parses");
        check(antigravity::parseSeed("18446744073709551615", v) && v == UINT64_MAX,
              "the largest 64-bit seed parses");
        check(!antigravity::parseSeed("18446744073709551616", v),
              "an overflowing seed is refused, not wrapped into a different seed");
        check(!antigravity::parseSeed("", v), "an empty seed is refused");
        check(!antigravity::parseSeed(nullptr, v), "an unset seed is refused");
        check(!antigravity::parseSeed("-1", v), "a negative seed is refused");
        check(!antigravity::parseSeed("42abc", v), "trailing characters are refused");
        check(!antigravity::parseSeed(" 42", v), "a leading space is refused");
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
