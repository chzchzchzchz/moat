// The sampler's top-p stopped sorting the whole vocabulary, and it now computes the raw
// log-probability the engine used to compute in two extra passes. Both are changes to the
// code every generated token passes through, so they are held to bit-for-bit agreement with
// what they replace — not to "close enough".
//
//   1. On distributions without exact ties, the new sampler draws the SAME token as the old
//      one from the same RNG state, every time.
//   2. With ties — which FP16 logits make common — the fast path keeps the same nucleus as a
//      full sort under the same total order, so the result does not depend on the algorithm.
//   3. The fused raw log-prob is bit-identical to the engine's old two-pass loop on finite
//      logits, at temperature 0.7 and at exactly 1.
//   4. Where the old loop went NaN, the new value is finite and correct, or -INF when
//      nothing was sampleable.
#include "sampling_oracle_old.h"
#include "../src/sampling.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace antigravity;

static int failures = 0;
static int checks = 0;
static void check(bool ok, const char* what) {
    checks++;
    std::printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

// The engine's old log-prob computation, as it stood in generate() and generateMultimodal().
static float engineOldLogprob(const float* logits, int V, int tok) {
    float max_logit = -1e9f;
    for (int i = 0; i < V; i++) {
        float v = logits[i];
        if (v > max_logit) max_logit = v;
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < V; i++) sum_exp += expf(logits[i] - max_logit);
    return logits[tok] - max_logit - logf(sum_exp);
}

// A reference top-p sampler: full std::sort, but with the new TOTAL order. Used to check the
// partial-sort path against when ties are present.
static int32_t fullSortReference(const float* logits, int V, float T, float top_p,
                                 std::mt19937& rng) {
    std::vector<float> p((size_t)V);
    const float inv = 1.0f / std::max(T, 1e-6f);
    float mx = -INFINITY;
    for (int i = 0; i < V; i++) {
        const float s = logits[i] * inv;
        p[(size_t)i] = std::isfinite(s) ? s : -INFINITY;
        if (std::isfinite(s) && s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        p[(size_t)i] = p[(size_t)i] == -INFINITY ? 0.0f : std::exp(p[(size_t)i] - mx);
        sum += p[(size_t)i];
    }
    for (int i = 0; i < V; i++) p[(size_t)i] /= sum;
    std::vector<int> order((size_t)V);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (p[(size_t)a] != p[(size_t)b]) return p[(size_t)a] > p[(size_t)b];
        return a < b;
    });
    float cum = 0.0f;
    int cutoff = V;
    for (int i = 0; i < V; i++) {
        cum += p[(size_t)order[(size_t)i]];
        if (cum >= top_p) { cutoff = i + 1; break; }
    }
    for (int i = cutoff; i < V; i++) p[(size_t)order[(size_t)i]] = 0.0f;
    float renorm = 0.0f;
    for (int i = 0; i < V; i++) renorm += p[(size_t)i];
    if (renorm > 0.0f && std::isfinite(renorm)) for (int i = 0; i < V; i++) p[(size_t)i] /= renorm;
    std::discrete_distribution<int> d(p.begin(), p.end());
    return (int32_t)d(rng);
}

int main() {
    // ---- 1. same token as the old sampler, distributions without exact ties ----------
    {
        int draws = 0, mismatches = 0;
        for (int V : {50, 1000, 32000}) {
            for (float T : {0.3f, 0.7f, 1.0f, 1.5f}) {
                for (float top_p : {0.5f, 0.9f, 0.95f, 1.0f}) {
                    for (int trial = 0; trial < 6; trial++) {
                        std::mt19937 g((uint32_t)(V * 131 + trial * 7 + (int)(T * 100) + (int)(top_p * 1000)));
                        std::normal_distribution<float> n(0.f, 3.f);
                        std::vector<float> logits((size_t)V);
                        // Distinct by construction: the index-dependent offset rules out
                        // exact ties among the logits themselves.
                        for (int i = 0; i < V; i++) logits[(size_t)i] = n(g) + (float)i * 1e-5f;
                        for (int seed = 0; seed < 8; seed++) {
                            std::mt19937 a((uint32_t)seed), b((uint32_t)seed);
                            SamplingStats sa, sb;
                            const int32_t t_old = old_impl::oldSampleTokenFromLogits(
                                logits.data(), V, T, top_p, a, sa);
                            const int32_t t_new = sampleTokenFromLogits(
                                logits.data(), V, T, top_p, b, sb);
                            draws++;
                            if (t_old != t_new) mismatches++;
                        }
                    }
                }
            }
        }
        std::printf("        (%d draws compared against the old sampler, %d differed)\n",
                    draws, mismatches);
        check(draws == 3 * 4 * 4 * 6 * 8, "every configuration was drawn from");
        check(mismatches == 0,
              "with no exact ties, the new sampler draws exactly the old one's token");
    }

    // ---- 2. heavy ties: the fast path keeps the same nucleus as a full sort -----------
    {
        int draws = 0, mismatches = 0;
        for (int V : {64, 1000, 32000}) {
            for (float top_p : {0.3f, 0.5f, 0.9f}) {
                for (int trial = 0; trial < 6; trial++) {
                    std::mt19937 g((uint32_t)(V + trial * 17 + (int)(top_p * 100)));
                    // Logits drawn from only 12 distinct values, so many tokens tie exactly,
                    // including at the nucleus boundary — the FP16 situation.
                    std::uniform_int_distribution<int> pick(0, 11);
                    std::vector<float> logits((size_t)V);
                    for (int i = 0; i < V; i++) logits[(size_t)i] = (float)pick(g) * 0.75f;
                    for (int seed = 0; seed < 8; seed++) {
                        std::mt19937 a((uint32_t)seed), b((uint32_t)seed);
                        SamplingStats sb;
                        const int32_t t_ref = fullSortReference(logits.data(), V, 0.7f, top_p, a);
                        const int32_t t_new = sampleTokenFromLogits(logits.data(), V, 0.7f, top_p, b, sb);
                        draws++;
                        if (t_ref != t_new) mismatches++;
                    }
                }
            }
        }
        std::printf("        (%d tied draws compared against a full sort, %d differed)\n",
                    draws, mismatches);
        check(mismatches == 0,
              "with exact ties, the partial sort keeps the same nucleus as a full sort");
    }

    // ---- 3. the fused raw log-prob equals the engine's old loop, bit for bit ----------
    {
        int compared = 0, differing = 0;
        float worst = 0.0f;
        for (int V : {32000, 151936}) {
            for (float T : {0.7f, 1.0f}) {
                for (int trial = 0; trial < 5; trial++) {
                    std::mt19937 g((uint32_t)(V + trial + (int)(T * 10)));
                    std::normal_distribution<float> n(0.f, 2.f);
                    std::vector<float> logits((size_t)V);
                    for (int i = 0; i < V; i++) logits[(size_t)i] = n(g);
                    std::mt19937 rng((uint32_t)trial);
                    SamplingStats st;
                    float lp = 0.0f;
                    const int32_t tok = sampleTokenFromLogits(logits.data(), V, T, 0.9f, rng, st, &lp);
                    const float want = engineOldLogprob(logits.data(), V, tok);
                    compared++;
                    if (std::memcmp(&lp, &want, sizeof(float)) != 0) {
                        differing++;
                        worst = std::max(worst, std::fabs(lp - want));
                    }
                }
            }
        }
        std::printf("        (%d log-probs compared, %d not bit-identical, worst |diff| %g)\n",
                    compared, differing, (double)worst);
        check(differing == 0,
              "the fused raw log-prob is bit-identical to the engine's old two-pass loop");
    }
    {
        // Temperature does not change what is reported: raw, not tempered, log-prob.
        std::vector<float> logits = {1.0f, 2.0f, 3.0f, 0.5f};
        std::mt19937 r1(3), r2(3);
        SamplingStats s;
        float lp_hot = 0, lp_cold = 0;
        const int32_t t1 = sampleTokenFromLogits(logits.data(), 4, 2.0f, 1.0f, r1, s, &lp_hot);
        const int32_t t2 = sampleTokenFromLogits(logits.data(), 4, 0.2f, 1.0f, r2, s, &lp_cold);
        check(std::fabs(lp_hot - engineOldLogprob(logits.data(), 4, t1)) < 1e-6f &&
              std::fabs(lp_cold - engineOldLogprob(logits.data(), 4, t2)) < 1e-6f,
              "the log-prob is the raw one whatever the temperature, as the engine reported");
    }
    {
        // Not asking for it changes nothing about the draw.
        std::vector<float> logits(1000);
        std::mt19937 g(9);
        std::normal_distribution<float> n(0.f, 2.f);
        for (auto& v : logits) v = n(g);
        std::mt19937 a(5), b(5);
        SamplingStats s;
        float lp = 0;
        const int32_t with = sampleTokenFromLogits(logits.data(), 1000, 0.7f, 0.9f, a, s, &lp);
        const int32_t without = sampleTokenFromLogits(logits.data(), 1000, 0.7f, 0.9f, b, s);
        check(with == without, "requesting the log-prob does not change the token drawn");
    }

    // ---- 4. non-finite logits ------------------------------------------------------------
    {
        // One NaN and one +Inf among finite logits. The old loop: max becomes +Inf, then
        // exp(Inf - Inf) is NaN, so the sum and the log-prob are NaN — and accumulated.
        std::vector<float> logits = {1.0f, NAN, 2.0f, INFINITY, 0.5f};
        const float old_lp_on_token0 = engineOldLogprob(logits.data(), 5, 0);
        check(std::isnan(old_lp_on_token0),
              "the old loop gives NaN when any logit is NaN or +Inf (the defect)");

        std::mt19937 rng(1);
        SamplingStats st;
        float lp = 0;
        const int32_t tok = sampleTokenFromLogits(logits.data(), 5, 1.0f, 1.0f, rng, st, &lp);
        check(tok == 0 || tok == 2 || tok == 4, "a finite token is drawn");
        check(std::isfinite(lp), "the new log-prob is finite");
        const std::vector<float> finite_only = {1.0f, 2.0f, 0.5f};
        const int idx = tok == 0 ? 0 : (tok == 2 ? 1 : 2);
        check(std::fabs(lp - engineOldLogprob(finite_only.data(), 3, idx)) < 1e-6f,
              "and equals the log-prob over the finite logits alone");
    }
    {
        std::vector<float> logits = {NAN, INFINITY, -INFINITY};
        std::mt19937 rng(1);
        SamplingStats st;
        float lp = 123.0f;
        sampleTokenFromLogits(logits.data(), 3, 0.7f, 0.9f, rng, st, &lp);
        check(lp == -INFINITY,
              "when nothing is sampleable the log-prob is -INF, so the channel ranks last");
        check(st.empty_distributions == 1, "and the empty distribution is still recorded");
    }
    {
        float lp = 123.0f;
        std::mt19937 rng(1);
        SamplingStats st;
        sampleTokenFromLogits((const float*)nullptr, 0, 0.7f, 0.9f, rng, st, &lp);
        check(lp == -INFINITY, "an empty vocabulary also reports -INF");
    }

    // ---- the nucleus boundary, where cumulative lands exactly on top_p ----------------
    // Random logits never make the running sum equal top_p exactly, so ">=" and ">" at the
    // boundary are indistinguishable on them — mutation testing showed the suite passing
    // with either. Four equal logits give probabilities of exactly 0.25 each, and 0.25 + 0.25
    // is exactly 0.5, so at top_p = 0.5 the nucleus is precisely two tokens. With the
    // index tie-break, those are tokens 0 and 1, and nothing else may ever be drawn.
    {
        std::vector<float> logits = {1.0f, 1.0f, 1.0f, 1.0f};
        bool only_first_two = true, saw0 = false, saw1 = false;
        for (int seed = 0; seed < 400; seed++) {
            std::mt19937 rng((uint32_t)seed);
            SamplingStats st;
            const int32_t t = sampleTokenFromLogits(logits.data(), 4, 1.0f, 0.5f, rng, st);
            if (t != 0 && t != 1) only_first_two = false;
            if (t == 0) saw0 = true;
            if (t == 1) saw1 = true;
        }
        check(only_first_two,
              "a cumulative exactly equal to top_p closes the nucleus there (>=, not >)");
        check(saw0 && saw1, "and both tokens inside it are reachable");
    }

    // ---- non-finite logits on the separate raw-sum path --------------------------------
    // At temperature exactly 1 the raw sum IS the tempered sum, so the test above never
    // touched the path that accumulates a separate raw sum. That path has its own exclusion
    // of non-finite logits, and the suite passed with it removed. Same check at T = 0.7.
    {
        std::vector<float> logits = {1.0f, NAN, 2.0f, INFINITY, 0.5f};
        const std::vector<float> finite_only = {1.0f, 2.0f, 0.5f};
        bool all_finite = true, all_correct = true;
        for (int seed = 0; seed < 50; seed++) {
            std::mt19937 rng((uint32_t)seed);
            SamplingStats st;
            float lp = 0;
            const int32_t tok = sampleTokenFromLogits(logits.data(), 5, 0.7f, 1.0f, rng, st, &lp);
            if (!std::isfinite(lp)) all_finite = false;
            const int idx = tok == 0 ? 0 : (tok == 2 ? 1 : 2);
            if (std::fabs(lp - engineOldLogprob(finite_only.data(), 3, idx)) > 1e-6f) all_correct = false;
        }
        check(all_finite, "at T = 0.7 too, a NaN or +Inf logit leaves the log-prob finite");
        check(all_correct, "and it equals the log-prob over the finite logits alone");
    }

    // ---- sampling the channels concurrently -------------------------------------------
    // The engine samples its channels one after another on one core. sampleChannels lets it
    // run them concurrently instead, and the claim is that this changes WHEN each channel is
    // computed and never WHAT. So run a whole multi-step decode three ways — serially, on real
    // threads, and serially in reverse order — and require the trajectories to be identical:
    // every token, every log-prob bit for bit, every channel's final RNG state, and the same
    // channels finishing at the same steps.
    {
        const int C = 8, V = 2000, steps = 40;
        // A fixed bank of logits per step and channel, the same for all three runs.
        std::vector<float> bank((size_t)steps * C * V);
        {
            std::mt19937 g(42);
            std::normal_distribution<float> n(0.f, 2.5f);
            for (auto& v : bank) v = n(g);
            // Make some channels emit the "EOS" token (id 7) eventually, so activity changes
            // mid-run and inactive channels are exercised.
            for (int t = 10; t < steps; t++)
                for (int c = 0; c < C; c += 3) bank[((size_t)t * C + c) * V + 7] = 40.0f;
        }

        struct Run {
            std::vector<std::vector<int32_t>> tokens;
            std::vector<std::vector<float>> logprobs;
            std::vector<std::mt19937> rngs;
            std::vector<uint64_t> non_finite;
        };
        auto run = [&](auto parallel_for) {
            Run r;
            r.tokens.assign(C, {});
            r.logprobs.assign(C, {});
            r.non_finite.assign(C, 0);
            for (int c = 0; c < C; c++) r.rngs.emplace_back(1000u + (uint32_t)c);
            std::vector<bool> active(C, true);       // the engine's type, on purpose
            for (int t = 0; t < steps; t++) {
                std::vector<int32_t> tok(C, -1);
                std::vector<float> lp(C, 0.0f);
                std::vector<SamplingStats> st(C);
                sampleChannels(bank.data() + (size_t)t * C * V, V, C,
                               [&](int c) { return (bool)active[(size_t)c]; },
                               0.7f, 0.9f, r.rngs.data(), tok.data(), lp.data(), st.data(),
                               parallel_for);
                // Phase two, serial, in channel order — as the engine does it.
                for (int c = 0; c < C; c++) {
                    r.non_finite[(size_t)c] += st[(size_t)c].non_finite_logits;
                    if (!active[(size_t)c]) continue;
                    r.tokens[(size_t)c].push_back(tok[(size_t)c]);
                    r.logprobs[(size_t)c].push_back(lp[(size_t)c]);
                    if (tok[(size_t)c] == 7) active[(size_t)c] = false;
                }
            }
            return r;
        };

        auto serial = [](int n, auto&& fn) { for (int i = 0; i < n; i++) fn(i); };
        auto reversed = [](int n, auto&& fn) { for (int i = n - 1; i >= 0; i--) fn(i); };
        auto threaded = [](int n, auto&& fn) {
            std::vector<std::thread> pool;
            for (int i = 0; i < n; i++) pool.emplace_back([&fn, i] { fn(i); });
            for (auto& th : pool) th.join();
        };

        const Run a = run(serial);
        const Run b = run(threaded);
        const Run c = run(reversed);

        // The three runs above all go through sampleChannels, so if sampleChannels itself were
        // wrong they would be wrong identically and agree with each other — comparing them
        // only proves the result does not depend on scheduling. Mutation testing showed it: a
        // sampleChannels that drew channel 0 twice passed. So also anchor to the loop the
        // engine had, written out directly: for each active channel in order, one call to the
        // sampler, with `continue` before the draw for an inactive one.
        Run ref;
        {
            ref.tokens.assign(C, {});
            ref.logprobs.assign(C, {});
            ref.non_finite.assign(C, 0);
            for (int ch = 0; ch < C; ch++) ref.rngs.emplace_back(1000u + (uint32_t)ch);
            std::vector<bool> active(C, true);
            for (int t = 0; t < steps; t++) {
                for (int ch = 0; ch < C; ch++) {
                    if (!active[(size_t)ch]) continue;
                    SamplingStats st;
                    float lp = -INFINITY;
                    const int32_t tok = sampleTokenFromLogits(
                        bank.data() + ((size_t)t * C + ch) * V, V, 0.7f, 0.9f,
                        ref.rngs[(size_t)ch], st, &lp);
                    ref.non_finite[(size_t)ch] += st.non_finite_logits;
                    ref.tokens[(size_t)ch].push_back(tok);
                    ref.logprobs[(size_t)ch].push_back(lp);
                    if (tok == 7) active[(size_t)ch] = false;
                }
            }
        }

        auto same = [&](const Run& x, const Run& y) {
            if (x.tokens != y.tokens) return false;
            for (int ch = 0; ch < C; ch++) {
                const auto& lx = x.logprobs[(size_t)ch];
                const auto& ly = y.logprobs[(size_t)ch];
                if (lx.size() != ly.size()) return false;
                if (!lx.empty() && std::memcmp(lx.data(), ly.data(), lx.size() * sizeof(float)) != 0)
                    return false;
                if (x.rngs[(size_t)ch] != y.rngs[(size_t)ch]) return false;
            }
            return x.non_finite == y.non_finite;
        };

        size_t total = 0, finished = 0;
        for (int ch = 0; ch < C; ch++) {
            total += a.tokens[(size_t)ch].size();
            if (!a.tokens[(size_t)ch].empty() && a.tokens[(size_t)ch].back() == 7) finished++;
        }
        std::printf("        (%zu tokens over %d steps and %d channels; %zu channels hit EOS "
                    "part-way)\n", total, steps, C, finished);
        check(finished > 0 && finished < (size_t)C,
              "some channels finish early, so inactive channels are actually exercised");
        check(same(ref, a), "sampleChannels, run serially, reproduces the engine's original "
                            "per-channel loop exactly");
        check(same(a, b), "sampled on real threads, every channel's trajectory is identical "
                          "to the serial one: tokens, log-probs bit for bit, RNG state");
        check(same(a, c), "and sampling the channels in reverse order changes nothing either");
    }
    {
        // An inactive channel must not draw: its RNG stays where it was and its output slot is
        // untouched, as when the old loop hit `continue` before sampling.
        const int C = 3, V = 50;
        std::vector<float> logits((size_t)C * V, 0.5f);
        std::vector<std::mt19937> rngs = {std::mt19937(1), std::mt19937(2), std::mt19937(3)};
        const std::mt19937 before = rngs[1];
        std::vector<int32_t> tok = {-1, -1, -1};
        std::vector<float> lp = {9.f, 9.f, 9.f};
        std::vector<SamplingStats> st(C);
        sampleChannels(logits.data(), V, C, [](int c) { return c != 1; }, 0.7f, 0.9f,
                       rngs.data(), tok.data(), lp.data(), st.data(),
                       [](int n, auto&& fn) { for (int i = 0; i < n; i++) fn(i); });
        check(rngs[1] == before, "an inactive channel's RNG does not advance");
        check(tok[1] == -1 && lp[1] == 9.f, "an inactive channel's outputs are left untouched");
        check(tok[0] >= 0 && tok[2] >= 0, "the active channels are sampled");
    }

    // ---- timing, reported rather than asserted (CI machines vary too much) -------------
    {
        const int V = 151936;
        std::mt19937 g(1);
        std::normal_distribution<float> n(0.f, 2.f);
        std::vector<float> logits((size_t)V);
        for (auto& v : logits) v = n(g);
        for (int k = 0; k < 20; k++) logits[(size_t)((k * 7919) % V)] = 12.f + k * 0.3f;
        const int reps = 20;
        SamplingStats st;
        std::mt19937 a(0), b(0);
        auto t0 = std::chrono::steady_clock::now();
        int32_t sink = 0;
        for (int r = 0; r < reps; r++) {
            const int32_t t = old_impl::oldSampleTokenFromLogits(logits.data(), V, 0.7f, 0.9f, a, st);
            sink ^= t;
            sink ^= (int32_t)engineOldLogprob(logits.data(), V, t);
        }
        auto t1 = std::chrono::steady_clock::now();
        float lp = 0;
        for (int r = 0; r < reps; r++) {
            sink ^= sampleTokenFromLogits(logits.data(), V, 0.7f, 0.9f, b, st, &lp);
        }
        auto t2 = std::chrono::steady_clock::now();
        const double old_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        const double new_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / reps;
        std::printf("        (V=151936, T=0.7, top_p=0.9: old sampler + old log-prob pass "
                    "%.2f ms, new %.2f ms per channel per token; %.1fx; sink %d)\n",
                    old_ms, new_ms, old_ms / new_ms, sink & 1);
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
