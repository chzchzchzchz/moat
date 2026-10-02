/*
 * End-to-end arithmetic check for the INT4 decode path, without a GPU.
 *
 * gemv_int4_kernel computes y[col] = sum_k x[k] * B[k * N + col], reading B from
 * super-blocks indexed by that same flat k * N + col. Whether that is correct
 * depends on something outside the kernel: loadTensor must hand it a buffer that
 * really is [K x N] row-major. safetensors stores projections as [out, in], so
 * the transposed load makes in_features the row stride — and if that reasoning is
 * wrong, or the K/N recorded for the buffer are swapped, nothing crashes. Every
 * index stays in bounds and the answer is simply wrong.
 *
 * So this reproduces the kernel's arithmetic against the packer's output and
 * compares it to a plain FP16 GEMV, and includes the transposed packing as a
 * negative control.
 *
 * Build and run from antigravity-engine/:
 *   c++ -std=c++17 -Wall -Wextra -Isrc tests/test_int4_gemv_math.cpp \
 *       -o bin/test_int4_gemv_math && ./bin/test_int4_gemv_math
 */

#include "superblock_pack.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace antigravity;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

static uint32_t g_state = 0x2468ACE0u;
static float nextUnit() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float)(g_state >> 8) / (float)(1u << 24);
}

// The kernel's inner loop, line for line, in host C++.
static void gemvInt4(const std::vector<uint16_t>& x, const std::vector<uint8_t>& blocks,
                     uint32_t K, uint32_t N, std::vector<float>& y) {
    y.assign(N, 0.0f);
    for (uint32_t col = 0; col < N; col++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            sum += fp16_bits_to_float(x[k]) * dequantSuperblockElement(blocks.data(), k * N + col);
        }
        y[col] = sum;
    }
}

int main() {
    // Shapes the engine actually uses, scaled down: K and N both multiples of 8,
    // and K * N a multiple of 256 so the weights pack exactly.
    const uint32_t K = 256, N = 64;

    std::vector<uint16_t> B(K * N), x(K);
    for (auto& v : B) v = float_to_fp16_bits((nextUnit() - 0.5f) * 0.1f);   // ~N(0, .03)
    for (auto& v : x) v = float_to_fp16_bits((nextUnit() - 0.5f) * 2.0f);   // activations

    check(superblockBytesFor(K * N) == (K * N / 256) * 144,
          "a [256 x 64] weight matrix packs into whole super-blocks");

    // FP16 reference, accumulated in float exactly as the kernel does.
    std::vector<float> y_ref(N, 0.0f);
    for (uint32_t col = 0; col < N; col++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; k++) {
            sum += fp16_bits_to_float(x[k]) * fp16_bits_to_float(B[k * N + col]);
        }
        y_ref[col] = sum;
    }

    std::vector<uint8_t> packed(superblockBytesFor(K * N));
    check(packSuperblocks(B.data(), B.size(), packed.data()), "packs the weight matrix");

    std::vector<float> y;
    gemvInt4(x, packed, K, N, y);

    // Error model: each of K products picks up an independent quantization error
    // of RMS step/sqrt(12), scaled by |x[k]|, so the output error grows as sqrt(K)
    // while the output itself grows as sqrt(K) too. The ratio is therefore roughly
    // the per-weight relative error, and does not depend on K. Predict it rather
    // than pick a threshold.
    double sum_sq_err = 0.0, sum_sq_ref = 0.0, sum_sq_predicted = 0.0;
    for (uint32_t col = 0; col < N; col++) {
        const double e = y[col] - y_ref[col];
        sum_sq_err += e * e;
        sum_sq_ref += (double)y_ref[col] * y_ref[col];
    }
    for (uint32_t g = 0; g < K * N / kInt4GroupSize; g++) {
        float alpha = 0.0f;
        for (uint32_t i = 0; i < kInt4GroupSize; i++) {
            const float v = std::fabs(fp16_bits_to_float(B[g * kInt4GroupSize + i]));
            if (v > alpha) alpha = v;
        }
        const double step = alpha / 7.0;
        sum_sq_predicted += kInt4GroupSize * step * step / 12.0;
    }
    double mean_sq_x = 0.0;
    for (uint32_t k = 0; k < K; k++) {
        const double v = fp16_bits_to_float(x[k]);
        mean_sq_x += v * v;
    }
    mean_sq_x /= K;
    // Per output element: K products, each with weight-error variance
    // (sum_sq_predicted / (K*N)) and activation power mean_sq_x.
    const double predicted_rms = std::sqrt(K * mean_sq_x * sum_sq_predicted / (double)(K * N));

    const double rms_err = std::sqrt(sum_sq_err / N);
    const double relative = std::sqrt(sum_sq_err / sum_sq_ref);
    printf("       GEMV RMS error = %.6f (predicted %.6f), relative to output = %.4f\n",
           rms_err, predicted_rms, relative);

    check(rms_err <= predicted_rms * 1.20,
          "INT4 GEMV error matches what quantization alone predicts");

    // Orientation. Packing B transposed is the wiring mistake this whole test is
    // for: the shapes still fit, every index is in range, and the result is junk.
    {
        std::vector<uint16_t> Bt(K * N);
        for (uint32_t k = 0; k < K; k++)
            for (uint32_t n = 0; n < N; n++)
                Bt[n * K + k] = B[k * N + n];

        std::vector<uint8_t> packed_t(superblockBytesFor(K * N));
        packSuperblocks(Bt.data(), Bt.size(), packed_t.data());

        std::vector<float> y_t;
        gemvInt4(x, packed_t, K, N, y_t);

        double sum_sq_err_t = 0.0;
        for (uint32_t col = 0; col < N; col++) {
            const double e = y_t[col] - y_ref[col];
            sum_sq_err_t += e * e;
        }
        const double relative_t = std::sqrt(sum_sq_err_t / sum_sq_ref);
        printf("       transposed packing: relative error = %.4f (correct: %.4f)\n",
               relative_t, relative);
        check(relative_t > relative * 10.0,
              "packing the weights transposed is unmistakably wrong (the test has teeth)");
    }

    // Column independence: changing one column of B must move only that output.
    {
        std::vector<uint16_t> B2 = B;
        for (uint32_t k = 0; k < K; k++) B2[k * N + 7] = float_to_fp16_bits(0.0f);

        std::vector<uint8_t> packed2(superblockBytesFor(K * N));
        packSuperblocks(B2.data(), B2.size(), packed2.data());
        std::vector<float> y2;
        gemvInt4(x, packed2, K, N, y2);

        check(std::fabs(y2[7]) < std::fabs(y[7]) * 0.01 + 1e-6,
              "zeroing column 7 zeroes output 7");

        // The other columns share super-blocks with column 7, so their scales can
        // shift; they must still track the reference far better than chance.
        double sum_sq = 0.0, sum_sq_ref_others = 0.0;
        for (uint32_t col = 0; col < N; col++) {
            if (col == 7) continue;
            const double e = y2[col] - y_ref[col];
            sum_sq += e * e;
            sum_sq_ref_others += (double)y_ref[col] * y_ref[col];
        }
        check(std::sqrt(sum_sq / sum_sq_ref_others) < 0.25,
              "zeroing one column leaves the other outputs essentially unchanged");
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
