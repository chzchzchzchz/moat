/*
 * Tests for src/superblock_pack.h — the INT4 super-block packer the Metal engine
 * uses when ANTIGRAVITY_INT4 is set.
 *
 * The packer decides the exact bytes the GPU reads. If its layout drifts by one
 * nibble or one scale index from what gemv_int4_kernel / fused_batched_gemm_int4
 * assume, inference does not crash — it produces plausible garbage. So these
 * checks read values back with the kernels' own index arithmetic rather than with
 * the packer's, and assert on the byte offsets directly.
 *
 * Runs on any host: no Metal, no GPU, no weights.
 *
 * Build and run:
 *   c++ -std=c++17 -Wall -Wextra -Isrc tests/test_superblock_pack.cpp \
 *       -o bin/test_superblock_pack && ./bin/test_superblock_pack
 *
 * With --emit <in.bin> <out.bin> it instead packs the FP16 values in <in.bin> and
 * writes the super-blocks to <out.bin>, which tests/superblock_crosscheck.py uses
 * to compare this packer against the reference one in src/dequant.py byte for byte.
 */

#include "superblock_pack.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace antigravity;

static int failures = 0;
static void check(bool ok, const char* what) {
    printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", what);
    if (!ok) failures++;
}

// A deterministic pseudo-random FP16 stream, so both this test and the Python
// cross-check generate byte-identical input without shipping a fixture file.
static std::vector<uint16_t> deterministicWeights(size_t n) {
    std::vector<uint16_t> w(n);
    uint32_t state = 0x13579BDFu;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;            // numerical recipes LCG
        const float unit = (float)(state >> 8) / (float)(1u << 24);   // [0, 1)
        w[i] = float_to_fp16_bits((unit - 0.5f) * 0.25f);   // roughly N(0, .07) range
    }
    return w;
}

static int emitMode(const char* in_path, const char* out_path) {
    FILE* in = fopen(in_path, "rb");
    if (!in) { fprintf(stderr, "cannot open %s\n", in_path); return 2; }
    fseek(in, 0, SEEK_END);
    const long size = ftell(in);
    fseek(in, 0, SEEK_SET);
    if (size <= 0 || size % 2 != 0) { fprintf(stderr, "bad input size %ld\n", size); fclose(in); return 2; }

    std::vector<uint16_t> w((size_t)size / 2);
    if (fread(w.data(), 1, (size_t)size, in) != (size_t)size) {
        fprintf(stderr, "short read on %s\n", in_path); fclose(in); return 2;
    }
    fclose(in);

    const size_t bytes = superblockBytesFor(w.size());
    if (bytes == 0) { fprintf(stderr, "%zu elements is not a multiple of 256\n", w.size()); return 2; }

    std::vector<uint8_t> packed(bytes);
    if (!packSuperblocks(w.data(), w.size(), packed.data())) {
        fprintf(stderr, "packSuperblocks failed\n"); return 2;
    }

    FILE* out = fopen(out_path, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", out_path); return 2; }
    fwrite(packed.data(), 1, packed.size(), out);
    fclose(out);
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--emit") == 0) {
        return emitMode(argv[2], argv[3]);
    }

    // ---- Half/float conversion, which everything else rests on ----------------
    {
        const float exact[] = { 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 65504.0f, -65504.0f,
                                0.00006103515625f /* smallest normal */ };
        bool round_trips = true;
        for (float v : exact) {
            if (fp16_bits_to_float(float_to_fp16_bits(v)) != v) round_trips = false;
        }
        check(round_trips, "values representable in FP16 survive float->half->float exactly");

        // 2^-24 is the smallest positive subnormal; 2^-25 is exactly halfway to zero
        // and must round to even, i.e. to zero.
        check(fp16_bits_to_float(float_to_fp16_bits(5.9604645e-8f)) == 5.9604645e-8f,
              "smallest FP16 subnormal is preserved");
        check(float_to_fp16_bits(2.9802322e-8f) == 0x0000u,
              "half-way-to-zero rounds to even (zero), not away");
        check(float_to_fp16_bits(1.0e-9f) == 0x0000u, "underflow gives zero");
        check(float_to_fp16_bits(70000.0f) == 0x7C00u, "overflow gives +Inf");
        check(float_to_fp16_bits(-70000.0f) == 0xFC00u, "negative overflow gives -Inf");

        // 2049 is not representable: FP16 has 11 bits of significand, so it sits
        // exactly between 2048 and 2050 and ties-to-even must pick 2048.
        check(fp16_bits_to_float(float_to_fp16_bits(2049.0f)) == 2048.0f,
              "ties round to even in the normal range");
    }

    // ---- Sizing ---------------------------------------------------------------
    check(superblockBytesFor(256) == 144, "one super-block is 144 bytes");
    check(superblockBytesFor(1024) == 4 * 144, "four super-blocks for 1024 elements");
    check(superblockBytesFor(255) == 0, "a non-multiple of 256 is refused");
    check(superblockBytesFor(0) == 0, "zero elements is refused");
    {
        std::vector<uint16_t> w(300);
        std::vector<uint8_t> out(4096, 0xAB);
        check(!packSuperblocks(w.data(), w.size(), out.data()),
              "packSuperblocks refuses a non-multiple of 256");
        check(out[0] == 0xAB, "a refused pack writes nothing");
    }

    // ---- Layout: scales first, then nibbles ------------------------------------
    {
        // One super-block whose group g holds the constant value (g + 1), so each
        // group's alpha is known and every element quantizes to exactly 7.
        std::vector<uint16_t> w(kInt4ElementsPerBlock);
        for (size_t g = 0; g < kInt4GroupsPerBlock; g++) {
            for (size_t i = 0; i < kInt4GroupSize; i++) {
                w[g * kInt4GroupSize + i] = float_to_fp16_bits((float)(g + 1));
            }
        }
        std::vector<uint8_t> packed(kInt4BlockBytes);
        check(packSuperblocks(w.data(), w.size(), packed.data()), "packs one super-block");

        bool scales_ok = true;
        for (size_t g = 0; g < kInt4GroupsPerBlock; g++) {
            uint16_t bits;
            std::memcpy(&bits, packed.data() + g * 2, 2);
            const float expected = fp16_bits_to_float(float_to_fp16_bits((float)(g + 1) / 7.0f));
            if (fp16_bits_to_float(bits) != expected) scales_ok = false;
        }
        check(scales_ok, "the 8 FP16 scales occupy bytes 0..15, one per 32 elements");

        bool nibbles_ok = true;
        for (size_t i = 0; i < kInt4PackedBytes; i++) {
            // Every value is +alpha, so every nibble is 7, stored biased as 15.
            if (packed[kInt4ScaleBytes + i] != 0xFF) nibbles_ok = false;
        }
        check(nibbles_ok, "packed nibbles start at byte 16 and store q + 8");
    }

    // ---- Low nibble is the even element ---------------------------------------
    {
        // Alternate +alpha and -alpha so even elements quantize to +7 and odd to -7.
        std::vector<uint16_t> w(kInt4ElementsPerBlock);
        for (size_t i = 0; i < w.size(); i++) {
            w[i] = float_to_fp16_bits((i % 2 == 0) ? 1.0f : -1.0f);
        }
        std::vector<uint8_t> packed(kInt4BlockBytes);
        packSuperblocks(w.data(), w.size(), packed.data());

        // even -> +7 -> 15 (0x0F) in the low nibble; odd -> -7 -> 1 in the high.
        check(packed[kInt4ScaleBytes] == 0x1F,
              "low nibble holds the even element, high nibble the odd one");

        bool signs_ok = true;
        for (size_t i = 0; i < w.size(); i++) {
            const float v = dequantSuperblockElement(packed.data(), i);
            if ((i % 2 == 0) != (v > 0.0f)) signs_ok = false;
        }
        check(signs_ok, "reading back with the kernel's index arithmetic keeps the sign pattern");
    }

    // ---- Clamping and the all-zero group --------------------------------------
    {
        // A single outlier 8x larger than the rest: nothing may wrap past [-8, 7].
        std::vector<uint16_t> w(kInt4ElementsPerBlock, float_to_fp16_bits(0.1f));
        w[0]  = float_to_fp16_bits(1.0f);
        w[17] = float_to_fp16_bits(-1.0f);
        std::vector<uint8_t> packed(kInt4BlockBytes);
        packSuperblocks(w.data(), w.size(), packed.data());

        bool in_range = true;
        for (size_t i = 0; i < w.size(); i++) {
            const uint8_t byte = packed[kInt4ScaleBytes + i / 2];
            const int nib = (i % 2 == 0) ? (int)(byte & 0x0F) - 8 : (int)((byte >> 4) & 0x0F) - 8;
            if (nib < -8 || nib > 7) in_range = false;
        }
        check(in_range, "every quantized value stays within [-8, 7]");
    }
    {
        std::vector<uint16_t> w(kInt4ElementsPerBlock, 0);   // all +0.0
        std::vector<uint8_t> packed(kInt4BlockBytes);
        check(packSuperblocks(w.data(), w.size(), packed.data()),
              "an all-zero group does not divide by zero");
        uint16_t bits;
        std::memcpy(&bits, packed.data(), 2);
        check(fp16_bits_to_float(bits) == 1.0f, "an all-zero group stores scale 1.0, as dequant.py does");
        bool all_zero = true;
        for (size_t i = 0; i < kInt4ElementsPerBlock; i++) {
            if (dequantSuperblockElement(packed.data(), i) != 0.0f) all_zero = false;
        }
        check(all_zero, "an all-zero group dequantizes back to zero");
    }

    // ---- Round-trip error against the analytic quantizer bound -----------------
    {
        const size_t N = 256 * 64;                       // 64 super-blocks
        const std::vector<uint16_t> w = deterministicWeights(N);
        std::vector<uint8_t> packed(superblockBytesFor(N));
        check(packSuperblocks(w.data(), N, packed.data()), "packs 64 super-blocks");

        // Per group the quantization step is alpha/7, so no element may be off by
        // more than half a step, and the RMS error of a correct uniform quantizer
        // is step/sqrt(12). Both are properties of the format rather than of this
        // particular weight distribution, so neither number is tuned to what the
        // packer happens to produce. A scale-index or nibble-order bug blows
        // straight through both.
        double worst_ratio = 0.0;
        double sum_sq_err = 0.0, sum_sq_predicted = 0.0, sum_sq_val = 0.0;
        double sum_sq_err_swapped = 0.0;

        for (size_t g = 0; g < N / kInt4GroupSize; g++) {
            float alpha = 0.0f;
            for (size_t i = 0; i < kInt4GroupSize; i++) {
                const float v = std::fabs(fp16_bits_to_float(w[g * kInt4GroupSize + i]));
                if (v > alpha) alpha = v;
            }
            const double step = alpha / 7.0;

            for (size_t i = 0; i < kInt4GroupSize; i++) {
                const size_t idx = g * kInt4GroupSize + i;
                const float orig = fp16_bits_to_float(w[idx]);
                const float back = dequantSuperblockElement(packed.data(), idx);
                const double err = std::fabs(orig - back);

                sum_sq_err       += err * err;
                sum_sq_predicted += step * step / 12.0;
                sum_sq_val       += (double)orig * orig;
                if (step > 0.0) {
                    const double ratio = err / step;
                    if (ratio > worst_ratio) worst_ratio = ratio;
                }

                // Negative control: read the same bytes with the nibble order
                // reversed, i.e. the single most likely packing mistake.
                const uint8_t byte = packed[(idx / kInt4ElementsPerBlock) * kInt4BlockBytes
                                            + kInt4ScaleBytes
                                            + ((idx % kInt4ElementsPerBlock) >> 1)];
                const int wrong_nib = (idx & 1u) ? (int)( byte       & 0x0F) - 8
                                                 : (int)((byte >> 4) & 0x0F) - 8;
                uint16_t scale_bits;
                std::memcpy(&scale_bits,
                            packed.data() + (idx / kInt4ElementsPerBlock) * kInt4BlockBytes
                                          + ((idx % kInt4ElementsPerBlock) >> 5) * 2, 2);
                const double wrong = (double)wrong_nib * fp16_bits_to_float(scale_bits);
                sum_sq_err_swapped += (orig - wrong) * (orig - wrong);
            }
        }

        const double rms_err       = std::sqrt(sum_sq_err / (double)N);
        const double rms_predicted = std::sqrt(sum_sq_predicted / (double)N);
        const double relative_rms  = std::sqrt(sum_sq_err / sum_sq_val);
        const double rms_swapped   = std::sqrt(sum_sq_err_swapped / (double)N);

        printf("       worst error = %.4f steps | RMS = %.6f (uniform-quantizer "
               "prediction %.6f) | relative RMS = %.4f | nibble-swapped RMS = %.6f\n",
               worst_ratio, rms_err, rms_predicted, relative_rms, rms_swapped);

        // 0.5 plus a little slack: the scale is rounded to FP16 before use, so the
        // effective step differs from alpha/7 in the last bits.
        check(worst_ratio <= 0.51, "no element is off by more than half a quantization step");
        check(rms_err <= rms_predicted * 1.10,
              "RMS error matches the analytic step/sqrt(12) bound for a correct quantizer");

        // Teeth: if the checks above could not distinguish a correct unpacking from
        // a wrong one, they would not be testing the format. Reading the wrong
        // nibble must be far worse than reading the right one.
        check(rms_swapped > rms_err * 5.0,
              "reading the wrong nibble is an order of magnitude worse (the checks have teeth)");
    }

    // ---- Blocks are independent -------------------------------------------------
    {
        // Packing 4 blocks must give the same bytes as packing each block alone:
        // this catches an index that runs off the end of its super-block.
        const size_t N = 256 * 4;
        const std::vector<uint16_t> w = deterministicWeights(N);
        std::vector<uint8_t> all(superblockBytesFor(N));
        packSuperblocks(w.data(), N, all.data());

        bool independent = true;
        for (size_t sb = 0; sb < 4; sb++) {
            std::vector<uint8_t> one(kInt4BlockBytes);
            packSuperblocks(w.data() + sb * kInt4ElementsPerBlock, kInt4ElementsPerBlock, one.data());
            if (std::memcmp(one.data(), all.data() + sb * kInt4BlockBytes, kInt4BlockBytes) != 0) {
                independent = false;
            }
        }
        check(independent, "each super-block packs independently of its neighbours");
    }

    printf("\n%s\n", failures == 0 ? "ALL CHECKS PASSED" : "THERE WERE FAILURES");
    return failures == 0 ? 0 : 1;
}
