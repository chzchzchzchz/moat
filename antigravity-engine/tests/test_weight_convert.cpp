// Tests for src/weight_convert.h.
//
// Two things are checked here. First, bf16_to_fp16 — which every BF16 checkpoint's weights
// pass through, and which had never been run by a test; what "verified" meant was a Python
// reimplementation agreeing with itself. Second, the F32 path, which did not exist: F32
// tensors took a branch commented "already FP16 or compatible, direct copy".
#include "../src/weight_convert.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace antigravity;

static int failures = 0;
static int checks = 0;
static void check(bool condition, const char* what) {
    checks++;
    if (!condition) { failures++; std::printf("  FAIL: %s\n", what); }
}

// BF16 -> float is exact: the 16 bits are the top 16 bits of the float.
static float bf16_as_float(uint16_t b) {
    const uint32_t bits = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

int main() {
    // ---- the values that matter most ------------------------------------------------
    check(bf16_to_fp16(0x0000) == 0x0000, "BF16 +0 is FP16 +0");
    check(bf16_to_fp16(0x8000) == 0x8000, "BF16 -0 is FP16 -0, sign preserved");
    check(bf16_to_fp16(0x3F80) == 0x3C00, "BF16 1.0 is FP16 1.0");
    check(bf16_to_fp16(0xBF80) == 0xBC00, "BF16 -1.0 is FP16 -1.0");
    check(bf16_to_fp16(0x4000) == 0x4000, "BF16 2.0 is FP16 2.0");
    check(bf16_to_fp16(0x3F00) == 0x3800, "BF16 0.5 is FP16 0.5");
    check(bf16_to_fp16(0x7F80) == 0x7C00, "BF16 +Inf is FP16 +Inf");
    check(bf16_to_fp16(0xFF80) == 0xFC00, "BF16 -Inf is FP16 -Inf");

    // NaN must stay NaN. The original shifted the mantissa right by 4, so any NaN with a
    // mantissa below 0x10 came out as Inf.
    {
        const uint16_t quiet = bf16_to_fp16(0x7FC0);      // mantissa 0x40 -> 0x04
        check((quiet & 0x7C00) == 0x7C00 && (quiet & 0x03FF) != 0,
              "a quiet BF16 NaN stays a NaN");
        const uint16_t signalling = bf16_to_fp16(0x7F81); // mantissa 0x01 -> would be 0
        check((signalling & 0x7C00) == 0x7C00 && (signalling & 0x03FF) != 0,
              "a BF16 NaN with a small mantissa stays a NaN rather than becoming Inf");
        // Every NaN pattern, not just those two.
        int nan_became_inf = 0;
        for (uint32_t m = 1; m <= 0x7F; m++) {
            for (uint32_t sign = 0; sign < 2; sign++) {
                const uint16_t in = (uint16_t)((sign << 15) | (0xFF << 7) | m);
                const uint16_t out = bf16_to_fp16(in);
                if ((out & 0x7C00) == 0x7C00 && (out & 0x03FF) == 0) nan_became_inf++;
            }
        }
        check(nan_became_inf == 0, "no BF16 NaN pattern converts to Inf");
        std::printf("  (all %d BF16 NaN patterns stay NaN)\n", 2 * 0x7F);
    }

    // Overflow: BF16's range is far wider than FP16's, so large finite values must become
    // Inf rather than wrapping to something small.
    check(bf16_to_fp16(0x7F00) == 0x7C00,
          "a large finite BF16 overflows to FP16 +Inf, not to a small number");
    check(bf16_to_fp16(0xFF00) == 0xFC00, "a large negative BF16 overflows to -Inf");

    // ---- every one of the 65,536 BF16 patterns, against an independent reference -----
    // The reference converts BF16 to float exactly and then to FP16 with round-to-nearest.
    // Widening a 7-bit mantissa to 10 bits is lossless, so inside FP16's normal range the
    // two must agree bit for bit. Outside it they may differ, and this reports where.
    {
        int normal_range_mismatch = 0;
        int subnormal_mismatch = 0;
        int compared = 0;
        uint16_t first_bad_in = 0, first_bad_got = 0, first_bad_want = 0;

        for (uint32_t raw = 0; raw <= 0xFFFF; raw++) {
            const uint16_t in = (uint16_t)raw;
            const uint16_t got = bf16_to_fp16(in);
            const float as_float = bf16_as_float(in);

            if (std::isnan(as_float)) continue;            // NaN payloads checked above
            const uint16_t want = float_to_fp16_bits(as_float);
            compared++;
            if (got == want) continue;

            // Which regime is this? FP16 subnormals have a zero exponent field.
            if ((want & 0x7C00) == 0) {
                subnormal_mismatch++;
            } else {
                if (normal_range_mismatch == 0) {
                    first_bad_in = in; first_bad_got = got; first_bad_want = want;
                }
                normal_range_mismatch++;
            }
        }
        std::printf("  (compared %d non-NaN BF16 patterns against float_to_fp16_bits)\n",
                    compared);
        check(compared > 65000, "nearly all 65,536 patterns were compared");
        if (normal_range_mismatch != 0) {
            std::printf("    first normal-range mismatch: in=0x%04X got=0x%04X want=0x%04X\n",
                        first_bad_in, first_bad_got, first_bad_want);
        }
        check(normal_range_mismatch == 0,
              "bf16_to_fp16 is exact for every pattern landing in FP16's normal range");
        // Subnormals used to differ on 1024 patterns, every one by exactly +1 ulp,
        // because the conversion truncated where the reference rounds to nearest. The
        // subnormal path now rounds, so this must be exact too — everywhere, not just in
        // the normal range.
        std::printf("    subnormal-range differences: %d\n", subnormal_mismatch);
        check(subnormal_mismatch == 0,
              "bf16_to_fp16 is exact in the subnormal range too, now that it rounds");
    }

    // ---- the weight range a real checkpoint occupies --------------------------------
    // Transformer weights sit within about +-10, well inside FP16's normal range, so every
    // value there must round-trip exactly.
    {
        int inexact = 0, examined = 0;
        for (uint32_t raw = 0; raw <= 0xFFFF; raw++) {
            const uint16_t in = (uint16_t)raw;
            const float as_float = bf16_as_float(in);
            if (std::isnan(as_float) || std::isinf(as_float)) continue;
            const float mag = std::fabs(as_float);
            if (mag < 1e-4f || mag > 10.0f) continue;
            examined++;
            if (fp16_bits_to_float(bf16_to_fp16(in)) != as_float) inexact++;
        }
        std::printf("  (%d BF16 patterns in the weight range 1e-4..10)\n", examined);
        check(examined == 4254,
              "4254 BF16 patterns fall in the weight range (a measured count, not a bound)");
        check(inexact == 0, "every BF16 value in the weight range converts exactly");
    }

    // ---- dtype names ----------------------------------------------------------------
    check(sourceDtypeFromName("BF16") == SourceDtype::BF16, "BF16 is recognised");
    check(sourceDtypeFromName("bf16") == SourceDtype::BF16,
          "the lowercase spelling the old code accepted still works");
    check(sourceDtypeFromName("bfloat16") == SourceDtype::BF16, "bfloat16 is recognised");
    check(sourceDtypeFromName("F16") == SourceDtype::FP16, "F16 is recognised");
    check(sourceDtypeFromName("F32") == SourceDtype::FP32, "F32 is recognised");
    check(sourceDtypeFromName("F64") == SourceDtype::Unsupported,
          "F64 is unsupported rather than silently treated as FP16");
    check(sourceDtypeFromName("I8") == SourceDtype::Unsupported, "I8 is unsupported");
    check(sourceDtypeFromName("") == SourceDtype::Unsupported, "an empty dtype is unsupported");
    check(sourceElementSize(SourceDtype::FP32) == 4, "an F32 element is four bytes");
    check(sourceElementSize(SourceDtype::BF16) == 2, "a BF16 element is two bytes");
    check(sourceElementSize(SourceDtype::FP16) == 2, "an F16 element is two bytes");

    // ---- the F32 path, and what the old code did with it ----------------------------
    {
        const std::vector<float> weights = {
            0.5f, -0.25f, 1.0f, -2.0f, 0.125f, 3.75f, -0.001953125f, 7.0f
        };
        const size_t n = weights.size();

        // What the new code produces.
        std::vector<uint16_t> converted(n);
        for (size_t i = 0; i < n; i++) {
            converted[i] = elementAsFp16(weights.data(), SourceDtype::FP32, i);
        }
        bool all_exact = true;
        for (size_t i = 0; i < n; i++) {
            if (fp16_bits_to_float(converted[i]) != weights[i]) all_exact = false;
        }
        check(all_exact, "every F32 weight converts to exactly the same value in FP16");

        // What the OLD code produced: memcpy of min(n*2, n*4) = n*2 bytes, i.e. the first
        // half of the F32 data reinterpreted as FP16.
        std::vector<uint16_t> old_path(n, 0);
        std::memcpy(old_path.data(), weights.data(), n * sizeof(uint16_t));

        int old_correct = 0;
        for (size_t i = 0; i < n; i++) {
            if (old_path[i] == converted[i]) old_correct++;
        }
        check(old_correct == 0,
              "the old direct-copy path got NONE of the F32 weights right");
        std::printf("  (old F32 path: %zu of %zu weights correct)\n",
                    (size_t)old_correct, n);

        // And it only read half the tensor, so the second half was never touched at all.
        // Demonstrate by showing the old path's bytes come entirely from weights[0..n/2).
        const size_t bytes_read_old = n * sizeof(uint16_t);
        const size_t bytes_present  = n * sizeof(float);
        check(bytes_read_old * 2 == bytes_present,
              "the old path read exactly half the F32 tensor's bytes");

        // A worked example, so the failure is legible rather than a count.
        std::printf("    weights[0] = %g; new gives %g; old gives %g\n",
                    (double)weights[0],
                    (double)fp16_bits_to_float(converted[0]),
                    (double)fp16_bits_to_float(old_path[0]));
    }

    // ---- F32 values outside FP16's range --------------------------------------------
    {
        const float big = 1e30f;      // beyond FP16's max of 65504
        const uint16_t h = elementAsFp16(&big, SourceDtype::FP32, 0);
        check(std::isinf(fp16_bits_to_float(h)) && fp16_bits_to_float(h) > 0,
              "an F32 value past FP16's range becomes +Inf, not a small number");
        const float tiny = 1e-30f;    // below FP16's smallest subnormal
        const uint16_t t = elementAsFp16(&tiny, SourceDtype::FP32, 0);
        check(fp16_bits_to_float(t) == 0.0f, "an F32 value below FP16's range becomes zero");
        const float neg_big = -1e30f;
        const uint16_t nb = elementAsFp16(&neg_big, SourceDtype::FP32, 0);
        check(std::isinf(fp16_bits_to_float(nb)) && fp16_bits_to_float(nb) < 0,
              "a large negative F32 becomes -Inf");
    }

    // ---- indexing: element i must come from element i, for each dtype ---------------
    {
        const float f32[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        for (size_t i = 0; i < 4; i++) {
            check(fp16_bits_to_float(elementAsFp16(f32, SourceDtype::FP32, i)) == f32[i],
                  "F32 indexing reads the right element");
        }
        const uint16_t bf16[4] = {0x3F80, 0x4000, 0x4080, 0x4100};   // 1, 2, 4, 8
        const float expect[4] = {1.0f, 2.0f, 4.0f, 8.0f};
        for (size_t i = 0; i < 4; i++) {
            check(fp16_bits_to_float(elementAsFp16(bf16, SourceDtype::BF16, i)) == expect[i],
                  "BF16 indexing reads the right element");
        }
        const uint16_t f16[4] = {0x3C00, 0x4000, 0x4400, 0x4800};    // 1, 2, 4, 8
        for (size_t i = 0; i < 4; i++) {
            check(elementAsFp16(f16, SourceDtype::FP16, i) == f16[i],
                  "F16 indexing passes the element through unchanged");
        }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
