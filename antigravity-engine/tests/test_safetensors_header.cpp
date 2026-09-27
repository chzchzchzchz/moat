// Tests for the safetensors header parser.
//
// The parser decides where every weight comes from. Each test below corresponds to a way
// it could point at the wrong bytes, or at bytes outside the file, without saying so.
#include "../src/safetensors_header.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace antigravity;

static int failures = 0;
static int checks = 0;

static void check(bool condition, const char* what) {
    checks++;
    if (!condition) {
        failures++;
        std::printf("  FAIL: %s\n", what);
    }
}

// Build a header for one FP16 tensor of `count` elements starting at byte 0.
static std::string oneTensor(const char* name, const char* dtype,
                             const char* shape, uint64_t start, uint64_t end) {
    return std::string("{\"") + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":"
         + shape + ",\"data_offsets\":[" + std::to_string(start) + ","
         + std::to_string(end) + "]}}";
}

int main() {
    // ---- the ordinary case -------------------------------------------------------
    {
        const std::string h = oneTensor("model.embed_tokens.weight", "BF16",
                                        "[32000,2048]", 0, 131072000);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 131072000);
        check(r.ok, "a well-formed single-tensor header parses");
        check(r.error.empty(), "a well-formed header reports no error");
        check(r.tensors.size() == 1, "one tensor is found");
        const auto& t = r.tensors.at("model.embed_tokens.weight");
        check(t.dtype == "BF16", "dtype is read");
        check(t.shape.size() == 2 && t.shape[0] == 32000 && t.shape[1] == 2048,
              "shape is read");
        check(t.offset_start == 0 && t.offset_end == 131072000, "offsets are read");
        check(t.byteLength() == 131072000, "byteLength is the offset span");
        check(t.elementCount() == 32000LL * 2048LL, "elementCount multiplies the shape");
    }

    // ---- several tensors, and the keys inside a value object are not mistaken for keys
    {
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]},"
            "\"b.weight\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[8,24]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 24);
        check(r.ok, "a two-tensor header parses");
        check(r.tensors.size() == 2, "exactly two tensors are found, not six");
        check(r.tensors.count("a.weight") == 1 && r.tensors.count("b.weight") == 1,
              "both tensor names are found");
        check(r.tensors.count("dtype") == 0,
              "'dtype' inside a value object is not taken for a tensor name");
        check(r.tensors.count("shape") == 0,
              "'shape' inside a value object is not taken for a tensor name");
    }

    // ---- __metadata__, flat: the case the old first-'}' skip happened to survive ----
    {
        const std::string h =
            "{\"__metadata__\":{\"format\":\"pt\"},"
            "\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "a header with flat __metadata__ parses");
        check(r.tensors.size() == 1, "__metadata__ is not counted as a tensor");
        check(r.tensors.count("a.weight") == 1, "the real tensor is still found");
    }

    // ---- __metadata__ with a NESTED object: the bug -------------------------------
    // The old code skipped __metadata__ with header_json.find('}', key_end), which stops
    // at the first '}'. Whether that is the end of metadata depends on where the nested
    // object sits, so the two orderings below behave differently under the old code. Both
    // are tested, because a test that only covers the lucky ordering guards nothing.
    //
    // Measured against a reproduction of the old skip:
    //   nested object LAST  -> found a.weight          (survives by coincidence)
    //   nested object FIRST -> found "format" ONLY, carrying a.weight's data_offsets
    //
    // In the second case the real tensor is absent from the map entirely, so loadTensor
    // never runs for it and its weight buffer keeps whatever it already held. A forward
    // pass over uninitialised weights is exactly the "output does not vary with input"
    // failure this repository already has a recorded example of.
    {
        // The ordering the old code got right. Kept so a future rewrite cannot regress it.
        const std::string h =
            "{\"__metadata__\":{\"format\":\"pt\",\"extra\":{\"tool\":\"convert\"}},"
            "\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "nested __metadata__ (nested object last) parses");
        check(r.tensors.size() == 1, "only the real tensor is found");
        check(r.tensors.count("a.weight") == 1, "the real tensor is found");
    }
    {
        // The ordering the old code got wrong.
        const std::string h =
            "{\"__metadata__\":{\"extra\":{\"tool\":\"convert\"},\"format\":\"pt\"},"
            "\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "nested __metadata__ (nested object first) parses");
        check(r.tensors.size() == 1,
              "nested __metadata__ is skipped whole, not up to its first brace");
        check(r.tensors.count("a.weight") == 1,
              "the real tensor is found, where the old code lost it");
        check(r.tensors.count("format") == 0,
              "a metadata key is not returned as a tensor holding the real one's offsets");
        check(r.tensors.count("extra") == 0,
              "a nested metadata key is not taken for a tensor");
    }

    // ---- a brace inside a string literal must not end the object ------------------
    // Measured against the old skip, this produced a tensor named "," .
    {
        const std::string h =
            "{\"__metadata__\":{\"note\":\"} oops\",\"format\":\"pt\"},"
            "\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "a brace inside a metadata string does not end the object");
        check(r.tensors.size() == 1, "only the real tensor is found");
        check(r.tensors.count("a.weight") == 1, "the real tensor is found");
        check(r.tensors.count(",") == 0,
              "a stray comma is not returned as a tensor, as the old code did");
    }
    {
        // An escaped quote inside a metadata string.
        const std::string h =
            "{\"__metadata__\":{\"note\":\"a \\\" quote\",\"format\":\"pt\"},"
            "\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "an escaped quote inside a metadata string is handled");
        check(r.tensors.count("a.weight") == 1,
              "the real tensor is found past an escaped quote");
    }

    // ---- THE MISSING CHECK: a tensor past the end of the file ---------------------
    // This is what a truncated download looks like. Before, it was mmapped and read.
    {
        const std::string h = oneTensor("a.weight", "F16", "[4]", 0, 8);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 4);  // 4 short
        check(!r.ok, "a tensor ending past the end of the file is refused");
        check(r.error.find("a.weight") != std::string::npos,
              "the refusal names the offending tensor");
        check(r.error.find("truncated") != std::string::npos,
              "the refusal says the checkpoint is truncated");
        check(r.tensors.empty(), "nothing is returned when the file is too short");
    }
    {
        // Exactly the right size is fine — the boundary is inclusive.
        const std::string h = oneTensor("a.weight", "F16", "[4]", 0, 8);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(r.ok, "a tensor ending exactly at the end of the file is accepted");
    }
    {
        // data_start alone past the file (an impossible header_len).
        const std::string h = oneTensor("a.weight", "F16", "[4]", 0, 8);
        auto r = parseSafetensorsHeader(h, 1000000, 100);
        check(!r.ok, "a data_start past the end of the file is refused");
    }

    // ---- malformed input returns an error instead of throwing ---------------------
    {
        auto r = parseSafetensorsHeader("", 8, 8);
        check(!r.ok, "an empty header is refused");
        check(r.error.find("no tensors") != std::string::npos,
              "an empty header says no tensors were found");
    }
    {
        // Non-numeric offsets. std::stoull threw here before.
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],"
            "\"data_offsets\":[abc,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 1000);
        check(!r.ok, "non-numeric data_offsets are refused, not thrown on");
        check(r.error.find("data_offsets") != std::string::npos,
              "the refusal names data_offsets");
    }
    {
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[x],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 1000);
        check(!r.ok, "a non-numeric shape is refused");
        check(r.error.find("shape") != std::string::npos, "the refusal names shape");
    }
    {
        const std::string h = "{\"a.weight\":{\"shape\":[4],\"data_offsets\":[0,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 1000);
        check(!r.ok, "a tensor with no dtype is refused");
        check(r.error.find("dtype") != std::string::npos, "the refusal names dtype");
    }
    {
        const std::string h = "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 1000);
        check(!r.ok, "a tensor with no data_offsets is refused");
    }
    {
        // One offset only, where the old code would have left offset_end indeterminate.
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 1000);
        check(!r.ok, "a one-element data_offsets is refused");
    }
    {
        const std::string h = oneTensor("a.weight", "F16", "[4]", 24, 8);  // reversed
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 100000);
        check(!r.ok, "reversed data_offsets are refused");
        check(r.error.find("reversed") != std::string::npos,
              "the refusal says the offsets are reversed");
    }
    {
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[-8,8]}}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 100000);
        check(!r.ok, "a negative data_offsets start is refused");
    }
    {
        const std::string h =
            "{\"a.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 100000);
        check(!r.ok, "an unterminated object is refused");
    }

    // ---- integer parsing helpers --------------------------------------------------
    {
        uint64_t u = 99;
        check(parseUInt64("0", u) && u == 0, "parseUInt64 reads zero");
        check(parseUInt64("18446744073709551615", u) && u == UINT64_MAX,
              "parseUInt64 reads the largest uint64");
        check(!parseUInt64("18446744073709551616", u),
              "parseUInt64 refuses an overflowing value instead of wrapping");
        check(!parseUInt64("", u), "parseUInt64 refuses an empty string");
        check(!parseUInt64("12a", u), "parseUInt64 refuses trailing rubbish");
        check(!parseUInt64("-1", u), "parseUInt64 refuses a negative");
        check(!parseUInt64(" 1", u), "parseUInt64 refuses a leading space");

        int64_t i = 99;
        check(parseInt64("-42", i) && i == -42, "parseInt64 reads a negative");
        check(parseInt64("9223372036854775807", i) && i == INT64_MAX,
              "parseInt64 reads the largest int64");
        check(!parseInt64("9223372036854775808", i),
              "parseInt64 refuses a value past INT64_MAX");
        check(!parseInt64("-", i), "parseInt64 refuses a bare minus sign");
    }

    // ---- whitespace, as some writers emit it --------------------------------------
    {
        const std::string h =
            "{ \"a.weight\" : { \"dtype\" : \"F16\" , \"shape\" : [ 4 , 2 ] , "
            "\"data_offsets\" : [ 0 , 16 ] } }";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 16);
        check(r.ok, "a header with whitespace parses");
        if (r.ok) {
            const auto& t = r.tensors.at("a.weight");
            check(t.shape.size() == 2 && t.shape[0] == 4 && t.shape[1] == 2,
                  "a spaced shape is read");
            check(t.offset_end == 16, "spaced offsets are read");
        }
    }

    // ---- a scalar tensor: empty shape ---------------------------------------------
    {
        const std::string h = oneTensor("a.scale", "F32", "[]", 0, 4);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 4);
        check(r.ok, "a zero-dimensional tensor parses");
        if (r.ok) {
            check(r.tensors.at("a.scale").elementCount() == 0,
                  "an empty shape reports zero elements rather than one");
        }
    }

    // ---- a realistic many-tensor header -------------------------------------------
    {
        std::string h = "{\"__metadata__\":{\"format\":\"pt\"}";
        uint64_t at = 0;
        for (int layer = 0; layer < 22; layer++) {
            for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                h += ",\"model.layers." + std::to_string(layer) + ".self_attn."
                   + proj + ".weight\":{\"dtype\":\"BF16\",\"shape\":[2048,2048],"
                   + "\"data_offsets\":[" + std::to_string(at) + ","
                   + std::to_string(at + 2048 * 2048 * 2) + "]}";
                at += 2048 * 2048 * 2;
            }
        }
        h += "}";
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + at);
        check(r.ok, "a 88-tensor header parses");
        check(r.tensors.size() == 88, "all 88 tensors are found");
        if (r.ok) {
            // Offsets must be contiguous and in the order written, not the map order.
            uint64_t expected = 0;
            bool contiguous = true;
            for (int layer = 0; layer < 22; layer++) {
                for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
                    const std::string key = "model.layers." + std::to_string(layer)
                                          + ".self_attn." + proj + ".weight";
                    if (r.tensors.at(key).offset_start != expected) contiguous = false;
                    expected += 2048 * 2048 * 2;
                }
            }
            check(contiguous, "every tensor keeps the offset it was given in the header");
        }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
