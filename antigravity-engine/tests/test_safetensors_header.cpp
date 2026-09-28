// Tests for the safetensors header parser.
//
// The parser decides where every weight comes from. Each test below corresponds to a way
// it could point at the wrong bytes, or at bytes outside the file, without saying so.
#include "../src/safetensors_header.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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
            check(r.tensors.at("a.scale").elementCount() == 1,
                  "an empty shape is a scalar: one element, as numpy and loadTensor agree");
            check(r.tensors.at("a.scale").byteLength() == 4,
                  "the scalar's four bytes are its F32 element");
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

    // ---- dtype element sizes ------------------------------------------------------
    {
        check(elementSizeForDtype("F32") == 4, "F32 is four bytes");
        check(elementSizeForDtype("F16") == 2, "F16 is two bytes");
        check(elementSizeForDtype("BF16") == 2, "BF16 is two bytes");
        check(elementSizeForDtype("F64") == 8, "F64 is eight bytes");
        check(elementSizeForDtype("I64") == 8, "I64 is eight bytes");
        check(elementSizeForDtype("I8") == 1, "I8 is one byte");
        check(elementSizeForDtype("BOOL") == 1, "BOOL is one byte");
        check(elementSizeForDtype("F8_E4M3") == 1, "F8_E4M3 is one byte");
        check(elementSizeForDtype("") == 0, "an empty dtype is unknown");
        check(elementSizeForDtype("COMPLEX128") == 0, "an unrecognised dtype is unknown");
        // The lowercase spellings loadTensor's dtype dispatch accepts are sized here too,
        // so a header using one is still cross-checked rather than skipped.
        check(elementSizeForDtype("f32") == 4, "the lowercase f32 spelling is sized");
        check(elementSizeForDtype("bf16") == 2, "the lowercase bf16 spelling is sized");
        check(elementSizeForDtype("bfloat16") == 2, "bfloat16 is sized");
        check(elementSizeForDtype("float16") == 2, "float16 is sized");
    }

    // ---- shape cross-checked against the byte span --------------------------------
    // Nothing compared these. loadTensor's BF16 conversion loop and its transposing loop
    // both iterate shape[0]*shape[1] elements with no reference to data_offsets, so a
    // shape claiming more elements than the span holds read past the tensor.
    {
        // 4 F32 elements need 16 bytes, not 8.
        const std::string h = oneTensor("a.weight", "F32", "[4]", 0, 8);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 8);
        check(!r.ok, "a shape implying more bytes than the span holds is refused");
        check(r.error.find("a.weight") != std::string::npos,
              "the shape/span refusal names the tensor");
        check(r.error.find("16") != std::string::npos
              && r.error.find("8") != std::string::npos,
              "the refusal gives both the implied and the actual byte count");
    }
    {
        const std::string h = oneTensor("a.weight", "F32", "[4]", 0, 16);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 16);
        check(r.ok, "an F32 shape matching its span is accepted");
    }
    {
        // The span larger than the shape needs is also a mismatch: loadTensor would
        // silently load a prefix.
        const std::string h = oneTensor("a.weight", "F16", "[4]", 0, 32);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 32);
        check(!r.ok, "a span larger than the shape requires is refused too");
    }
    {
        const std::string h = oneTensor("a.weight", "BF16", "[32000,2048]", 0, 131072000);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 131072000);
        check(r.ok, "a real-sized BF16 embedding matrix checks out");
    }
    {
        // An unknown dtype is not cross-checked and not refused: a checkpoint may carry
        // tensors this engine never loads.
        const std::string h = oneTensor("a.weight", "COMPLEX128", "[4]", 0, 7);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size() + 7);
        check(r.ok, "an unknown dtype is parsed rather than refused");
        check(r.tensors.at("a.weight").dtype == "COMPLEX128",
              "the unknown dtype is reported as-is so the caller can refuse it");
    }
    {
        // A zero dimension: 0 elements, 0 bytes. Legal and consistent.
        const std::string h = oneTensor("a.weight", "F32", "[0,2048]", 0, 0);
        auto r = parseSafetensorsHeader(h, 8 + h.size(), 8 + h.size());
        check(r.ok, "a tensor with a zero dimension and no bytes is consistent");
        if (r.ok) check(r.tensors.at("a.weight").elementCount() == 0,
                        "a zero dimension gives zero elements");
    }

    // ---- a REAL header, cross-checked against Python's json ------------------------
    // Every header above is one I wrote, and a parser tested only against its author's
    // synthetic input is not tested against reality. This is the actual 32,280-byte
    // header of Qwen/Qwen2.5-0.5B-Instruct's model.safetensors: 290 tensors plus
    // __metadata__, names 40 characters long, offsets tiling the file exactly. The
    // expected values are what Python's json module reports for the same bytes, so this
    // requires the C++ parser to agree with a real JSON parser, not with me.
    // See tests/fixtures/README_header_fixture.md.
    {
        const uint64_t kDataStart = 8 + 32280;     // the file's real header_len
        const uint64_t kFileSize  = 988097824;     // the file's real size

        std::ifstream hf("tests/fixtures/qwen2.5-0.5b-instruct.header.json",
                         std::ios::binary);
        check(hf.is_open(), "the real header fixture can be opened");
        if (hf.is_open()) {
            std::stringstream hb;
            hb << hf.rdbuf();
            const std::string header = hb.str();
            check(header.size() == 32280, "the fixture is the expected 32280 bytes");

            auto r = parseSafetensorsHeader(header, kDataStart, kFileSize);
            check(r.ok, "the real header parses");
            if (!r.ok) std::printf("    (error was: %s)\n", r.error.c_str());

            if (r.ok) {
                check(r.tensors.size() == 290,
                      "290 tensors are found (291 keys less __metadata__)");

                // Compare every field against Python's json, line by line.
                std::ifstream ef("tests/fixtures/qwen2.5-0.5b-instruct.expected.tsv");
                check(ef.is_open(), "the expected-values fixture can be opened");
                size_t compared = 0;
                bool all_match = true;
                std::string line;
                while (std::getline(ef, line)) {
                    if (line.empty()) continue;
                    // name \t dtype \t shape \t start \t end
                    std::vector<std::string> field;
                    size_t at = 0;
                    while (true) {
                        const size_t tab = line.find('\t', at);
                        field.push_back(line.substr(at, tab == std::string::npos
                                                        ? std::string::npos : tab - at));
                        if (tab == std::string::npos) break;
                        at = tab + 1;
                    }
                    if (field.size() != 5) { all_match = false; break; }

                    auto it = r.tensors.find(field[0]);
                    if (it == r.tensors.end()) { all_match = false; break; }
                    const TensorEntry& t = it->second;

                    if (t.dtype != field[1]) all_match = false;

                    std::string got_shape;
                    for (size_t i = 0; i < t.shape.size(); i++) {
                        if (i) got_shape += ",";
                        got_shape += std::to_string(t.shape[i]);
                    }
                    if (got_shape != field[2]) all_match = false;

                    uint64_t want_start = 0, want_end = 0;
                    if (!parseUInt64(field[3], want_start)
                        || !parseUInt64(field[4], want_end)) all_match = false;
                    if (t.offset_start != want_start || t.offset_end != want_end) {
                        all_match = false;
                    }
                    compared++;
                }
                check(compared == 290, "all 290 expected lines were read");
                check(all_match,
                      "every tensor's dtype, shape and both offsets match Python's json");

                // The offsets tile the file with no gaps and no overlap, and the last one
                // ends exactly at the end of the file — so the inclusive upper boundary of
                // the bounds check is exercised against a real checkpoint's geometry.
                uint64_t largest_end = 0;
                for (const auto& kv : r.tensors) {
                    if (kv.second.offset_end > largest_end) largest_end = kv.second.offset_end;
                }
                check(kDataStart + largest_end == kFileSize,
                      "the last tensor ends exactly at the end of the real file");

                // One byte short must be refused, on the real header.
                auto short_r = parseSafetensorsHeader(header, kDataStart, kFileSize - 1);
                check(!short_r.ok,
                      "the real header is refused when the file is one byte short");
            }
        }
    }

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
