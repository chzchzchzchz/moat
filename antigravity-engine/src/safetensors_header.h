#pragma once
//
// Parsing the safetensors header, in a Metal-free header so it can be tested.
//
// This decides where every weight in the model comes from, and it was a hand-rolled
// scan over the JSON with three defects, none of which would announce itself:
//
//   1. TensorInfo's offset_start and offset_end had no initialiser, and the struct was
//      default-constructed. A tensor whose "data_offsets" failed to parse therefore
//      carried indeterminate offsets, and loadTensor read from raw_data + garbage.
//
//   2. std::stoll and std::stoull throw on malformed input. parseSafetensors is reached
//      through AntigravityEngineLoadModel, which is extern "C", so a bad header threw an
//      exception across a C boundary instead of returning an error code.
//
//   3. Nothing checked that a tensor's offsets lay inside the file. A truncated download
//      — the most likely way a multi-gigabyte checkpoint goes wrong — would be mmapped
//      and read past its end, giving whatever followed it in the address space as
//      weights. Garbage weights produce confident, constant output, which is what
//      gsm8k_full_checkpoint.json in this repository looks like.
//
// None of the three is proven to have caused that run. All three are the kind of thing
// that would, and none of them was detectable from the outside.
//
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace antigravity {

struct TensorEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset_start = 0;   // explicit: these were indeterminate before
    uint64_t offset_end = 0;

    uint64_t byteLength() const {
        return offset_end >= offset_start ? offset_end - offset_start : 0;
    }
    int64_t elementCount() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return shape.empty() ? 0 : n;
    }
};

struct HeaderParseResult {
    bool ok = false;
    std::string error;                            // empty when ok
    std::map<std::string, TensorEntry> tensors;
};

// Parse a decimal integer without throwing. Returns false on anything unexpected,
// including trailing rubbish, so a malformed header is an error rather than a guess.
inline bool parseUInt64(const std::string& text, uint64_t& out) {
    if (text.empty()) return false;
    uint64_t value = 0;
    for (char c : text) {
        if (c < '0' || c > '9') return false;
        const uint64_t digit = (uint64_t)(c - '0');
        if (value > (UINT64_MAX - digit) / 10) return false;   // overflow
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

inline bool parseInt64(const std::string& text, int64_t& out) {
    if (text.empty()) return false;
    const bool negative = text[0] == '-';
    uint64_t magnitude = 0;
    if (!parseUInt64(negative ? text.substr(1) : text, magnitude)) return false;
    if (magnitude > (uint64_t)INT64_MAX) return false;
    out = negative ? -(int64_t)magnitude : (int64_t)magnitude;
    return true;
}

inline std::string stripSpaces(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out.push_back(c);
    }
    return out;
}

// Find the object that starts at `open` and return one past its closing brace, honouring
// nesting and string literals. The previous code skipped __metadata__ by finding the
// first '}' after the key, which lands inside any nested object.
inline size_t matchBrace(const std::string& s, size_t open) {
    if (open >= s.size() || s[open] != '{') return std::string::npos;
    int depth = 0;
    bool in_string = false;
    for (size_t i = open; i < s.size(); i++) {
        const char c = s[i];
        if (in_string) {
            if (c == '\\') { i++; continue; }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') { in_string = true; continue; }
        if (c == '{') depth++;
        else if (c == '}') {
            depth--;
            if (depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}

// Extract a quoted string value for `key` from one tensor's object.
inline bool findStringValue(const std::string& object, const std::string& key,
                           std::string& out) {
    const std::string quoted = "\"" + key + "\"";
    const size_t at = object.find(quoted);
    if (at == std::string::npos) return false;
    const size_t colon = object.find(':', at + quoted.size());
    if (colon == std::string::npos) return false;
    const size_t open = object.find('"', colon);
    if (open == std::string::npos) return false;
    const size_t close = object.find('"', open + 1);
    if (close == std::string::npos) return false;
    out = object.substr(open + 1, close - open - 1);
    return true;
}

// Extract a bracketed list of integers for `key`.
inline bool findIntArray(const std::string& object, const std::string& key,
                         std::vector<int64_t>& out) {
    const std::string quoted = "\"" + key + "\"";
    const size_t at = object.find(quoted);
    if (at == std::string::npos) return false;
    const size_t open = object.find('[', at + quoted.size());
    if (open == std::string::npos) return false;
    const size_t close = object.find(']', open);
    if (close == std::string::npos) return false;

    out.clear();
    const std::string body = object.substr(open + 1, close - open - 1);
    size_t start = 0;
    while (start <= body.size()) {
        const size_t comma = body.find(',', start);
        const std::string token =
            stripSpaces(body.substr(start, comma == std::string::npos
                                           ? std::string::npos : comma - start));
        if (!token.empty()) {
            int64_t value = 0;
            if (!parseInt64(token, value)) return false;
            out.push_back(value);
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

// Parse the header and validate every tensor against the file.
//
// `data_start` is 8 + header_len, and `file_size` the whole file. Offsets are relative to
// data_start, so a tensor is only readable when data_start + offset_end <= file_size.
// Checking that here is what turns a truncated checkpoint from silently wrong weights
// into a refusal naming the tensor.
inline HeaderParseResult parseSafetensorsHeader(const std::string& header_json,
                                                uint64_t data_start,
                                                uint64_t file_size) {
    HeaderParseResult result;
    size_t pos = 0;

    while (pos < header_json.size()) {
        const size_t key_start = header_json.find('"', pos);
        if (key_start == std::string::npos) break;
        const size_t key_end = header_json.find('"', key_start + 1);
        if (key_end == std::string::npos) break;
        const std::string key = header_json.substr(key_start + 1, key_end - key_start - 1);

        const size_t open = header_json.find('{', key_end);
        if (open == std::string::npos) break;
        const size_t after = matchBrace(header_json, open);
        if (after == std::string::npos) {
            result.error = "unterminated object for key '" + key + "'";
            return result;
        }

        if (key == "__metadata__") {
            pos = after;                 // brace-matched, so nesting is handled
            continue;
        }

        const std::string object = header_json.substr(open, after - open);
        TensorEntry entry;

        if (!findStringValue(object, "dtype", entry.dtype)) {
            result.error = "tensor '" + key + "' has no dtype";
            return result;
        }
        if (!findIntArray(object, "shape", entry.shape)) {
            result.error = "tensor '" + key + "' has no parseable shape";
            return result;
        }
        std::vector<int64_t> offsets;
        if (!findIntArray(object, "data_offsets", offsets) || offsets.size() != 2) {
            result.error = "tensor '" + key + "' has no parseable data_offsets";
            return result;
        }
        if (offsets[0] < 0 || offsets[1] < offsets[0]) {
            result.error = "tensor '" + key + "' has a reversed or negative data_offsets";
            return result;
        }
        entry.offset_start = (uint64_t)offsets[0];
        entry.offset_end = (uint64_t)offsets[1];

        // The check that was missing entirely.
        if (data_start > file_size || entry.offset_end > file_size - data_start) {
            result.error = "tensor '" + key + "' ends at byte "
                         + std::to_string(data_start + entry.offset_end)
                         + " but the file is only " + std::to_string(file_size)
                         + " bytes; the checkpoint is truncated or its header is wrong";
            return result;
        }

        result.tensors[key] = entry;
        pos = after;
    }

    if (result.tensors.empty()) {
        result.error = "no tensors found in the header";
        return result;
    }
    result.ok = true;
    return result;
}

}  // namespace antigravity
