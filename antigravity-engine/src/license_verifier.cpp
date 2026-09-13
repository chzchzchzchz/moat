#include "license_verifier.h"
#include "monocypher-ed25519.h"
#include <chrono>
#include <cstring>
#include <algorithm>
#include <sstream>

// Production public key matching tools/license_keygen.py
const uint8_t AntigravityLicenseVerifier::DEFAULT_PUBLIC_KEY[32] = {
    0x63, 0xf0, 0xc1, 0xa4, 0x09, 0xff, 0x13, 0x50, 
    0x2b, 0x14, 0x1a, 0xb6, 0x24, 0x27, 0x34, 0xce, 
    0xd6, 0x99, 0xc3, 0xd5, 0x9b, 0x35, 0x45, 0xb4, 
    0xb7, 0x4d, 0x62, 0x25, 0x29, 0x9a, 0x57, 0xec
};

bool AntigravityLicensePayload::hasFeature(const std::string& feature) const {
    for (const auto& f : features) {
        if (f == feature) return true;
    }
    return false;
}

bool AntigravityLicensePayload::isExpired() const {
    if (expiry_epoch == 0) return false;
    auto now = std::chrono::system_clock::now();
    int64_t now_epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    return now_epoch > expiry_epoch;
}

AntigravityLicenseVerifier::AntigravityLicenseVerifier(const uint8_t* public_key_32) {
    if (public_key_32) {
        std::memcpy(pubkey_, public_key_32, 32);
    } else {
        std::memcpy(pubkey_, DEFAULT_PUBLIC_KEY, 32);
    }
}

static const std::string b64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::vector<uint8_t> AntigravityLicenseVerifier::decodeBase64URLBytes(const std::string& input) {
    std::string b64 = input;
    std::replace(b64.begin(), b64.end(), '-', '+');
    std::replace(b64.begin(), b64.end(), '_', '/');
    while (b64.size() % 4 != 0) {
        b64.push_back('=');
    }

    std::vector<uint8_t> out;
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[b64_table[i]] = i;

    int val = 0, valb = -8;
    for (uint8_t c : b64) {
        if (c == '=') break;
        if (T[c] == -1) continue;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

std::string AntigravityLicenseVerifier::decodeBase64URL(const std::string& input) {
    auto bytes = decodeBase64URLBytes(input);
    return std::string(bytes.begin(), bytes.end());
}

AntigravityLicensePayload AntigravityLicenseVerifier::verify(const std::string& license_token) const {
    AntigravityLicensePayload result;

    size_t dot_pos = license_token.find('.');
    if (dot_pos == std::string::npos) {
        result.error_message = "Invalid license format (missing period delimiter)";
        return result;
    }

    std::string payload_b64 = license_token.substr(0, dot_pos);
    std::string sig_b64 = license_token.substr(dot_pos + 1);

    std::vector<uint8_t> payload_bytes = decodeBase64URLBytes(payload_b64);
    std::vector<uint8_t> sig_bytes = decodeBase64URLBytes(sig_b64);

    if (sig_bytes.size() != 64) {
        result.error_message = "Invalid signature length (expected 64 bytes)";
        return result;
    }

    // Verify Ed25519 signature
    int check = crypto_ed25519_check(sig_bytes.data(), pubkey_, payload_bytes.data(), payload_bytes.size());
    if (check != 0) {
        result.error_message = "Cryptographic signature verification failed";
        return result;
    }

    // Parse JSON payload
    std::string json_str(payload_bytes.begin(), payload_bytes.end());

    // Licensee
    auto find_str = [&](const std::string& key) -> std::string {
        size_t k = json_str.find("\"" + key + "\"");
        if (k == std::string::npos) return "";
        size_t colon = json_str.find(':', k);
        if (colon == std::string::npos) return "";
        size_t q1 = json_str.find('"', colon);
        if (q1 == std::string::npos) return "";
        size_t q2 = json_str.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return json_str.substr(q1 + 1, q2 - q1 - 1);
    };

    auto find_int64 = [&](const std::string& key, int64_t def = 0) -> int64_t {
        size_t k = json_str.find("\"" + key + "\"");
        if (k == std::string::npos) return def;
        size_t colon = json_str.find(':', k);
        if (colon == std::string::npos) return def;
        size_t start = json_str.find_first_of("0123456789-", colon);
        if (start == std::string::npos) return def;
        size_t end = json_str.find_first_not_of("0123456789-", start);
        std::string num_str = json_str.substr(start, end - start);
        try { return std::stoll(num_str); } catch (...) { return def; }
    };

    result.licensee = find_str("licensee");
    result.expiry_epoch = find_int64("expiryEpoch", 0);
    result.max_channels = (int32_t)find_int64("maxChannels", 8);

    // Features array
    size_t f_pos = json_str.find("\"features\"");
    if (f_pos != std::string::npos) {
        size_t b1 = json_str.find('[', f_pos);
        size_t b2 = json_str.find(']', b1);
        if (b1 != std::string::npos && b2 != std::string::npos) {
            std::string arr = json_str.substr(b1 + 1, b2 - b1 - 1);
            size_t idx = 0;
            while (true) {
                size_t q1 = arr.find('"', idx);
                if (q1 == std::string::npos) break;
                size_t q2 = arr.find('"', q1 + 1);
                if (q2 == std::string::npos) break;
                result.features.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
                idx = q2 + 1;
            }
        }
    }

    if (result.isExpired()) {
        result.error_message = "License has expired";
        result.valid = false;
    } else {
        result.valid = true;
    }

    return result;
}
