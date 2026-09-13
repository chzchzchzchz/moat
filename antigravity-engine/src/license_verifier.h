#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct AntigravityLicensePayload {
    bool valid = false;
    std::string licensee;
    int64_t expiry_epoch = 0;
    int32_t max_channels = 8;
    std::vector<std::string> features;
    std::string error_message;

    bool hasFeature(const std::string& feature) const;
    bool isExpired() const;
};

class AntigravityLicenseVerifier {
public:
    static const uint8_t DEFAULT_PUBLIC_KEY[32];

    explicit AntigravityLicenseVerifier(const uint8_t* public_key_32 = DEFAULT_PUBLIC_KEY);

    AntigravityLicensePayload verify(const std::string& license_token) const;

    static std::string decodeBase64URL(const std::string& input);
    static std::vector<uint8_t> decodeBase64URLBytes(const std::string& input);

private:
    uint8_t pubkey_[32];
};
