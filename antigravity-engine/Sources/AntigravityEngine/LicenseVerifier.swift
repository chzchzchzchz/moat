//
// Project Antigravity — Public Swift API Wrapper & Engine SDK
// Apple Silicon Edge Engine (iOS / macOS, M1-M4, A17 Pro, A18 Pro)
//

import Foundation
// CryptoKit predates strict concurrency checking and does not mark
// Curve25519.Signing.PublicKey as Sendable, which warns on this type's stored
// property. The key is immutable and only read, so suppress at the import as the
// compiler's own remark suggests rather than weakening LicenseVerifier's Sendable
// conformance.
@preconcurrency import CryptoKit
#if canImport(UIKit)
import UIKit
#else
import IOKit
#endif

/// License payload containing customer and feature information
public struct LicensePayload: Codable, Sendable {
    public let licensee: String
    public let expiryEpoch: Int64  // Unix timestamp, 0 = perpetual
    public let features: [String]  // e.g. ["clinical", "sdk", "multimodal"]
    public let maxChannels: Int    // Maximum parallel channels allowed
    public let deviceHash: String? // Optional hardware binding
}

public enum LicenseStatus: Sendable {
    case valid(LicensePayload)
    case expired(LicensePayload)
    case invalidSignature
    case invalidFormat
    case featureNotLicensed(String)
}

public enum LicenseError: Error, LocalizedError {
    case invalidFormat
    case signatureVerificationFailed
    case expired(expiryDate: Date)
    case decodingError
    case featureNotLicensed(feature: String)
    case noLicenseProvided
    
    public var errorDescription: String? {
        switch self {
        case .invalidFormat:
            return "The license key has an invalid format."
        case .signatureVerificationFailed:
            return "The license signature verification failed."
        case .expired(let expiryDate):
            return "The license expired on \(expiryDate)."
        case .decodingError:
            return "Failed to decode the license payload."
        case .featureNotLicensed(let feature):
            return "The feature '\(feature)' is not licensed."
        case .noLicenseProvided:
            return "No license key was provided."
        }
    }
}

/// Offline Ed25519 license verifier using Apple CryptoKit.
/// Embeds a 32-byte public key and validates license signatures
/// without any network access.
public final class LicenseVerifier: Sendable {
    private let publicKey: Curve25519.Signing.PublicKey

    /// Initialize with raw 32-byte Ed25519 public key
    public init(publicKeyBytes: [UInt8]) throws {
        self.publicKey = try Curve25519.Signing.PublicKey(rawRepresentation: publicKeyBytes)
    }

    /// Initialize with base64-encoded public key string
    public init(publicKeyBase64: String) throws {
        guard let data = Data(base64Encoded: publicKeyBase64) else {
            throw LicenseError.invalidFormat
        }
        self.publicKey = try Curve25519.Signing.PublicKey(rawRepresentation: data)
    }

    private static func decodeBase64URL(_ base64URL: String) -> Data? {
        var base64 = base64URL
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        
        let paddingCount = base64.count % 4
        if paddingCount > 0 {
            base64 += String(repeating: "=", count: 4 - paddingCount)
        }
        
        return Data(base64Encoded: base64)
    }

    /// Verify a license key string and return the payload if valid
    /// License format: BASE64(payload_json).BASE64(64_byte_signature)
    public func verify(licenseKey: String) throws -> LicensePayload {
        let parts = licenseKey.components(separatedBy: ".")
        guard parts.count == 2 else {
            throw LicenseError.invalidFormat
        }
        
        guard let payloadData = Self.decodeBase64URL(parts[0]),
              let signatureData = Self.decodeBase64URL(parts[1]) else {
            throw LicenseError.invalidFormat
        }
        
        guard publicKey.isValidSignature(signatureData, for: payloadData) else {
            throw LicenseError.signatureVerificationFailed
        }
        
        let payload: LicensePayload
        do {
            payload = try JSONDecoder().decode(LicensePayload.self, from: payloadData)
        } catch {
            throw LicenseError.decodingError
        }
        
        if payload.expiryEpoch != 0 {
            let expiryDate = Date(timeIntervalSince1970: TimeInterval(payload.expiryEpoch))
            if Date() > expiryDate {
                throw LicenseError.expired(expiryDate: expiryDate)
            }
        }
        
        return payload
    }

    /// Check license status without throwing
    public func status(licenseKey: String) -> LicenseStatus {
        let parts = licenseKey.components(separatedBy: ".")
        guard parts.count == 2 else {
            return .invalidFormat
        }
        
        guard let payloadData = Self.decodeBase64URL(parts[0]),
              let signatureData = Self.decodeBase64URL(parts[1]) else {
            return .invalidFormat
        }
        
        guard publicKey.isValidSignature(signatureData, for: payloadData) else {
            return .invalidSignature
        }
        
        let payload: LicensePayload
        do {
            payload = try JSONDecoder().decode(LicensePayload.self, from: payloadData)
        } catch {
            return .invalidFormat
        }
        
        if payload.expiryEpoch != 0 {
            let expiryDate = Date(timeIntervalSince1970: TimeInterval(payload.expiryEpoch))
            if Date() > expiryDate {
                return .expired(payload)
            }
        }
        
        return .valid(payload)
    }

    /// Verify that a specific feature is licensed
    public func verifyFeature(_ feature: String, licenseKey: String) throws -> LicensePayload {
        let payload = try verify(licenseKey: licenseKey)
        guard payload.features.contains(feature) else {
            throw LicenseError.featureNotLicensed(feature: feature)
        }
        return payload
    }

    /// Get device-specific hash for hardware binding
    /// Uses identifierForVendor on iOS, serial number hash on macOS
    public static var currentDeviceHash: String {
        let identifier: String
        #if canImport(UIKit)
        identifier = UIDevice.current.identifierForVendor?.uuidString ?? "unknown-ios-device"
        #else
        identifier = getMacSerialNumber()
        #endif
        
        let hash = SHA256.hash(data: Data(identifier.utf8))
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
    
    #if !canImport(UIKit)
    private static func getMacSerialNumber() -> String {
        let platformExpert = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOPlatformExpertDevice"))
        guard platformExpert > 0 else { return "unknown-mac-device" }
        
        guard let serialNumber = IORegistryEntryCreateCFProperty(platformExpert, kIOPlatformSerialNumberKey as CFString, kCFAllocatorDefault, 0).takeUnretainedValue() as? String else {
            IOObjectRelease(platformExpert)
            return "unknown-mac-device"
        }
        
        IOObjectRelease(platformExpert)
        return serialNumber
    }
    #endif
}
