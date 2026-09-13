//
// Project Antigravity — Public Swift API Wrapper & Engine SDK
// Apple Silicon Edge Engine (iOS / macOS, M1-M4, A17 Pro, A18 Pro)
//

import Foundation
import Security

/// Hardware-backed AES-256 encrypted storage using iOS/macOS Keychain
/// with Data Protection Class A (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`).
/// Data is encrypted at rest by Keychain Data Protection and cannot be extracted
/// even with physical device access when locked.
public final class SecureStorage: Sendable {
    public enum KeychainError: Error, LocalizedError {
        case unhandledStatus(OSStatus)
        case itemNotFound
        case duplicateItem
        case authenticationFailed
        
        public var errorDescription: String? {
            switch self {
            case .unhandledStatus(let status):
                return "Unhandled Keychain status: \(status)"
            case .itemNotFound:
                return "Item not found in Keychain"
            case .duplicateItem:
                return "Item already exists in Keychain"
            case .authenticationFailed:
                return "Biometric authentication failed"
            }
        }
    }

    private let serviceName: String

    public init(serviceName: String = "org.antigravity.secure-storage") {
        self.serviceName = serviceName
    }

    // Save data with Class A hardware-backed encryption
    public func save(account: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status == errSecDuplicateItem {
            throw KeychainError.duplicateItem
        } else if status != errSecSuccess {
            throw KeychainError.unhandledStatus(status)
        }
    }

    // Load data (fails if device is locked)
    public func load(account: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecItemNotFound {
            throw KeychainError.itemNotFound
        } else if status != errSecSuccess {
            throw KeychainError.unhandledStatus(status)
        }
        
        guard let data = dataTypeRef as? Data else {
            throw KeychainError.itemNotFound
        }
        
        return data
    }

    // Update existing secret
    public func update(account: String, data: Data) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        
        let attributes: [String: Any] = [
            kSecValueData as String: data
        ]
        
        let status = SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
        
        if status == errSecItemNotFound {
            throw KeychainError.itemNotFound
        } else if status != errSecSuccess {
            throw KeychainError.unhandledStatus(status)
        }
    }

    // Delete secret
    public func delete(account: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        
        let status = SecItemDelete(query as CFDictionary)
        
        if status != errSecSuccess && status != errSecItemNotFound {
            throw KeychainError.unhandledStatus(status)
        }
    }

    // Check if an item exists
    public func exists(account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: false,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        let status = SecItemCopyMatching(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    // Save with biometric protection (FaceID/TouchID required to read)
    public func saveWithBiometric(account: String, data: Data, reason: String) throws {
        var error: Unmanaged<CFError>?
        guard let accessControl = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .userPresence,
            &error
        ) else {
            throw KeychainError.unhandledStatus(errSecParam)
        }
        
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessControl as String: accessControl
        ]
        
        let status = SecItemAdd(query as CFDictionary, nil)
        
        if status == errSecDuplicateItem {
            throw KeychainError.duplicateItem
        } else if status != errSecSuccess {
            throw KeychainError.unhandledStatus(status)
        }
    }

    // Load biometric-protected data
    public func loadWithBiometric(account: String) throws -> Data {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        
        var dataTypeRef: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &dataTypeRef)
        
        if status == errSecItemNotFound {
            throw KeychainError.itemNotFound
        } else if status == errSecAuthFailed {
            throw KeychainError.authenticationFailed
        } else if status != errSecSuccess {
            throw KeychainError.unhandledStatus(status)
        }
        
        guard let data = dataTypeRef as? Data else {
            throw KeychainError.itemNotFound
        }
        
        return data
    }

    // Generate and store a random 256-bit encryption key for SQLite database encryption
    public func getOrCreateDatabaseKey() throws -> Data {
        let account = "sqlite-encryption-key"
        
        if exists(account: account) {
            return try load(account: account)
        }
        
        var keyData = Data(count: 32)
        let result = keyData.withUnsafeMutableBytes {
            SecRandomCopyBytes(kSecRandomDefault, 32, $0.baseAddress!)
        }
        
        guard result == errSecSuccess else {
            throw KeychainError.unhandledStatus(result)
        }
        
        try save(account: account, data: keyData)
        return keyData
    }
}
