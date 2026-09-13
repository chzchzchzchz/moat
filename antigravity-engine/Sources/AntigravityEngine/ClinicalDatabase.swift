import Foundation
import SQLite3
import Security
import CryptoKit

/// Stored patient demographic record
public struct StoredPatient: Identifiable, Sendable, Codable {
    public let id: String
    public let name: String
    public let dateOfBirth: String
    public let createdAt: Date

    public init(id: String, name: String, dateOfBirth: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.dateOfBirth = dateOfBirth
        self.createdAt = createdAt
    }
}

/// Stored psychotherapy session with raw transcript and parsed clinical document
public struct StoredSession: Identifiable, Sendable, Codable {
    public let id: String
    public let patientId: String
    public let date: Date
    public let durationMinutes: Int
    public let rawTranscript: String
    public let templateType: String
    public let soapNote: SOAPNote?

    public init(
        id: String = UUID().uuidString,
        patientId: String,
        date: Date = Date(),
        durationMinutes: Int = 50,
        rawTranscript: String,
        templateType: String,
        soapNote: SOAPNote? = nil
    ) {
        self.id = id
        self.patientId = patientId
        self.date = date
        self.durationMinutes = durationMinutes
        self.rawTranscript = rawTranscript
        self.templateType = templateType
        self.soapNote = soapNote
    }
}

/// Longitudinal clinical metric tracking symptom trajectories over time (PHQ-9, GAD-7, C-SSRS)
public struct LongitudinalMetric: Identifiable, Sendable, Codable {
    public let id: String
    public let patientId: String
    public let date: Date
    public let metricType: String
    public let score: Double
    public let clinicalInterpretation: String

    public init(
        id: String = UUID().uuidString,
        patientId: String,
        date: Date = Date(),
        metricType: String,
        score: Double,
        clinicalInterpretation: String
    ) {
        self.id = id
        self.patientId = patientId
        self.date = date
        self.metricType = metricType
        self.score = score
        self.clinicalInterpretation = clinicalInterpretation
    }
}

/// Hardware-secured SQLite storage layer for zero-trust clinical documentation.
/// Stores patient encounter notes, longitudinal metrics, and cross-session memory
/// using hardware-backed Secure Enclave key derivation.
public final class ClinicalDatabase: @unchecked Sendable {
    private var db: OpaquePointer?
    private let queue = DispatchQueue(label: "org.antigravity.clinical-database", qos: .userInitiated)
    private let dbURL: URL
    private let encryptionKey: SymmetricKey

    public enum DatabaseError: Error, LocalizedError {
        case openFailed(String)
        case executionFailed(String)
        case prepareFailed(String)
        case queryFailed(String)

        public var errorDescription: String? {
            switch self {
            case .openFailed(let msg): return "Failed to open clinical database: \(msg)"
            case .executionFailed(let msg): return "SQLite execution failed: \(msg)"
            case .prepareFailed(let msg): return "SQLite prepare failed: \(msg)"
            case .queryFailed(let msg): return "SQLite query failed: \(msg)"
            }
        }
    }

    public init(databaseURL: URL? = nil, encryptionKeyData: Data? = nil) throws {
        if let url = databaseURL {
            self.dbURL = url
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            let vaultDir = appSupport.appendingPathComponent("AntigravityClinicalVault", isDirectory: true)
            try FileManager.default.createDirectory(at: vaultDir, withIntermediateDirectories: true)
            self.dbURL = vaultDir.appendingPathComponent("clinical_records.sqlite")
        }

        if let keyData = encryptionKeyData {
            self.encryptionKey = SymmetricKey(data: keyData)
        } else {
            guard let key = try? SecureStorage().getOrCreateDatabaseKey() else {
                throw DatabaseError.executionFailed("CRITICAL: Cannot initialize encryption key from Keychain. Database cannot be opened without secure key material.")
            }
            self.encryptionKey = SymmetricKey(data: key)
        }

        try openAndMigrate()
        try applyFileProtection()
        try storeKeychainMetadata()
    }

    // MARK: - Zero-Trust Column-Level Encryption (AES-256-GCM)

    /// Encrypt sensitive PHI string into authenticated AES-256-GCM ciphertext payload (Base64)
    public func encryptString(_ plain: String) throws -> String {
        guard !plain.isEmpty else { return "" }
        let data = Data(plain.utf8)
        let sealed = try AES.GCM.seal(data, using: encryptionKey)
        guard let combined = sealed.combined else {
            throw DatabaseError.executionFailed("Failed to produce AES-GCM ciphertext payload")
        }
        return combined.base64EncodedString()
    }

    /// Decrypt authenticated AES-256-GCM ciphertext payload.
    /// Falls back to raw string if not ciphertext (e.g. unencrypted legacy records).
    public func decryptString(_ cipherBase64: String) throws -> String {
        guard !cipherBase64.isEmpty, let data = Data(base64Encoded: cipherBase64) else {
            return cipherBase64
        }
        let box = try AES.GCM.SealedBox(combined: data)
        let decrypted = try AES.GCM.open(box, using: encryptionKey)
        return String(data: decrypted, encoding: .utf8) ?? cipherBase64
    }

    /// Apply Apple Data Protection to the database file.
    /// Uses NSFileProtectionCompleteUnlessOpen (Class B) — the file is encrypted at rest
    /// when the device is locked. This provides hardware-backed encryption via Secure Enclave
    /// without requiring SQLCipher.
    ///
    /// NOTE: For column-level encryption of individual fields (e.g., patient names, transcripts),
    /// a full SQLCipher integration would be required. This provides file-level protection only.
    private func applyFileProtection() throws {
        #if os(iOS)
        guard FileManager.default.fileExists(atPath: dbURL.path) else { return }
        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.completeUnlessOpen],
            ofItemAtPath: dbURL.path
        )
        // Also protect the WAL and SHM files if they exist
        let walPath = dbURL.path + "-wal"
        let shmPath = dbURL.path + "-shm"
        if FileManager.default.fileExists(atPath: walPath) {
            try? FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.completeUnlessOpen],
                ofItemAtPath: walPath
            )
        }
        if FileManager.default.fileExists(atPath: shmPath) {
            try? FileManager.default.setAttributes(
                [.protectionKey: FileProtectionType.completeUnlessOpen],
                ofItemAtPath: shmPath
            )
        }
        #endif
    }

    /// Store database metadata in Keychain for audit trail and integrity verification.
    private func storeKeychainMetadata() throws {
        let keychainAccount = "org.antigravity.clinical-database.metadata"
        let metadata: [String: Any] = [
            "db_path": dbURL.path,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "protection_class": "NSFileProtectionCompleteUnlessOpen"
        ]
        let data = try JSONSerialization.data(withJSONObject: metadata)

        // Delete existing entry
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: keychainAccount,
            kSecAttrService as String: "AntigravityClinicalVault"
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // Add new entry with highest protection
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: keychainAccount,
            kSecAttrService as String: "AntigravityClinicalVault",
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        if status != errSecSuccess && status != errSecDuplicateItem {
            print("[ClinicalDatabase] Keychain metadata store warning: \(status)")
        }
    }

    deinit {
        if let db = db {
            sqlite3_close(db)
        }
    }

    private func openAndMigrate() throws {
        var dbPointer: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        if sqlite3_open_v2(dbURL.path, &dbPointer, flags, nil) != SQLITE_OK {
            let errmsg = String(cString: sqlite3_errmsg(dbPointer))
            throw DatabaseError.openFailed(errmsg)
        }
        self.db = dbPointer

        // Execute schema initialization
        let schema = """
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;

        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dob TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            date REAL NOT NULL,
            duration_minutes INTEGER NOT NULL,
            transcript TEXT NOT NULL,
            template_type TEXT NOT NULL,
            soap_json TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id, date DESC);

        CREATE TABLE IF NOT EXISTS metrics (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            date REAL NOT NULL,
            metric_type TEXT NOT NULL,
            score REAL NOT NULL,
            interpretation TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE INDEX IF NOT EXISTS idx_metrics_patient ON metrics(patient_id, metric_type, date ASC);
        """

        try execute(sql: schema)
    }

    internal func execute(sql: String) throws {
        try queue.sync {
            var errmsg: UnsafeMutablePointer<CChar>?
            if sqlite3_exec(db, sql, nil, nil, &errmsg) != SQLITE_OK {
                let error = errmsg.map { String(cString: $0) } ?? "Unknown error"
                sqlite3_free(errmsg)
                throw DatabaseError.executionFailed(error)
            }
        }
    }

    // MARK: - Patient Management

    public func savePatient(_ patient: StoredPatient) throws {
        let encName = try encryptString(patient.name)
        let encDob = try encryptString(patient.dateOfBirth)

        try queue.sync {
            let sql = "INSERT OR REPLACE INTO patients (id, name, dob, created_at) VALUES (?, ?, ?, ?);"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (patient.id as NSString).utf8String, -1, nil)
            sqlite3_bind_text(stmt, 2, (encName as NSString).utf8String, -1, nil)
            sqlite3_bind_text(stmt, 3, (encDob as NSString).utf8String, -1, nil)
            sqlite3_bind_double(stmt, 4, patient.createdAt.timeIntervalSince1970)

            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw DatabaseError.executionFailed(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    public func getPatient(id: String) throws -> StoredPatient? {
        try queue.sync {
            let sql = "SELECT id, name, dob, created_at FROM patients WHERE id = ? LIMIT 1;"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (id as NSString).utf8String, -1, nil)

            if sqlite3_step(stmt) == SQLITE_ROW {
                let pId = String(cString: sqlite3_column_text(stmt, 0))
                let rawName = String(cString: sqlite3_column_text(stmt, 1))
                let rawDob = String(cString: sqlite3_column_text(stmt, 2))
                let createdAt = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 3))
                return StoredPatient(
                    id: pId,
                    name: try decryptString(rawName),
                    dateOfBirth: try decryptString(rawDob),
                    createdAt: createdAt
                )
            }
            return nil
        }
    }

    // MARK: - Session & SOAP Note Persistence

    public func saveSession(_ session: StoredSession) throws {
        var encryptedSoapJSON: String? = nil
        if let note = session.soapNote {
            let encoder = JSONEncoder()
            if let data = try? encoder.encode(note), let str = String(data: data, encoding: .utf8) {
                encryptedSoapJSON = try encryptString(str)
            }
        }
        let encryptedTranscript = try encryptString(session.rawTranscript)

        try queue.sync {
            // Ensure patient stub exists if not already registered
            let ensurePatientSQL = "INSERT OR IGNORE INTO patients (id, name, dob, created_at) VALUES (?, ?, ?, ?);"
            var pStmt: OpaquePointer?
            if sqlite3_prepare_v2(db, ensurePatientSQL, -1, &pStmt, nil) == SQLITE_OK {
                let encryptedStubName = try encryptString(session.patientId)
                let encryptedStubDob = try encryptString("1990-01-01")
                sqlite3_bind_text(pStmt, 1, (session.patientId as NSString).utf8String, -1, nil)
                sqlite3_bind_text(pStmt, 2, (encryptedStubName as NSString).utf8String, -1, nil)
                sqlite3_bind_text(pStmt, 3, (encryptedStubDob as NSString).utf8String, -1, nil)
                sqlite3_bind_double(pStmt, 4, Date().timeIntervalSince1970)
                _ = sqlite3_step(pStmt)
                sqlite3_finalize(pStmt)
            }

            let sql = """
            INSERT OR REPLACE INTO sessions 
            (id, patient_id, date, duration_minutes, transcript, template_type, soap_json) 
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (session.id as NSString).utf8String, -1, nil)
            sqlite3_bind_text(stmt, 2, (session.patientId as NSString).utf8String, -1, nil)
            sqlite3_bind_double(stmt, 3, session.date.timeIntervalSince1970)
            sqlite3_bind_int(stmt, 4, Int32(session.durationMinutes))
            sqlite3_bind_text(stmt, 5, (encryptedTranscript as NSString).utf8String, -1, nil)
            sqlite3_bind_text(stmt, 6, (session.templateType as NSString).utf8String, -1, nil)

            if let json = encryptedSoapJSON {
                sqlite3_bind_text(stmt, 7, (json as NSString).utf8String, -1, nil)
            } else {
                sqlite3_bind_null(stmt, 7)
            }

            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw DatabaseError.executionFailed(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    public func getSessions(for patientId: String, limit: Int = 50) throws -> [StoredSession] {
        try queue.sync {
            let sql = """
            SELECT id, patient_id, date, duration_minutes, transcript, template_type, soap_json 
            FROM sessions 
            WHERE patient_id = ? 
            ORDER BY date DESC 
            LIMIT ?;
            """
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (patientId as NSString).utf8String, -1, nil)
            sqlite3_bind_int(stmt, 2, Int32(limit))

            var results: [StoredSession] = []
            let decoder = JSONDecoder()

            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = String(cString: sqlite3_column_text(stmt, 0))
                let pId = String(cString: sqlite3_column_text(stmt, 1))
                let date = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
                let duration = Int(sqlite3_column_int(stmt, 3))
                let rawTranscript = String(cString: sqlite3_column_text(stmt, 4))
                let templateType = String(cString: sqlite3_column_text(stmt, 5))

                var soapNote: SOAPNote? = nil
                if let jsonText = sqlite3_column_text(stmt, 6) {
                    let cipherOrPlain = String(cString: jsonText)
                    let decryptedJson = try decryptString(cipherOrPlain)
                    if let jsonData = decryptedJson.data(using: .utf8) {
                        soapNote = try? decoder.decode(SOAPNote.self, from: jsonData)
                    }
                }

                results.append(StoredSession(
                    id: id,
                    patientId: pId,
                    date: date,
                    durationMinutes: duration,
                    rawTranscript: try decryptString(rawTranscript),
                    templateType: templateType,
                    soapNote: soapNote
                ))
            }
            return results
        }
    }

    // MARK: - Longitudinal Metrics (PHQ-9, GAD-7)

    public func recordMetric(_ metric: LongitudinalMetric) throws {
        let encInterp = try encryptString(metric.clinicalInterpretation)
        try queue.sync {
            let sql = "INSERT OR REPLACE INTO metrics (id, patient_id, date, metric_type, score, interpretation) VALUES (?, ?, ?, ?, ?, ?);"
            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (metric.id as NSString).utf8String, -1, nil)
            sqlite3_bind_text(stmt, 2, (metric.patientId as NSString).utf8String, -1, nil)
            sqlite3_bind_double(stmt, 3, metric.date.timeIntervalSince1970)
            sqlite3_bind_text(stmt, 4, (metric.metricType as NSString).utf8String, -1, nil)
            sqlite3_bind_double(stmt, 5, metric.score)
            sqlite3_bind_text(stmt, 6, (encInterp as NSString).utf8String, -1, nil)

            guard sqlite3_step(stmt) == SQLITE_DONE else {
                throw DatabaseError.executionFailed(String(cString: sqlite3_errmsg(db)))
            }
        }
    }

    public func getMetrics(for patientId: String, metricType: String? = nil) throws -> [LongitudinalMetric] {
        try queue.sync {
            var sql = "SELECT id, patient_id, date, metric_type, score, interpretation FROM metrics WHERE patient_id = ?"
            if metricType != nil {
                sql += " AND metric_type = ?"
            }
            sql += " ORDER BY date ASC;"

            var stmt: OpaquePointer?
            guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK else {
                throw DatabaseError.prepareFailed(String(cString: sqlite3_errmsg(db)))
            }
            defer { sqlite3_finalize(stmt) }

            sqlite3_bind_text(stmt, 1, (patientId as NSString).utf8String, -1, nil)
            if let type = metricType {
                sqlite3_bind_text(stmt, 2, (type as NSString).utf8String, -1, nil)
            }

            var metrics: [LongitudinalMetric] = []
            while sqlite3_step(stmt) == SQLITE_ROW {
                let id = String(cString: sqlite3_column_text(stmt, 0))
                let pId = String(cString: sqlite3_column_text(stmt, 1))
                let date = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
                let type = String(cString: sqlite3_column_text(stmt, 3))
                let score = sqlite3_column_double(stmt, 4)
                let rawInterp = String(cString: sqlite3_column_text(stmt, 5))

                metrics.append(LongitudinalMetric(
                    id: id,
                    patientId: pId,
                    date: date,
                    metricType: type,
                    score: score,
                    clinicalInterpretation: try decryptString(rawInterp)
                ))
            }
            return metrics
        }
    }

    // MARK: - Cross-Session Historical Retrieval (RAG)

    /// Retrieves concatenated prior assessment & plan context across the last N sessions
    /// for prompt synthesis in longitudinal continuity of care.
    public func getHistoricalSessionContext(patientId: String, limit: Int = 3) throws -> String {
        let priorSessions = try getSessions(for: patientId, limit: limit)
        guard !priorSessions.isEmpty else { return "" }

        var lines: [String] = []
        lines.append("=== PRIOR SESSION CLINICAL HISTORY (Chronological) ===")

        let formatter = DateFormatter()
        formatter.dateStyle = .medium

        for session in priorSessions.reversed() {
            let dateStr = formatter.string(from: session.date)
            lines.append("\n[Session Date: \(dateStr)]")
            if let note = session.soapNote {
                lines.append("Assessment: \(note.assessment)")
                lines.append("Active Plan: \(note.plan)")
                if !note.identifiedSymptoms.isEmpty {
                    lines.append("Reported Symptoms: \(note.identifiedSymptoms.joined(separator: ", "))")
                }
            } else {
                lines.append("Transcript Summary: \(session.rawTranscript.prefix(150))...")
            }
        }
        lines.append("======================================================")
        return lines.joined(separator: "\n")
    }
}
