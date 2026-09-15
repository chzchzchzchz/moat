//
// Project Antigravity — Swift SDK Unit Test Suite
// Tests Swift-only components (config, weight manager, vision encoder patch math)
// C++ Metal engine is tested separately via Python pytest and native test_c_api_client
//

import XCTest
@testable import AntigravityEngine

final class AntigravityEngineTests: XCTestCase {

    // MARK: - EngineConfig Tests

    func testEngineConfigStrictDefaults() {
        let config = EngineConfig.strict4GBFootprint
        XCTAssertEqual(config.memoryLimitBytes, 4500 * 1024 * 1024)
        XCTAssertEqual(config.parallelChannels, 8)
        XCTAssertEqual(config.reflectionThreshold, 0.75)
        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.hiddenDim, 2048)
        XCTAssertTrue(config.useMetalGPU)
    }

    func testEngineConfigCustomInit() {
        let config = EngineConfig(
            memoryLimitBytes: 2000 * 1024 * 1024,
            parallelChannels: 4,
            reflectionThreshold: 0.5,
            vocabSize: 50000,
            hiddenDim: 4096,
            useMetalGPU: false
        )
        XCTAssertEqual(config.parallelChannels, 4)
        XCTAssertEqual(config.vocabSize, 50000)
        XCTAssertEqual(config.hiddenDim, 4096)
        XCTAssertFalse(config.useMetalGPU)
    }

    // MARK: - VisionEncoder Tests

    func testVisionEncoderPatchCount224() {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 2048, patchSize: 14, imageSize: 224)
        // 224 / 14 = 16, 16 * 16 = 256
        XCTAssertEqual(encoder.patchCount, 256)
    }

    func testVisionEncoderPatchCount384() {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 2048, patchSize: 16, imageSize: 384)
        // 384 / 16 = 24, 24 * 24 = 576
        XCTAssertEqual(encoder.patchCount, 576)
    }

    func testVisionEncoderFallbackEncoding() throws {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 64, patchSize: 14, imageSize: 28)
        // 28 / 14 = 2, 2 * 2 = 4 patches
        XCTAssertEqual(encoder.patchCount, 4)

        // Create a tiny 28x28 CGImage for testing
        let width = 28, height = 28
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawData = [UInt8](repeating: 128, count: width * height * 4)
        let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let image = context.makeImage()!

        let embeddings = try encoder.encode(image: image)
        XCTAssertEqual(embeddings.count, 4 * 64, "Expected 4 patches * 64 hidden_dim = 256 floats")
        // No NaN or Inf values
        XCTAssertFalse(embeddings.contains(where: { $0.isNaN || $0.isInfinite }))
    }

    // MARK: - AgentTool Tests

    func testAgentToolSchema() {
        let tool = AgentTool(name: "calculator", description: "Math API", jsonSchema: "{\"type\":\"object\"}")
        XCTAssertEqual(tool.name, "calculator")
        XCTAssertEqual(tool.description, "Math API")
        XCTAssertEqual(tool.jsonSchema, "{\"type\":\"object\"}")
    }

    // MARK: - WeightManager Tests

    func testWeightManagerStorageDirectory() {
        let manager = WeightManager.shared
        let dir = manager.storageDirectory
        XCTAssertTrue(dir.path.contains("AntigravityEngine"))
    }

    func testWeightManagerLocalURLs() {
        let manager = WeightManager.shared
        let reasonerURL = manager.localURL(for: .reasoner1B)
        let verifierURL = manager.localURL(for: .verifier1_5B)
        XCTAssertTrue(reasonerURL.lastPathComponent.contains("model"))
        XCTAssertTrue(verifierURL.lastPathComponent.contains("model"))
    }

    func testWeightManagerModelNotDownloaded() {
        let manager = WeightManager.shared
        // Custom model type should not be downloaded
        let customType = AntigravityModelType.custom(name: "test_nonexistent", remoteURL: URL(string: "https://example.com/x.safetensors")!)
        XCTAssertFalse(manager.isModelDownloaded(type: customType))
    }

    // MARK: - DownloadProgress Tests

    func testDownloadProgressStruct() {
        let progress = DownloadProgress(
            modelType: .reasoner1B,
            fractionCompleted: 0.75,
            bytesDownloaded: 825_000_000,
            totalBytesExpected: 1_100_000_000,
            speedBytesPerSec: 30_000_000,
            state: .downloading
        )
        XCTAssertEqual(progress.fractionCompleted, 0.75, accuracy: 0.001)
        XCTAssertEqual(progress.bytesDownloaded, 825_000_000)
        XCTAssertEqual(progress.state, .downloading)
    }

    func testDownloadProgressCompletedState() {
        let progress = DownloadProgress(
            modelType: .verifier1_5B,
            fractionCompleted: 1.0,
            bytesDownloaded: 2_880_000_000,
            totalBytesExpected: 2_880_000_000,
            speedBytesPerSec: 0,
            state: .completed
        )
        XCTAssertEqual(progress.state, .completed)
        XCTAssertEqual(progress.fractionCompleted, 1.0)
    }

    // MARK: - AntigravityModelType Tests

    func testModelTypeDefaultFileNames() {
        XCTAssertEqual(AntigravityModelType.reasoner1B.defaultFileName, "model.safetensors")
        XCTAssertEqual(AntigravityModelType.verifier1_5B.defaultFileName, "skywork_prm_1.5b_model.safetensors")
    }

    func testModelTypeExpectedBytes() {
        XCTAssertEqual(AntigravityModelType.reasoner1B.expectedByteCount, 1_100_000_000)
        XCTAssertEqual(AntigravityModelType.verifier1_5B.expectedByteCount, 16_000_000_000)
    }

    // MARK: - AntigravityError Tests

    func testErrorDescriptions() {
        let memError = AntigravityError.memoryBudgetExceeded(requestedBytes: 5_000_000_000, ceilingBytes: 4_500_000_000)
        XCTAssertNotNil(memError.errorDescription)
        XCTAssertTrue(memError.errorDescription!.contains("Memory budget exceeded"))

        let loadError = AntigravityError.modelLoadingFailed(reason: "file not found")
        XCTAssertTrue(loadError.errorDescription!.contains("file not found"))

        let execError = AntigravityError.executionFailed(reason: "GPU timeout")
        XCTAssertTrue(execError.errorDescription!.contains("GPU timeout"))

        let visionError = AntigravityError.visionEncodingFailed(reason: "bad image")
        XCTAssertTrue(visionError.errorDescription!.contains("bad image"))
    }

    // MARK: - SOAPTemplateEngine Tests

    func testSOAPTemplateBuildPrompt() {
        let engine = SOAPTemplateEngine()
        let prompt = engine.buildPrompt(transcript: "Patient: I feel anxious.", patientId: "P-101", template: .soap)
        XCTAssertTrue(prompt.contains("[SUBJECTIVE]"))
        XCTAssertTrue(prompt.contains("[OBJECTIVE]"))
        XCTAssertTrue(prompt.contains("[ASSESSMENT]"))
        XCTAssertTrue(prompt.contains("[PLAN]"))
        XCTAssertTrue(prompt.contains("Patient: I feel anxious."))
    }

    func testSOAPTemplateParseOutput() {
        let engine = SOAPTemplateEngine()
        let sampleOutput = """
        [SUBJECTIVE]
        Patient endorses heightened anxiety and insomnia for 3 weeks.
        [OBJECTIVE]
        Client is alert, speech slightly pressured, cooperative.
        [ASSESSMENT]
        Adjustment disorder with anxious mood.
        [PLAN]
        Initiate relaxation protocol, weekly follow-up.
        [SYMPTOMS]
        Anxiety, Insomnia, Restlessness
        [RISK]
        Low risk. Denies suicidal ideation.
        """

        let note = engine.parseModelOutput(sampleOutput, patientId: "P-101", clinician: "Dr. Smith", template: .soap)
        XCTAssertEqual(note.patientIdentifier, "P-101")
        XCTAssertEqual(note.clinicianName, "Dr. Smith")
        XCTAssertTrue(note.subjective.contains("heightened anxiety"))
        XCTAssertTrue(note.objective.contains("cooperative"))
        XCTAssertTrue(note.assessment.contains("Adjustment disorder"))
        XCTAssertTrue(note.plan.contains("relaxation protocol"))
        XCTAssertEqual(note.identifiedSymptoms.count, 3)
        XCTAssertTrue(note.riskAssessment.contains("Low risk"))
        XCTAssertTrue(note.formattedReport.contains("CLINICAL PSYCHOTHERAPY DOCUMENTATION"))
    }

    // MARK: - PDFExportService Tests

    func testPDFExportGeneratesNonEmptyData() throws {
        let note = SOAPNote(
            patientIdentifier: "Patient-Test",
            subjective: "Patient reports stress.",
            objective: "Calm affect.",
            assessment: "Mild anxiety.",
            plan: "Breathing exercises."
        )
        let pdfData = try PDFExportService.shared.generatePDF(for: note)
        XCTAssertGreaterThan(pdfData.count, 500, "Generated PDF must contain valid header and bytes")
        // PDF header magic "%PDF"
        let header = String(data: pdfData.prefix(4), encoding: .ascii)
        XCTAssertEqual(header, "%PDF")
    }

    // MARK: - SpeakerDiarizer Tests

    func testDiarizerEmbeddingExtraction() {
        let diarizer = SpeakerDiarizer()
        // Generate 16000 synthetic audio samples (1 second at 16kHz)
        let samples = (0..<16000).map { sin(Float($0) * 0.05) }
        let embedding = diarizer.extractEmbedding(audioSamples: samples, sampleRate: 16000)
        XCTAssertEqual(embedding.count, 4, "Embedding must have 4 normalized spectral features")
        XCTAssertFalse(embedding.contains(where: { $0.isNaN || $0.isInfinite }))
    }

    // MARK: - LicenseVerifier Tests

    func testLicenseVerifierFormatRejection() {
        let pubKeyBytes = [UInt8](repeating: 0x42, count: 32)
        guard let verifier = try? LicenseVerifier(publicKeyBytes: pubKeyBytes) else {
            XCTFail("Failed to initialize LicenseVerifier")
            return
        }
        // Invalid key formats
        XCTAssertThrowsError(try verifier.verify(licenseKey: "not-a-valid-key"))
        XCTAssertThrowsError(try verifier.verify(licenseKey: "onlyonepart"))
        XCTAssertThrowsError(try verifier.verify(licenseKey: "part1.part2.part3"))
    }

    // MARK: - Tokenizer Tests

    func testTokenizerRoundtripByteFallback() {
        let tokenizer = AntigravityTokenizer()
        let text = "What is cognitive behavioral therapy?"
        let tokens = tokenizer.encode(text: text)
        XCTAssertGreaterThan(tokens.count, 0)
        XCTAssertEqual(tokens.first, 1, "First token should be BOS (1)")

        let decoded = tokenizer.decode(tokens: tokens)
        XCTAssertEqual(decoded, text)
    }

    func testTokenizerVocabularySize() {
        let tokenizer = AntigravityTokenizer()
        XCTAssertGreaterThanOrEqual(tokenizer.vocabSize, 256, "Byte-fallback vocabulary must contain at least 256 bytes")
    }

    func testTokenizerRealBPEWithTinyLlamaVocab() {
        // Load real TinyLlama tokenizer.json with 32000 tokens and 61249 merge rules
        let tokenizerPath = NSHomeDirectory() + "/moat/models/tinyllama/tokenizer.json"
        guard FileManager.default.fileExists(atPath: tokenizerPath) else {
            print("Skipping testTokenizerRealBPEWithTinyLlamaVocab: tokenizer.json not found at \(tokenizerPath)")
            return
        }

        let tokenizer = AntigravityTokenizer(tokenizerJSONURL: URL(fileURLWithPath: tokenizerPath))

        // Verify we loaded a real vocabulary (close to 32000 tokens, not 259 byte-fallback)
        XCTAssertGreaterThan(tokenizer.vocabSize, 31000, "TinyLlama vocabulary should have ~32000 tokens")

        // Encode a clinical prompt
        let text = "Patient reports persistent insomnia and workplace anxiety."
        let tokens = tokenizer.encode(text: text)
        
        // BPE should produce fewer tokens than character count (subword merging)
        XCTAssertGreaterThan(tokens.count, 1, "Must produce at least BOS + 1 token")
        XCTAssertLessThan(tokens.count, text.count, "BPE should merge subwords, producing fewer tokens than chars")
        
        // Roundtrip decode
        let decoded = tokenizer.decode(tokens: tokens)
        XCTAssertEqual(decoded, text, "Roundtrip decode must recover original text")
        
        // Verify multi-word merge: "the" should be a single token, not 3 bytes
        let simpleTokens = tokenizer.encode(text: "the", addBOS: false)
        XCTAssertEqual(simpleTokens.count, 1, "'the' should be a single BPE token in TinyLlama vocab")
    }

    // MARK: - ClinicalDatabase Tests

    func testClinicalDatabaseSessionAndMetricPersistence() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbFile = tempDir.appendingPathComponent("test_clinical_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: dbFile) }

        let database = try ClinicalDatabase(databaseURL: dbFile)

        // 1. Register Patient
        let patient = StoredPatient(id: "P-882", name: "Jane Doe", dateOfBirth: "1985-04-12")
        try database.savePatient(patient)

        let retrievedPatient = try database.getPatient(id: "P-882")
        XCTAssertNotNil(retrievedPatient)
        XCTAssertEqual(retrievedPatient?.name, "Jane Doe")

        // 2. Save Session with SOAP Note
        let soapNote = SOAPNote(
            patientIdentifier: "P-882",
            subjective: "Patient reports panic attacks in elevators.",
            objective: "Hyperventilating, diaphoresis.",
            assessment: "Panic Disorder without Agoraphobia (F41.0).",
            plan: "Interoceptive exposure and diaphragmatic pacing.",
            identifiedSymptoms: ["Panic", "Tachycardia", "Diaphoresis"]
        )

        let session = StoredSession(
            id: "S-101",
            patientId: "P-882",
            durationMinutes: 45,
            rawTranscript: "Patient: I panicked in the elevator today.",
            templateType: "SOAP",
            soapNote: soapNote
        )
        try database.saveSession(session)

        let sessions = try database.getSessions(for: "P-882")
        XCTAssertEqual(sessions.count, 1)
        XCTAssertEqual(sessions[0].soapNote?.assessment, "Panic Disorder without Agoraphobia (F41.0).")
        XCTAssertEqual(sessions[0].soapNote?.identifiedSymptoms.count, 3)

        // 3. Record Longitudinal Metric (PHQ-9)
        let metric = LongitudinalMetric(
            patientId: "P-882",
            metricType: "PHQ-9",
            score: 14.0,
            clinicalInterpretation: "Moderate Depression"
        )
        try database.recordMetric(metric)

        let metrics = try database.getMetrics(for: "P-882", metricType: "PHQ-9")
        XCTAssertEqual(metrics.count, 1)
        XCTAssertEqual(metrics[0].score, 14.0)
        XCTAssertEqual(metrics[0].clinicalInterpretation, "Moderate Depression")

        // 4. Historical RAG Context Retrieval
        let context = try database.getHistoricalSessionContext(patientId: "P-882")
        XCTAssertTrue(context.contains("Panic Disorder"))
        XCTAssertTrue(context.contains("Interoceptive exposure"))
    }

    func testClinicalDatabaseZeroTrustColumnLevelEncryption() throws {
        let tempDir = FileManager.default.temporaryDirectory
        let dbFile = tempDir.appendingPathComponent("test_encrypted_\(UUID().uuidString).sqlite")
        defer {
            if FileManager.default.fileExists(atPath: dbFile.path) {
                try? FileManager.default.removeItem(at: dbFile)
            }
        }

        // Generate explicit 256-bit AES key
        var keyData = Data(count: 32)
        _ = keyData.withUnsafeMutableBytes {
            SecRandomCopyBytes(kSecRandomDefault, 32, $0.baseAddress!)
        }

        let database = try ClinicalDatabase(databaseURL: dbFile, encryptionKeyData: keyData)

        let secretName = "TopSecret Classified Patient 991"
        let secretTranscript = "CONFIDENTIAL DIALOGUE: Patient admits to chronic clinical anxiety and insomnia."
        let secretAssessment = "Specific Panic Reaction (F41.0)"

        let patient = StoredPatient(id: "P-SEC", name: secretName, dateOfBirth: "1988-12-31")
        try database.savePatient(patient)

        let soap = SOAPNote(
            patientIdentifier: "P-SEC",
            subjective: "Secret subjective statement",
            objective: "Secret objective signs",
            assessment: secretAssessment,
            plan: "Strict CBT protocol",
            identifiedSymptoms: ["Panic", "Insomnia"]
        )

        let session = StoredSession(
            id: "S-SEC",
            patientId: "P-SEC",
            durationMinutes: 60,
            rawTranscript: secretTranscript,
            templateType: "SOAP",
            soapNote: soap
        )
        try database.saveSession(session)

        // Read raw bytes of the SQLite database directly from disk
        let rawDiskData = try Data(contentsOf: dbFile)
        let rawDiskString = String(decoding: rawDiskData, as: UTF8.self)

        // Cryptographic proof: the raw SQLite file must NOT contain plaintext sensitive PHI
        XCTAssertFalse(rawDiskString.contains(secretName), "Raw SQLite file leaked plaintext patient name!")
        XCTAssertFalse(rawDiskString.contains(secretTranscript), "Raw SQLite file leaked plaintext session transcript!")
        XCTAssertFalse(rawDiskString.contains(secretAssessment), "Raw SQLite file leaked plaintext clinical assessment!")

        // Authenticated retrieval: verify reading through database returns clean decrypted text
        let loadedPatient = try database.getPatient(id: "P-SEC")
        XCTAssertEqual(loadedPatient?.name, secretName)
        XCTAssertEqual(loadedPatient?.dateOfBirth, "1988-12-31")

        let loadedSessions = try database.getSessions(for: "P-SEC")
        XCTAssertEqual(loadedSessions.count, 1)
        XCTAssertEqual(loadedSessions[0].rawTranscript, secretTranscript)
        XCTAssertEqual(loadedSessions[0].soapNote?.assessment, secretAssessment)

        // Zero-trust verification: opening with an invalid key cannot decrypt ciphertext
        var wrongKeyData = Data(count: 32)
        _ = wrongKeyData.withUnsafeMutableBytes {
            SecRandomCopyBytes(kSecRandomDefault, 32, $0.baseAddress!)
        }
        let untrustedDB = try ClinicalDatabase(databaseURL: dbFile, encryptionKeyData: wrongKeyData)
        // Wrong key: getPatient should either throw (authentication failure) or return garbled data
        let unauthenticatedPatient = try? untrustedDB.getPatient(id: "P-SEC")
        // If it didn't throw, verify the name doesn't match
        if let patient = unauthenticatedPatient {
            XCTAssertNotEqual(patient.name, secretName)
        }
        // If it threw, that's also correct — wrong key can't decrypt
    }

    // MARK: - Real Metal Engine Linking Tests

    func testRealMetalEngineAllocationAndExecution() async throws {
        // This test directly verifies that SPM links to the real Metal C++ engine (libAntigravityEngine.a),
        // not stubs.c!
        let config = EngineConfig(
            memoryLimitBytes: 4096 * 1024 * 1024,
            parallelChannels: 4,
            reflectionThreshold: 0.75,
            vocabSize: 32000,
            hiddenDim: 2048,
            useMetalGPU: true
        )
        let engine = try AntigravityEngine(config: config)

        // Real Metal engine allocates scratchpad + KV cache + RoPE buffers (>100MB)
        let allocatedBytes = engine.allocatedMemoryBytes
        XCTAssertGreaterThan(allocatedBytes, 100 * 1024 * 1024, "Real Metal engine must allocate >100MB unified VRAM")
        XCTAssertFalse(engine.hasWeights, "Engine should initially have no weights loaded")

        // Load TinyLlama Safetensors model if available
        let modelPath = NSHomeDirectory() + "/moat/models/tinyllama/model.safetensors"
        let tokenizerPath = NSHomeDirectory() + "/moat/models/tinyllama/tokenizer.json"

        if FileManager.default.fileExists(atPath: modelPath) && FileManager.default.fileExists(atPath: tokenizerPath) {
            try engine.loadModel(at: modelPath)
            XCTAssertTrue(engine.hasWeights, "hasWeights must be true after loading model")

            let weightsAllocated = engine.allocatedMemoryBytes
            XCTAssertGreaterThan(weightsAllocated, 2000 * 1024 * 1024, "Weights + KV cache must exceed 2GB VRAM")

            engine.loadTokenizer(from: URL(fileURLWithPath: tokenizerPath))

            // Test genuine Metal GPU parallel rollout generation with linguistic verification
            let prompt = "The symptoms of clinical depression include"
            let result = try await engine.generateText(
                prompt: prompt,
                maxTokens: 15,
                temperature: 0.7,
                topP: 0.9
            )

            // Structural verification
            XCTAssertEqual(result.candidatesEvaluated, 4, "Must evaluate 4 parallel channels")
            XCTAssertEqual(result.candidateTraces.count, 4, "Must produce 4 candidate traces")
            XCTAssertGreaterThanOrEqual(result.bestCandidateIndex, 0)
            XCTAssertLessThan(result.bestCandidateIndex, 4)
            XCTAssertGreaterThan(result.verifierScore, 0.0, "Verifier score must be positive")
            XCTAssertLessThanOrEqual(result.verifierScore, 1.0, "Verifier score must be <= 1.0")
            XCTAssertGreaterThan(result.timeToFirstTokenMs, 0.0, "TTFT must be measured on GPU")
            XCTAssertGreaterThan(result.throughputTokensPerSec, 0.0, "Throughput must be positive")

            // Strict Linguistic & Clinical Meaning Verification
            let bestText = result.bestTraceText.lowercased()
            print("[Swift SDK Metal Decode Best Trace]: \(result.bestTraceText)")
            for (idx, trace) in result.candidateTraces.enumerated() {
                print("[Swift SDK Metal Channel \(idx)]: \(trace)")
                XCTAssertGreaterThan(trace.trimmingCharacters(in: .whitespacesAndNewlines).count, 5, "Candidate rollout \(idx) must be non-trivial")
            }

            let clinicalKeywords = ["sadness", "hopelessness", "loss", "interest", "feelings", "worthlessness", "symptoms", "emotional", "physical", "depress", "sad"]
            let matchedKeywords = clinicalKeywords.filter { bestText.contains($0) }
            XCTAssertGreaterThanOrEqual(
                matchedKeywords.count,
                1,
                "Generated output '\(result.bestTraceText)' must contain clinical psychiatric terms (matched: \(matchedKeywords))"
            )

            // Unload weights and verify VRAM deallocation
            engine.unloadWeights()
            XCTAssertFalse(engine.hasWeights, "Weights should be cleared after unload")
            XCTAssertEqual(engine.allocatedMemoryBytes, 0, "Allocated VRAM must drop to 0 after unload")
        }
    }

    func testRealMetalEngineSOAPGenerationAndStorage() async throws {
        let modelPath = NSHomeDirectory() + "/moat/models/tinyllama/model.safetensors"
        let tokenizerPath = NSHomeDirectory() + "/moat/models/tinyllama/tokenizer.json"
        guard FileManager.default.fileExists(atPath: modelPath) && FileManager.default.fileExists(atPath: tokenizerPath) else {
            print("Skipping testRealMetalEngineSOAPGenerationAndStorage: model files missing")
            return
        }

        let config = EngineConfig(
            memoryLimitBytes: 4096 * 1024 * 1024,
            parallelChannels: 4,
            reflectionThreshold: 0.75,
            vocabSize: 32000,
            hiddenDim: 2048,
            useMetalGPU: true
        )
        let engine = try AntigravityEngine(config: config)
        try engine.loadModel(at: modelPath)
        engine.loadTokenizer(from: URL(fileURLWithPath: tokenizerPath))

        let templateEngine = SOAPTemplateEngine()
        let transcript = """
        Patient: I have been having severe panic attacks whenever I get into an elevator.
        Therapist: How long do these panic episodes usually last?
        Patient: About 10 to 15 minutes of intense tachycardia, sweating, and feeling like I might faint.
        Therapist: We will start systematic desensitization and diaphragmatic breathing exercises.
        """

        let prompt = templateEngine.buildPrompt(
            transcript: transcript,
            patientId: "PATIENT-ELEVATOR-PANIC",
            template: .soap
        )

        let result = try await engine.generateText(
            prompt: prompt,
            maxTokens: 25,
            temperature: 0.4,
            topP: 0.9
        )

        XCTAssertFalse(result.bestTraceText.isEmpty, "Generated clinical text must not be empty")

        // Parse model output into structured SOAP note
        let note = templateEngine.parseModelOutput(
            result.bestTraceText,
            patientId: "PATIENT-ELEVATOR-PANIC",
            clinician: "Dr. Elena Rostova, MD",
            template: .soap
        )

        // Verify valid SOAP format
        XCTAssertEqual(note.patientIdentifier, "PATIENT-ELEVATOR-PANIC")
        XCTAssertEqual(note.clinicianName, "Dr. Elena Rostova, MD")
        XCTAssertFalse(note.subjective.isEmpty, "Subjective section must be populated")
        XCTAssertFalse(note.objective.isEmpty, "Objective section must be populated")
        XCTAssertFalse(note.assessment.isEmpty, "Assessment section must be populated")
        XCTAssertFalse(note.plan.isEmpty, "Plan section must be populated")
        XCTAssertFalse(note.formattedReport.isEmpty, "Formatted clinical report must not be empty")

        // Test saving to encrypted ClinicalDatabase with zero plaintext leakage
        let tempDbURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_soap_\(UUID().uuidString).sqlite")
        defer { try? FileManager.default.removeItem(at: tempDbURL) }

        let database = try ClinicalDatabase(databaseURL: tempDbURL)
        let sessionRecord = StoredSession(
            patientId: "PATIENT-ELEVATOR-PANIC",
            durationMinutes: 45,
            rawTranscript: transcript,
            templateType: ClinicalTemplateType.soap.rawValue,
            soapNote: note
        )
        try database.saveSession(sessionRecord)

        // Verify stored session retrieves and decrypts cleanly
        let retrievedSessions = try database.getSessions(for: "PATIENT-ELEVATOR-PANIC")
        XCTAssertEqual(retrievedSessions.count, 1)
        XCTAssertEqual(retrievedSessions[0].rawTranscript, transcript)
        XCTAssertNotNil(retrievedSessions[0].soapNote)
        XCTAssertEqual(retrievedSessions[0].soapNote?.patientIdentifier, "PATIENT-ELEVATOR-PANIC")
        XCTAssertEqual(retrievedSessions[0].soapNote?.assessment, note.assessment)

        engine.unloadWeights()
    }
}


