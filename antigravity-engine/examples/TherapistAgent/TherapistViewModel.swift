import SwiftUI
import Combine
import AntigravityEngine

@MainActor
public final class TherapistViewModel: ObservableObject {
    @Published public var sessionTranscript: String = """
    Patient: I've been feeling really overwhelmed lately, especially at work.
    Therapist: Tell me more about what's making you feel overwhelmed.
    Patient: It's the deadlines. My boss keeps piling on projects, and I can't sleep because I'm thinking about them.
    Therapist: It sounds like the stress is deeply impacting your rest. How long has the insomnia been happening?
    Patient: About three weeks now. I just want to quit.
    """
    
    @Published public var analysisOutput: String = ""
    @Published public var isAnalyzing: Bool = false
    @Published public var isRecording: Bool = false
    @Published public var currentStatus: String = "Ready. Air-Gapped Mode Active."
    @Published public var selectedTemplate: ClinicalTemplateType = .soap
    @Published public var currentSOAPNote: SOAPNote? = nil
    @Published public var isEncryptedInKeychain: Bool = false
    @Published public var lastExportedPDFURL: URL? = nil
    @Published public var patientIdentifier: String = "Patient #8492"
    @Published public var audioLevels: [Float] = Array(repeating: 0.08, count: 24)
    @Published public var modelLoadProgress: String = ""
    @Published public var isModelLoaded: Bool = false
    @Published public var isDownloadingModel: Bool = false
    @Published public var downloadProgress: Double = 0.0

    private var engine: AntigravityEngine?
    private let speechService = SpeechService()
    private var diarizer = SpeakerDiarizer()
    private let secureStorage = SecureStorage()
    private var database: ClinicalDatabase?
    private let templateEngine = SOAPTemplateEngine()
    private let pdfExporter = PDFExportService.shared

    /// Directory containing model weights (model.safetensors) and tokenizer (tokenizer.json).
    /// Override this before init if using a custom model path.
    public static var modelDirectory: String? = nil

    public init() {
        setupEngine()
        setupDatabase()
        setupAudioMonitoring()
        // Kick off async model loading
        Task { await loadModelWeights() }
    }

    private func setupDatabase() {
        do {
            self.database = try ClinicalDatabase()
        } catch {
            print("Clinical database init error: \(error.localizedDescription)")
        }
    }

    private func setupAudioMonitoring() {
        Task {
            await speechService.setAudioLevelHandler { [weak self] level in
                Task { @MainActor in
                    self?.pushAudioLevel(level)
                }
            }
        }
    }

    public func pushAudioLevel(_ level: Float) {
        if audioLevels.count >= 24 {
            audioLevels.removeFirst()
        }
        audioLevels.append(level)
    }

    private func setupEngine() {
        do {
            self.engine = try AntigravityEngine(config: .strict4GBFootprint)
            self.currentStatus = "Engine initialized. Loading model weights..."
        } catch {
            self.currentStatus = "Engine initialization: \(error.localizedDescription)"
        }
    }

    /// Search for model weights in standard locations and load them.
    private func loadModelWeights() async {
        guard let engine = self.engine else { return }

        let candidateDirs = Self.resolveModelDirectories()
        var loadedPath: String? = nil

        for dir in candidateDirs {
            let safetensorsPath = (dir as NSString).appendingPathComponent("model.safetensors")
            let fp16Path = (dir as NSString).appendingPathComponent("model_fp16.safetensors")
            let modelPath = FileManager.default.fileExists(atPath: fp16Path) ? fp16Path : safetensorsPath

            guard FileManager.default.fileExists(atPath: modelPath) else { continue }

            modelLoadProgress = "Loading weights from \(dir)..."
            do {
                try engine.loadModel(at: modelPath)
                loadedPath = dir
                break
            } catch {
                modelLoadProgress = "Failed to load \(modelPath): \(error.localizedDescription)"
                print("[TherapistViewModel] Failed to load model from \(dir): \(error)")
            }
        }

        guard let modelDir = loadedPath else {
            currentStatus = "⚠️ No model weights found. Place model.safetensors in ~/moat/models/tinyllama/ or click Download."
            modelLoadProgress = "No model found"
            isModelLoaded = false
            return
        }

        // Load tokenizer
        let tokenizerPath = (modelDir as NSString).appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerPath) {
            engine.loadTokenizer(from: URL(fileURLWithPath: tokenizerPath))
            modelLoadProgress = "Model and tokenizer loaded"
        } else {
            modelLoadProgress = "Model loaded (no tokenizer.json found — using byte-level fallback)"
        }

        let memMB = engine.allocatedMemoryBytes / (1024 * 1024)
        currentStatus = "Model loaded (\(memMB) MB on Metal GPU). Ready for clinical synthesis."
        isModelLoaded = true
    }

    /// Download model weights from Hugging Face CDN using WeightManager and load them.
    public func downloadAndLoadModel() async {
        isDownloadingModel = true
        currentStatus = "Downloading TinyLlama 1.1B weights from HuggingFace..."
        downloadProgress = 0.0

        let progressStream = WeightManager.shared.downloadModel(type: .reasoner1B)
        for await progress in progressStream {
            self.downloadProgress = progress.fractionCompleted
            switch progress.state {
            case .downloading:
                let mb = progress.bytesDownloaded / (1024 * 1024)
                let totalMB = progress.totalBytesExpected / (1024 * 1024)
                self.currentStatus = "Downloading: \(mb)MB / \(totalMB)MB (\(Int(progress.fractionCompleted * 100))%)"
            case .completed:
                self.currentStatus = "Download complete! Loading weights into Metal GPU..."
                await loadModelWeights()
                self.isDownloadingModel = false
                return
            case .failed(let err):
                self.currentStatus = "Download failed: \(err)"
                self.isDownloadingModel = false
                return
            case .idle:
                break
            }
        }
        self.isDownloadingModel = false
    }

    /// Resolve candidate directories to search for model weights.
    private static func resolveModelDirectories() -> [String] {
        var dirs: [String] = []

        // 1. Explicit override
        if let explicit = modelDirectory {
            dirs.append(explicit)
        }

        // 2. Standard project model directories
        let homeDir = NSHomeDirectory()
        dirs.append(contentsOf: [
            "\(homeDir)/moat/models/tinyllama",
            "\(homeDir)/moat/models/qwen",
            "\(homeDir)/moat/models/qwen3_5_0_8b_4bit",
            "\(homeDir)/moat/models/qwen3_5_2b_4bit",
        ])

        // 3. WeightManager storage directory
        dirs.append(WeightManager.shared.storageDirectory.path)

        // 4. App bundle resources (for distributed .app builds)
        if let bundlePath = Bundle.main.resourcePath {
            dirs.append(bundlePath)
        }

        // 5. Documents directory (for user-provided models on iOS)
        if let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            dirs.append(docsDir.appendingPathComponent("models").path)
        }

        return dirs
    }

    public func toggleRecording() async {
        if isRecording {
            await stopRecording()
        } else {
            await startRecording()
        }
    }

    public func startRecording() async {
        do {
            let authorized = await speechService.requestAuthorization()
            guard authorized else {
                currentStatus = "Microphone or Speech Recognition permission denied."
                return
            }

            try await speechService.startRecording()
            isRecording = true
            currentStatus = "Recording & transcribing live on-device (Zero network egress)..."
        } catch {
            currentStatus = "Recording error: \(error.localizedDescription)"
            isRecording = false
        }
    }

    public func stopRecording() async {
        isRecording = false
        let transcript = await speechService.stopRecording()
        if !transcript.isEmpty {
            self.sessionTranscript = transcript
            currentStatus = "Live session recorded. Ready for clinical synthesis."
        } else {
            currentStatus = "Recording stopped."
        }
    }

    public func runAnalysis() async {
        isAnalyzing = true
        currentStatus = "Processing clinical reasoning on Apple Silicon Metal GPU..."

        do {
            // Retrieve longitudinal historical context from SQLite database
            var fullTranscript = sessionTranscript
            if let db = self.database {
                let priorContext = (try? db.getHistoricalSessionContext(patientId: patientIdentifier, limit: 3)) ?? ""
                if !priorContext.isEmpty {
                    fullTranscript = priorContext + "\n\n" + sessionTranscript
                }
            }

            let prompt = templateEngine.buildPrompt(
                transcript: fullTranscript,
                patientId: patientIdentifier,
                template: selectedTemplate
            )

            var generatedText = ""
            if let engine = self.engine, engine.hasWeights {
                let result = try await engine.generateText(
                    prompt: prompt,
                    maxTokens: 256,
                    temperature: 0.3,
                    topP: 0.85
                )
                generatedText = result.bestTraceText
            } else {
                // No model weights loaded — show honest error, not fake output
                self.currentStatus = "⚠️ Cannot generate: No model weights loaded. Load a model first."
                self.analysisOutput = "[ERROR] Model weights not loaded.\n\nTo generate real clinical notes, you need to:\n1. Download a model (e.g., TinyLlama 1.1B) to ~/moat/models/tinyllama/\n2. Ensure model.safetensors and tokenizer.json are present\n3. Restart the app\n\nThe model will load automatically on startup."
                isAnalyzing = false
                return
            }

            let note = templateEngine.parseModelOutput(
                generatedText,
                patientId: patientIdentifier,
                clinician: "Dr. Alex Vance, PsyD",
                template: selectedTemplate
            )

            self.currentSOAPNote = note
            self.analysisOutput = note.formattedReport
            self.currentStatus = "SOAP Note Generated Offline via Apple Silicon Edge Core."

            // 1. Hardware Keychain Encrypted Backup
            try await saveEncryptedRecord(note: note)

            // 2. Structured SQLite Encrypted Record
            if let db = self.database {
                let record = StoredSession(
                    patientId: patientIdentifier,
                    durationMinutes: 50,
                    rawTranscript: sessionTranscript,
                    templateType: selectedTemplate.rawValue,
                    soapNote: note
                )
                try? db.saveSession(record)
            }
        } catch {
            self.currentStatus = "Synthesis error: \(error.localizedDescription)"
        }

        isAnalyzing = false
    }

    public func saveEncryptedRecord(note: SOAPNote) async throws {
        let encoder = JSONEncoder()
        let noteData = try encoder.encode(note)
        let accountKey = "clinical_session_\(note.id.uuidString)"
        
        try secureStorage.save(account: accountKey, data: noteData)
        self.isEncryptedInKeychain = true
    }

    public func exportPDF() {
        guard let note = currentSOAPNote else { return }
        do {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("SOAP_\(note.patientIdentifier.replacingOccurrences(of: " ", with: "_")).pdf")
            try pdfExporter.exportToFile(note: note, destinationURL: tempURL)
            self.lastExportedPDFURL = tempURL
            self.currentStatus = "Exported encrypted clinical PDF: \(tempURL.lastPathComponent)"
        } catch {
            self.currentStatus = "PDF Export error: \(error.localizedDescription)"
        }
    }
}
