//
// Project Antigravity — Developer Kit Demo View Model
// Coordinates weight downloading, vision patch encoding, engine rollout,
// and step-level agent reflection for the SwiftUI Demo App.
//

import SwiftUI
import Combine
import AntigravityEngine

@MainActor
public final class DemoViewModel: ObservableObject {
    @Published public var selectedImage: CGImage?
    @Published public var promptText: String = "Solve the equation 2^x = 16 and verify."
    @Published public var isDownloading: Bool = false
    @Published public var downloadProgressFraction: Double = 0.0
    @Published public var downloadSpeedText: String = ""
    @Published public var isRunningReasoning: Bool = false
    @Published public var currentStatus: String = "Ready"

    @Published public var bestTraceOutput: String = ""
    @Published public var verifierScore: Float = 0.0
    @Published public var candidatesEvaluated: Int = 0
    @Published public var reflectionTriggered: Bool = false
    @Published public var ttftMs: Double = 0.0
    @Published public var totalLatencyMs: Double = 0.0
    @Published public var throughputTokPerSec: Double = 0.0
    @Published public var channelTraces: [String] = []
    @Published public var bestCandidateIndex: Int = 0

    private var engine: AntigravityEngine?
    private let weightManager = WeightManager.shared
    // No CoreML vision model ships with this demo, so the encoder cannot produce real
    // patch embeddings and encode(image:) will throw. The image branch below surfaces that
    // instead of feeding the model shape-correct colour averages and captioning the result.
    private let visionEncoder = VisionEncoder()

    public init() {
        setupEngine()
    }

    private func setupEngine() {
        do {
            self.engine = try AntigravityEngine(config: .strict4GBFootprint)
            self.currentStatus = "Engine initialized (Metal GPU ready)"
        } catch {
            self.currentStatus = "Initialization error: \(error.localizedDescription)"
        }
    }

    /// Check if TinyLlama weights are cached locally
    public var isModelCached: Bool {
        return weightManager.isModelDownloaded(type: .reasoner1B)
    }

    /// Download model weights with live streaming progress UI
    public func downloadWeights() async {
        isDownloading = true
        currentStatus = "Downloading model weights..."

        let stream = weightManager.downloadModel(type: .reasoner1B)
        for await progress in stream {
            self.downloadProgressFraction = progress.fractionCompleted
            let speedMB = progress.speedBytesPerSec / (1024 * 1024)
            self.downloadSpeedText = String(format: "%.1f MB/s", speedMB)
        }

        isDownloading = false
        currentStatus = "Model weights downloaded."
    }

    /// Run full parallel Best-of-N reasoning over prompt
    public func runReasoning() async {
        guard let engine = self.engine else {
            currentStatus = "Error: Engine not initialized."
            return
        }

        isRunningReasoning = true
        currentStatus = "Running parallel N=8 reasoning channels on Metal GPU..."

        do {
            // Ensure model weights are loaded
            if !engine.hasWeights {
                if !isModelCached {
                    await downloadWeights()
                }
                let localURL = weightManager.localURL(for: .reasoner1B)
                try engine.loadModel(at: localURL.path)
            }

            // Load the real BPE vocabulary if it was downloaded alongside the weights.
            // Without this the engine falls back to byte-level tokens, which a model
            // trained on BPE ids cannot interpret.
            let tokenizerURL = weightManager.localTokenizerURL
            if FileManager.default.fileExists(atPath: tokenizerURL.path) {
                engine.loadTokenizer(from: tokenizerURL)
            }

            // Tokenize through the engine's tokenizer. This used to map raw UTF-8 bytes to
            // ids inline (byte + 3), which bypassed the tokenizer entirely and meant a
            // correctly loaded vocabulary could never take effect.
            let inputTokens = engine.tokenizer.encode(text: promptText)

            // Perform reasoning
            let result: AntigravityGenerationResult
            if let image = selectedImage, visionEncoder.canProduceEmbeddings {
                currentStatus = "Encoding visual patches via CoreML ANE..."
                let patchEmbeddings = try visionEncoder.encode(image: image)

                currentStatus = "Executing multimodal decode loop on Metal GPU..."
                result = try await engine.reasonMultimodal(
                    textTokens: inputTokens,
                    imageEmbeddings: patchEmbeddings,
                    patchCount: visionEncoder.patchCount,
                    maxTokens: 60
                )
            } else {
                if selectedImage != nil {
                    currentStatus = "Image reasoning needs a CoreML vision model, which this "
                                  + "demo does not ship. Running text-only instead."
                }
                result = try await engine.reason(
                    promptTokens: inputTokens,
                    maxTokens: 60
                )
            }

            // Populate UI results
            self.bestTraceOutput = result.bestTraceText
            self.verifierScore = result.verifierScore
            self.candidatesEvaluated = result.candidatesEvaluated
            self.reflectionTriggered = result.reflectionTriggered
            self.ttftMs = result.timeToFirstTokenMs
            self.totalLatencyMs = result.totalLatencyMs
            self.throughputTokPerSec = result.throughputTokensPerSec
            self.channelTraces = result.candidateTraces
            self.bestCandidateIndex = result.bestCandidateIndex

            self.currentStatus = result.reflectionTriggered ?
                "Reflection Triggered (Score \(String(format: "%.2f", result.verifierScore)) < 0.75) — Refinement Verified" :
                "Reasoning Complete — Best Trace Verified"

        } catch {
            self.currentStatus = "Execution error: \(error.localizedDescription)"
        }

        isRunningReasoning = false
    }
}
