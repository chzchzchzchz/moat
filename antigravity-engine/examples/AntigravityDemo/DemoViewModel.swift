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

    private var engine: AntigravityEngine?
    private let weightManager = WeightManager.shared
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

            if case .completed = progress.state {
                self.isDownloading = false
                self.currentStatus = "Model weights downloaded and cached successfully!"
            } else if case .failed(let err) = progress.state {
                self.isDownloading = false
                self.currentStatus = "Download error: \(err)"
            }
        }
    }

    /// Execute parallel Best-of-N reasoning & step-level reflection
    public func runReasoning() async {
        guard let engine = self.engine else { return }

        isRunningReasoning = true
        bestTraceOutput = ""
        channelTraces = []
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

            // Perform reasoning
            let result: AntigravityGenerationResult
            if let image = selectedImage {
                currentStatus = "Encoding visual patches via CoreML ANE..."
                let patchEmbeddings = try visionEncoder.encode(image: image)
                let dummyTokens: [Int32] = [1, 10, 100, 200]

                currentStatus = "Executing multimodal decode loop on Metal GPU..."
                result = try await engine.reasonMultimodal(
                    textTokens: dummyTokens,
                    imageEmbeddings: patchEmbeddings,
                    patchCount: visionEncoder.patchCount,
                    maxTokens: 60
                )
            } else {
                let dummyTokens: [Int32] = [1, 5, 20, 50, 100]
                result = try await engine.reason(
                    promptTokens: dummyTokens,
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

            self.currentStatus = result.reflectionTriggered ?
                "Reflection Triggered (Score \(String(format: "%.2f", result.verifierScore)) < 0.75) — Refinement Verified" :
                "Reasoning Complete — Best Trace Verified"

        } catch {
            self.currentStatus = "Execution error: \(error.localizedDescription)"
        }

        isRunningReasoning = false
    }
}
