//
// Project Antigravity — Public Swift API Wrapper & Engine SDK
// Apple Silicon Edge Engine (iOS / macOS, M1-M4, A17 Pro, A18 Pro)
//

import Foundation
import CoreGraphics
import CAntigravityEngine

/// Configuration for the Antigravity local inference engine.
public struct EngineConfig: Sendable {
    public let memoryLimitBytes: Int64
    public let parallelChannels: Int
    public let reflectionThreshold: Float
    public let vocabSize: Int
    public let hiddenDim: Int
    public let useMetalGPU: Bool

    public static let strict4GBFootprint = EngineConfig(
        memoryLimitBytes: 4500 * 1024 * 1024,
        parallelChannels: 8,
        reflectionThreshold: 0.75,
        vocabSize: 32000,
        hiddenDim: 2048,
        useMetalGPU: true
    )

    public init(
        memoryLimitBytes: Int64 = 4500 * 1024 * 1024,
        parallelChannels: Int = 8,
        reflectionThreshold: Float = 0.75,
        vocabSize: Int = 32000,
        hiddenDim: Int = 2048,
        useMetalGPU: Bool = true
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.parallelChannels = parallelChannels
        self.reflectionThreshold = reflectionThreshold
        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim
        self.useMetalGPU = useMetalGPU
    }
}

/// Generation Result returned by the Antigravity Engine
public struct AntigravityGenerationResult: Sendable {
    public let bestTraceText: String
    public let verifierScore: Float
    public let bestCandidateIndex: Int
    public let candidateTraces: [String]
    public let candidatesEvaluated: Int
    public let reflectionTriggered: Bool
    public let timeToFirstTokenMs: Double
    public let totalLatencyMs: Double
    public let totalTokensGenerated: Int
    public let throughputTokensPerSec: Double
}

/// Errors for Antigravity Engine
public enum AntigravityError: Error, LocalizedError {
    case memoryBudgetExceeded(requestedBytes: Int64, ceilingBytes: Int64)
    case modelLoadingFailed(reason: String)
    case executionFailed(reason: String)
    case visionEncodingFailed(reason: String)

    public var errorDescription: String? {
        switch self {
        case .memoryBudgetExceeded(let req, let ceiling):
            return "Memory budget exceeded: requested \(req / 1024 / 1024)MB, ceiling is \(ceiling / 1024 / 1024)MB"
        case .modelLoadingFailed(let reason):
            return "Model loading failed: \(reason)"
        case .executionFailed(let reason):
            return "Engine execution failed: \(reason)"
        case .visionEncodingFailed(let reason):
            return "Vision encoding failed: \(reason)"
        }
    }
}

/// Primary public entry point for Antigravity Engine on iOS / macOS
public final class AntigravityEngine {
    private var engineHandle: OpaquePointer?
    public let config: EngineConfig
    public private(set) var loadedModelPath: String?
    private let queue = DispatchQueue(label: "org.antigravity.engine")

    public init(config: EngineConfig = .strict4GBFootprint) throws {
        self.config = config
        var cConfig = AntigravityConfig(
            n_channels: Int32(config.parallelChannels),
            vocab_size: Int32(config.vocabSize),
            hidden_dim: Int32(config.hiddenDim),
            max_seq_len: 2048,
            use_metal_gpu: config.useMetalGPU
        )

        self.engineHandle = AntigravityEngineCreate(&cConfig)
        guard self.engineHandle != nil else {
            throw AntigravityError.executionFailed(reason: "Failed to create Metal C++ Engine context")
        }
    }

    deinit {
        if let handle = engineHandle {
            AntigravityEngineDestroy(handle)
        }
    }

    public func loadModel(at path: String) throws {
        try queue.sync {
            guard let handle = engineHandle else {
                throw AntigravityError.executionFailed(reason: "Engine context is deallocated")
            }

            let result = AntigravityEngineLoadModel(handle, path)
            guard result == 0 else {
                throw AntigravityError.modelLoadingFailed(reason: "Failed to parse/load Safetensors weights at \(path)")
            }
            self.loadedModelPath = path
        }
    }

    public func unloadWeights() {
        queue.sync {
            if let handle = engineHandle {
                AntigravityEngineUnloadWeights(handle)
            }
        }
    }

    public var hasWeights: Bool {
        queue.sync {
            guard let handle = engineHandle else { return false }
            return AntigravityEngineHasWeights(handle)
        }
    }

    public var allocatedMemoryBytes: UInt64 {
        queue.sync {
            guard let handle = engineHandle else { return 0 }
            return AntigravityEngineGetAllocatedMemoryBytes(handle)
        }
    }

    public func reason(
        promptTokens: [Int32],
        maxTokens: Int = 50,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> AntigravityGenerationResult {
        return try queue.sync {
            guard let handle = engineHandle else {
                throw AntigravityError.executionFailed(reason: "Engine context is deallocated")
            }
            guard AntigravityEngineHasWeights(handle) else {
                throw AntigravityError.executionFailed(reason: "Model weights not loaded into Metal VRAM")
            }

            let N = config.parallelChannels
            let maxNew = maxTokens

            var outTokens = [Int32](repeating: 0, count: N * maxNew)
            var outLogprobs = [Float](repeating: 0.0, count: N)
            var outTokenCounts = [Int32](repeating: 0, count: N)
            var ttftMs: Double = 0.0
            var totalMs: Double = 0.0

            let ret = promptTokens.withUnsafeBufferPointer { promptBuffer in
                outTokens.withUnsafeMutableBufferPointer { tokensBuffer in
                    outLogprobs.withUnsafeMutableBufferPointer { logprobsBuffer in
                        outTokenCounts.withUnsafeMutableBufferPointer { countsBuffer in
                            AntigravityEngineNativeGenerate(
                                handle,
                                promptBuffer.baseAddress,
                                Int32(promptTokens.count),
                                Int32(maxNew),
                                temperature,
                                topP,
                                tokensBuffer.baseAddress,
                                logprobsBuffer.baseAddress,
                                countsBuffer.baseAddress,
                                &ttftMs,
                                &totalMs
                            )
                        }
                    }
                }
            }

            guard ret == 0 else {
                throw AntigravityError.executionFailed(reason: "AntigravityEngineNativeGenerate failed with code \(ret)")
            }

            var outScores = [Float](repeating: 0.0, count: N)
            let bestIndex = outTokens.withUnsafeBufferPointer { tokensBuf -> Int32 in
                outScores.withUnsafeMutableBufferPointer { scoresBuf in
                    AntigravityEngineVerifyCandidates(
                        handle,
                        tokensBuf.baseAddress,
                        Int32(maxNew),
                        scoresBuf.baseAddress
                    )
                }
            }

            let selectedIdx = Int(bestIndex >= 0 ? bestIndex : 0)
            let bestScore = outScores[selectedIdx]
            let reflectionTriggered = bestScore < config.reflectionThreshold

            var candidateTraces: [String] = []
            for c in 0..<N {
                let count = Int(outTokenCounts[c])
                let tokens = (0..<count).map { outTokens[c * maxNew + $0] }
                candidateTraces.append("Channel \(c): " + tokens.map { String($0) }.joined(separator: " "))
            }

            let totalTokens = outTokenCounts.reduce(0) { $0 + Int($1) }
            let throughput = totalMs > 0 ? (Double(totalTokens) / (totalMs / 1000.0)) : 0.0

            return AntigravityGenerationResult(
                bestTraceText: candidateTraces[selectedIdx],
                verifierScore: bestScore,
                bestCandidateIndex: selectedIdx,
                candidateTraces: candidateTraces,
                candidatesEvaluated: N,
                reflectionTriggered: reflectionTriggered,
                timeToFirstTokenMs: ttftMs,
                totalLatencyMs: totalMs,
                totalTokensGenerated: totalTokens,
                throughputTokensPerSec: throughput
            )
        }
    }

    public func reasonMultimodal(
        textTokens: [Int32],
        imageEmbeddings: [Float],
        patchCount: Int,
        maxTokens: Int = 50,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> AntigravityGenerationResult {
        return try queue.sync {
            guard let handle = engineHandle else {
                throw AntigravityError.executionFailed(reason: "Engine context is deallocated")
            }
            guard AntigravityEngineHasWeights(handle) else {
                throw AntigravityError.executionFailed(reason: "Model weights not loaded into Metal VRAM")
            }

            let N = config.parallelChannels
            let maxNew = maxTokens

            var outTokens = [Int32](repeating: 0, count: N * maxNew)
            var outLogprobs = [Float](repeating: 0.0, count: N)
            var outTokenCounts = [Int32](repeating: 0, count: N)
            var ttftMs: Double = 0.0
            var totalMs: Double = 0.0

            let ret = textTokens.withUnsafeBufferPointer { textBuf in
                imageEmbeddings.withUnsafeBufferPointer { imgBuf in
                    outTokens.withUnsafeMutableBufferPointer { tokensBuf in
                        outLogprobs.withUnsafeMutableBufferPointer { logprobsBuf in
                            outTokenCounts.withUnsafeMutableBufferPointer { countsBuf in
                                AntigravityEngineNativeGenerateMultimodal(
                                    handle,
                                    textBuf.baseAddress,
                                    Int32(textTokens.count),
                                    imgBuf.baseAddress,
                                    Int32(patchCount),
                                    Int32(maxNew),
                                    temperature,
                                    topP,
                                    tokensBuf.baseAddress,
                                    logprobsBuf.baseAddress,
                                    countsBuf.baseAddress,
                                    &ttftMs,
                                    &totalMs
                                )
                            }
                        }
                    }
                }
            }

            guard ret == 0 else {
                throw AntigravityError.executionFailed(reason: "AntigravityEngineNativeGenerateMultimodal failed with code \(ret)")
            }

            var outScores = [Float](repeating: 0.0, count: N)
            let bestIndex = outTokens.withUnsafeBufferPointer { tokensBuf -> Int32 in
                outScores.withUnsafeMutableBufferPointer { scoresBuf in
                    AntigravityEngineVerifyCandidates(
                        handle,
                        tokensBuf.baseAddress,
                        Int32(maxNew),
                        scoresBuf.baseAddress
                    )
                }
            }

            let selectedIdx = Int(bestIndex >= 0 ? bestIndex : 0)
            let bestScore = outScores[selectedIdx]

            var candidateTraces: [String] = []
            for c in 0..<N {
                let count = Int(outTokenCounts[c])
                let tokens = (0..<count).map { outTokens[c * maxNew + $0] }
                candidateTraces.append("Channel \(c): " + tokens.map { String($0) }.joined(separator: " "))
            }

            let totalTokens = outTokenCounts.reduce(0) { $0 + Int($1) }
            let throughput = totalMs > 0 ? (Double(totalTokens) / (totalMs / 1000.0)) : 0.0

            return AntigravityGenerationResult(
                bestTraceText: candidateTraces[selectedIdx],
                verifierScore: bestScore,
                bestCandidateIndex: selectedIdx,
                candidateTraces: candidateTraces,
                candidatesEvaluated: N,
                reflectionTriggered: bestScore < config.reflectionThreshold,
                timeToFirstTokenMs: ttftMs,
                totalLatencyMs: totalMs,
                totalTokensGenerated: totalTokens,
                throughputTokensPerSec: throughput
            )
        }
    }

    public func sanitizeBuffers() {
        queue.sync {
            if let handle = engineHandle {
                AntigravityEngineSanitizeBuffers(handle)
            }
        }
    }
}
