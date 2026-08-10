//
// Project Antigravity — Public Swift API Wrapper
// Universal Framework Binding for iOS / macOS Apple Silicon Engine
//

import Foundation
import CAntigravityEngine

/// Public configuration for the Antigravity local inference engine.
public struct EngineConfig {
    /// Maximum allowed physical RAM in bytes (default: 4.5GB iOS app entitlement ceiling)
    public let memoryLimitBytes: Int64
    /// Number of parallel reasoning traces to rollout (N=4, 8, 16)
    public let parallelChannels: Int
    /// Reflection threshold tau for adaptive verification (default: 0.75)
    public let reflectionThreshold: Float
    /// Path to precomputed exponential LUT
    public let enableLUTAcceleration: Bool

    public static let strict4GBFootprint = EngineConfig(
        memoryLimitBytes: 4500 * 1024 * 1024,
        parallelChannels: 8,
        reflectionThreshold: 0.75,
        enableLUTAcceleration: true
    )

    public init(
        memoryLimitBytes: Int64 = 4500 * 1024 * 1024,
        parallelChannels: Int = 8,
        reflectionThreshold: Float = 0.75,
        enableLUTAcceleration: Bool = true
    ) {
        self.memoryLimitBytes = memoryLimitBytes
        self.parallelChannels = parallelChannels
        self.reflectionThreshold = reflectionThreshold
        self.enableLUTAcceleration = enableLUTAcceleration
    }
}

/// Generation Result returned by the Antigravity Engine
public struct AntigravityGenerationResult {
    public let bestTraceText: String
    public let verifierScore: Float
    public let candidatesEvaluated: Int
    public let reflectionTriggered: Bool
    public let tokenSavingsPercentage: Float
    public let latencyMilliseconds: Double
    public let totalTokensGenerated: Int
}

/// Error types for Antigravity Engine
public enum AntigravityError: Error {
    case memoryBudgetExceeded(requestedBytes: Int64, ceilingBytes: Int64)
    case modelLoadingFailed(reason: String)
    case executionFailed(reason: String)
}

/// Primary public entry point for Antigravity Engine
public final class AntigravityEngine {
    private var engineHandle: antigravity_engine_t?
    private let config: EngineConfig

    /// Initialize the engine with specified configuration and model path
    public init(modelPath: String = "models/model.gguf", config: EngineConfig = .strict4GBFootprint) throws {
        self.config = config
        var cConfig = antigravity_config_t(
            memory_limit_bytes: config.memoryLimitBytes,
            parallel_channels: UInt32(config.parallelChannels),
            reflection_threshold: config.reflectionThreshold,
            enable_lut: config.enableLUTAcceleration
        )
        self.engineHandle = antigravity_engine_create(&cConfig, modelPath)
        guard self.engineHandle != nil else {
            throw AntigravityError.executionFailed(reason: "Failed to initialize Metal C++ Engine")
        }
    }

    deinit {
        if let handle = engineHandle {
            antigravity_engine_destroy(handle)
        }
    }

    /// Run an offline parallel Best-of-N reasoning query
    public func generateRollouts(
        prompt: String,
        maxTokens: Int = 50,
        temperature: Float = 0.7
    ) async throws -> AntigravityGenerationResult {
        guard let handle = engineHandle else {
            throw AntigravityError.executionFailed(reason: "Engine handle deallocated")
        }

        let resPtr = antigravity_generate_rollouts(handle, prompt, UInt32(maxTokens), temperature)
        guard let res = resPtr?.pointee else {
            throw AntigravityError.executionFailed(reason: "Rollout generation returned NULL pointer")
        }

        defer { antigravity_free_rollout_result(resPtr) }

        let vresPtr = antigravity_verify_candidates(handle, resPtr)
        guard let vres = vresPtr?.pointee else {
            throw AntigravityError.executionFailed(reason: "Candidate verification returned NULL pointer")
        }

        defer { antigravity_free_verification_result(vresPtr) }

        let bestIdx = Int(res.best_candidate_index)
        guard bestIdx < Int(res.candidate_count), let traceCStr = res.candidates[bestIdx].trace_text else {
            throw AntigravityError.executionFailed(reason: "Invalid candidate trace output from engine")
        }

        let bestTrace = String(cString: traceCStr)
        let score = vres.confidence_score

        return AntigravityGenerationResult(
            bestTraceText: bestTrace,
            verifierScore: score,
            candidatesEvaluated: Int(res.candidate_count),
            reflectionTriggered: res.reflection_triggered,
            tokenSavingsPercentage: res.token_savings_pct,
            latencyMilliseconds: res.total_latency_ms,
            totalTokensGenerated: Int(res.candidate_count) * maxTokens
        )
    }

    /// Zero out all internal Metal buffers for Secure Enclave compliance
    public func sanitizeBuffers() {
        if let handle = engineHandle {
            antigravity_sanitize_buffers(handle)
        }
    }
}
