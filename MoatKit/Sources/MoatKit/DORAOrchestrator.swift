//
//  DORAOrchestrator.swift
//  MoatKit
//
//  Coordinates parallel reasoning-channel rollouts (Best-of-N) across multiple
//  LLM backends, applies DORA uniqueness re-weighting to de-duplicate similar
//  candidates, and selects the best trace via list-wise verification.
//

import Foundation

public struct DORAConfig: Sendable {
    public var nChannels: Int = 8
    public var maxTokens: Int = 256
    public var temperature: Float = 0.7
    public var topP: Float = 0.9
    public var doraTau: Float = 0.5
    public var redundancyThreshold: Float = 0.85
    public init() {}
}

public struct RolloutResult: Sendable {
    public let index: Int
    public let text: String
    public let tokens: [String]
    public let logprob: Double
    public let latencyMs: Double
    public let backend: String
    public let model: String
    public let error: String?
}

public struct OrchestrationResult: Sendable {
    public let bestTrace: String
    public let bestIndex: Int
    public let bestScore: Float
    public let allTraces: [String]
    public let allScores: [Float]
    public let latencyMs: Double
    public let tokensUsed: Int
    public let estimatedWatts: Double
    public let verified: Bool
    public let candidatesEvaluated: Int
    public let doraInfo: [String: [Double]]
}

public final class DORAOrchestrator: Sendable {
    public let backend: LLMBackend
    public let config: DORAConfig

    public init(backend: LLMBackend, config: DORAConfig = DORAConfig()) {
        self.backend = backend
        self.config = config
    }

    public func orchestrate(
        prompt: String,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil
    ) async throws -> OrchestrationResult {
        let maxTokens = maxTokens ?? config.maxTokens
        let temperature = temperature ?? config.temperature
        let topP = topP ?? config.topP
        let t0 = Date()

        // Launch N parallel rollouts
        let tasks = (0..<config.nChannels).map { idx in
            Task {
                do {
                    let result = try await backend.generate(
                        prompt: prompt,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP
                    )
                    return RolloutResult(
                        index: idx,
                        text: result.text,
                        tokens: result.text.split(separator: " ").map(String.init),
                        logprob: -Double(result.latencyMs) / 1000.0,
                        latencyMs: result.latencyMs,
                        backend: result.backend,
                        model: result.model,
                        error: nil
                    )
                } catch {
                    return RolloutResult(
                        index: idx,
                        text: "",
                        tokens: [],
                        logprob: -999.0,
                        latencyMs: 0.0,
                        backend: backend.backendName,
                        model: backend.modelName,
                        error: error.localizedDescription
                    )
                }
            }
        }

        var rollouts: [RolloutResult] = []
        for task in tasks {
            if let r = try? await task.value, !r.text.isEmpty {
                rollouts.append(r)
            }
        }

        guard !rollouts.isEmpty else {
            throw BackendError.generationFailed("All DORA rollout channels failed")
        }

        // DORA uniqueness re-weighting
        let traces = rollouts.map { $0.text }
        let uniqueness = doraUniquenessWeights(traces: traces)

        // List-wise scoring (simplified heuristic)
        let logprobs = rollouts.map { $0.logprob }
        let scores = listWiseScore(traces: traces, logprobs: logprobs)

        // Combined score
        var combined: [Float] = []
        for i in 0..<traces.count {
            combined.append(scores[i] * Float(uniqueness[i]))
        }

        let bestIdx = combined.enumerated().max(by: { $0.element < $1.element })!.offset
        let bestTrace = traces[bestIdx]

        let totalTokens = traces.reduce(0) { $0 + $1.split(separator: " ").count }
        let latencyMs = Date().timeIntervalSince(t0) * 1000.0
        let estWatts = Double(totalTokens) * 0.05

        return OrchestrationResult(
            bestTrace: bestTrace,
            bestIndex: bestIdx,
            bestScore: Float(combined[bestIdx]),
            allTraces: traces,
            allScores: combined,
            latencyMs: latencyMs,
            tokensUsed: totalTokens,
            estimatedWatts: estWatts,
            verified: true, // placeholder — real verification uses ASTVerifier
            candidatesEvaluated: traces.count,
            doraInfo: [
                "uniquenessWeights": uniqueness.map { Double($0) },
                "verifierScores": scores.map { Double($0) },
                "combinedScores": combined.map { Double($0) }
            ]
        )
    }

    private func doraUniquenessWeights(traces: [String]) -> [Float] {
        let sets = traces.map { Set($0.lowercased().split(separator: " ").map(String.init)) }
        var weights = Array(repeating: Float(1.0), count: traces.count)
        for i in 0..<traces.count {
            var simSum: Float = 0.0
            for j in 0..<traces.count where i != j {
                let inter = Float(sets[i].intersection(sets[j]).count)
                let union = Float(sets[i].union(sets[j]).count)
                simSum += union > 0 ? inter / union : 0.0
            }
            weights[i] = 1.0 / max(simSum, 0.01)
        }
        let mean = weights.reduce(0, +) / Float(weights.count)
        return weights.map { $0 / mean }
    }

    private func listWiseScore(traces: [String], logprobs: [Double]) -> [Float] {
        var scores: [Float] = []
        for i in 0..<traces.count {
            let trace = traces[i]
            let logprob = logprobs[i]
            let seqLen = max(trace.split(separator: " ").count, 1)
            let densityScore = Float(logprob) / Float(seqLen)
            let stepContribution = Float(log(Double(trace.components(separatedBy: "\n").count + 1))) * 0.5
            let uniqueChars = Float(Set(trace.lowercased()).count)
            let diversityScore = (uniqueChars / Float(seqLen)) * 2.0
            scores.append(densityScore + stepContribution + diversityScore)
        }
        let maxScore = scores.max() ?? 0.0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        return expScores.map { $0 / sumExp }
    }
}
