//
//  CognitiveRouter.swift
//  MoatKit
//
//  Cognitive Router — intent classification + tiered dispatch.
//  Tier 1: tiny SLM reflex (<600 MB, <100 ms)
//  Tier 2: local reasoner + DORA parallel rollouts (N=8)
//  Tier 3: cloud fallback
//

import Foundation

public enum Tier: Int, CaseIterable, Sendable {
    case tier1Reflex = 1      // tiny SLM, single pass
    case tier2Reasoner = 2    // local reasoner + DORA + verification
    case tier3Cloud = 3       // cloud fallback
}

public struct RouteDecision: Sendable {
    public let tier: Tier
    public let reason: String
    public let model: String
    public let backend: String
    public let nChannels: Int
    public let maxTokens: Int
}

public struct RouteResult: Sendable {
    public let text: String
    public let tier: Tier
    public let model: String
    public let backend: String
    public let latencyMs: Double
    public let tokensUsed: Int
    public let estimatedWatts: Double
    public let verified: Bool
    public let candidatesEvaluated: Int
    public let reflectionTriggered: Bool
}

public enum ComplexitySignal: String, CaseIterable {
    case math = "math"
    case code = "code"
    case logic = "logic"
    case complex = "complex"
}

/// Lightweight, zero-dependency complexity classifier.
/// Scores a prompt 0.0–1.0 in sub-millisecond on a phone CPU.
public final class ComplexityClassifier: Sendable {
    private let mathKeywords: Set<String> = [
        "solve", "calculate", "compute", "equation", "formula", "derivative",
        "integral", "proof", "prove", "theorem", "algorithm", "recursive",
        "fibonacci", "prime", "factorial", "sqrt", "logarithm", "exponent",
    ]
    private let codeKeywords: Set<String> = [
        "def ", "class ", "import ", "function", "return ", "if __name__",
        "for ", "while ", "try:", "except", "lambda", "yield", "async",
    ]
    private let logicKeywords: Set<String> = [
        "because", "therefore", "implies", "contradiction", "consistent",
        "valid", "invalid", "premise", "conclusion", "deduce", "infer",
    ]
    private let complexWords: Set<String> = [
        "analyze", "compare", "evaluate", "design", "architecture", "optimize",
        "synthesize", "comprehensive", "multi-step", "reasoning", "strategy",
    ]

    public init() {}

    /// Returns a score in [0.0, 1.0]. Higher = more complex = more reasoning budget.
    public func classify(_ prompt: String) -> Float {
        let text = prompt.lowercased()
        let words = text.split(separator: " ").map { String($0) }
        let nWords = max(words.count, 1)

        var score: Float = 0.0

        // Length signal
        score += min(Float(nWords) / 60.0, 1.0) * 0.25

        // Keyword signals
        let mathHits = mathKeywords.filter { text.contains($0) }.count
        let codeHits = codeKeywords.filter { text.contains($0) }.count
        let logicHits = logicKeywords.filter { text.contains($0) }.count
        let complexHits = complexWords.filter { text.contains($0) }.count

        score += min(
            Float(mathHits) * 0.15 + Float(codeHits) * 0.15 +
            Float(logicHits) * 0.10 + Float(complexHits) * 0.10,
            0.45
        )

        // Structure signals
        if text.components(separatedBy: "?").count > 3 { score += 0.10 }
        if text.contains("step") || text.contains("step-by-step") { score += 0.05 }

        return min(score, 1.0)
    }

    public func selectTier(_ score: Float) -> Tier {
        if score <= 0.30 { return .tier1Reflex }
        if score <= 0.70 { return .tier2Reasoner }
        return .tier3Cloud
    }
}

/// Router configuration
public struct RouterConfig: Sendable {
    public var nChannels: Int = 8
    public var maxTokens: Int = 256
    public var temperature: Float = 0.7
    public var topP: Float = 0.9
    public var tier1Model: String = "llama-3.2-1b"
    public var tier2Model: String = "qwen2.5-3b"
    public var tier3Model: String = "gpt-4o-mini"
    public var backendURLs: [String: String] = [
        "llama.cpp": "http://127.0.0.1:8080",
        "mlx": "http://127.0.0.1:8081",
        "ollama": "http://127.0.0.1:11434/v1",
        "strata": "http://127.0.0.1:8082",
        "cloud": "https://api.openai.com/v1",
    ]
    public var enableVerification: Bool = true
    public var reflectionThreshold: Float = 0.75
    public var memoryBudgetBytes: Int = 2_500 * 1024 * 1024
    public var thermalBudgetWatts: Float = 5.0
    public var verbose: Bool = false

    public init() {}
}

/// The Cognitive Router — main entry point.
/// NOTE: This class is NOT Sendable because it holds mutable state
/// (backends dict, stats dict). All access is synchronized via dedicated
/// dispatch queues.
public final class CognitiveRouter {
    public let config: RouterConfig
    private let classifier = ComplexityClassifier()

    // Backends (lazy, thread-safe)
    private var backends: [String: LLMBackend] = [:]
    private let queue = DispatchQueue(label: "moat.router.backends", attributes: .concurrent)

    // Stats
    private var statsData: [String: Int] = [
        "total": 0, "tier1": 0, "tier2": 0, "tier3": 0,
        "verified": 0, "reflection": 0, "circuitBreaker": 0
    ]
    private let statsQueue = DispatchQueue(label: "moat.router.stats", attributes: .concurrent)

    // Memory & thermal guards (owned by this router)
    private let memoryGuard = MemoryGuard()
    private let thermalMonitor = ThermalMonitor()

    public init(config: RouterConfig = RouterConfig()) {
        self.config = config
    }

    public func route(
        prompt: String,
        maxTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        forceTier: Tier? = nil
    ) -> RouteResult {
        statsUpdate(key: "total", delta: 1)

        let maxTokens = maxTokens ?? config.maxTokens
        let temperature = temperature ?? config.temperature
        let topP = topP ?? config.topP

        // 1. Classify complexity
        let score = classifier.classify(prompt)
        let tier = forceTier ?? classifier.selectTier(score)

        if config.verbose {
            print("[Router] score=\(String(format: "%.2f", score)) → tier=\(tier)")
        }

        // 2. Guardrails
        try? memoryGuard.check()
        try? thermalMonitor.update(watts: 0.0)

        // 3. Dispatch
        switch tier {
        case .tier1Reflex:
            return routeTier1(prompt: prompt, maxTokens: maxTokens, temperature: temperature, topP: topP, score: score)
        case .tier2Reasoner:
            return routeTier2(prompt: prompt, maxTokens: maxTokens, temperature: temperature, topP: topP, score: score)
        case .tier3Cloud:
            return routeTier3(prompt: prompt, maxTokens: maxTokens, temperature: temperature, topP: topP, score: score)
        }
    }

    // MARK: - Tier 1: Reflex (tiny SLM, single pass)

    private func routeTier1(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        score: Float
    ) -> RouteResult {
        statsUpdate(key: "tier1", delta: 1)

        let backend = backend(for: "ollama")
        let result = backend.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )

        try? memoryGuard.check()
        try? thermalMonitor.update(watts: Float(result.estimatedWatts))

        return RouteResult(
            text: result.text,
            tier: .tier1Reflex,
            model: config.tier1Model,
            backend: "ollama",
            latencyMs: result.latencyMs,
            tokensUsed: result.tokensUsed,
            estimatedWatts: result.estimatedWatts,
            verified: false,
            candidatesEvaluated: 1,
            reflectionTriggered: false
        )
    }

    // MARK: - Tier 2: Reasoner (DORA parallel rollouts + verification)

    private func routeTier2(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        score: Float
    ) -> RouteResult {
        statsUpdate(key: "tier2", delta: 1)

        let backend = backend(for: "mlx")
        let result = try await doraRollout(
            backend: backend,
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )

        try? memoryGuard.check()
        try? thermalMonitor.update(watts: Float(result.estimatedWatts))

        return result
    }

    private func doraRollout(
        backend: LLMBackend,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ) async -> RouteResult {
        let nChannels = config.nChannels

        // Launch N parallel rollouts
        let tasks = (0..<nChannels).map { idx in
            Task {
                let t0 = Date()
                do {
                    let result = try await backend.generate(
                        prompt: prompt,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP
                    )
                    return (
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
                    return (
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

        var rolloutResults: [(index: Int, text: String, tokens: [String], logprob: Double, latencyMs: Double, backend: String, model: String, error: String?)] = []
        for task in tasks {
            if let r = try? await task.value, !r.text.isEmpty {
                rolloutResults.append(r)
            }
        }

        guard !rolloutResults.isEmpty else {
            throw BackendError.generationFailed("All DORA rollout channels failed")
        }

        // DORA uniqueness re-weighting
        let traces = rolloutResults.map { $0.text }
        let uniqueness = doraUniquenessWeights(traces: traces)

        // List-wise verifier scoring (heuristic)
        let logprobs = rolloutResults.map { $0.logprob }
        let verifierScores = listWiseScore(traces: traces, logprobs: logprobs)

        // Combined: verifier * uniqueness
        var combined: [Float] = []
        for i in 0..<traces.count {
            combined.append(verifierScores[i] * Float(uniqueness[i]))
        }

        let bestIdx = combined.enumerated().max(by: { $0.element < $1.element })!.offset
        let bestTrace = traces[bestIdx]

        // Verification + self-healing
        var verified = false
        var verificationResult: VerificationResult?
        var reflectionTriggered = false

        if config.enableVerification {
            let statedAnswer = extractStatedAnswer(bestTrace)
            let expression = extractMathExpression(bestTrace)

            let astVerifier = ASTVerifier()
            let z3Verifier = Z3Verifier()

            let healingResult = selfHealingLoop(
                dispatch: { p in
                    let r = try! backend.generate(
                        prompt: p,
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP
                    )
                    return r.text
                },
                verifiers: [astVerifier, z3Verifier],
                initialPrompt: prompt,
                context: ["expression": expression ?? "", "statedAnswer": statedAnswer ?? ""]
            )

            if let last = healingResult.attempts.last {
                verificationResult = last.result
                verified = last.result.passed
                if healingResult.attempts.count > 1 {
                    reflectionTriggered = true
                    statsUpdate(key: "reflection", delta: 1)
                }
            }

            if verified {
                self._statsUpdate(key: "verified", delta: 1)
                self._statsUpdate(key: "circuitBreaker", delta: 1)
            }
        }

        let totalTokens = traces.reduce(0) { $0 + $1.split(separator: " ").count }
        let latencyMs = Date().timeIntervalSince(Date()) * 1000.0 // placeholder
        let estWatts = Double(totalTokens) * 0.05

        return RouteResult(
            text: bestTrace,
            tier: .tier2Reasoner,
            model: config.tier2Model,
            backend: backend.backendName,
            latencyMs: latencyMs,
            tokensUsed: totalTokens,
            estimatedWatts: estWatts,
            verified: verified,
            candidatesEvaluated: traces.count,
            reflectionTriggered: reflectionTriggered
        )
    }

    // MARK: - Tier 3: Cloud fallback

    private func routeTier3(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        score: Float
    ) -> RouteResult {
        statsUpdate(key: "tier3", delta: 1)

        let backend: LLMBackend
        do {
            backend = try self.backend(for: "cloud")
        } catch {
            // Fall back to Tier 2 if cloud unavailable
            return routeTier2(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                score: score
            )
        }

        let result = backend.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )

        try? memoryGuard.check()
        try? thermalMonitor.update(watts: Float(result.estimatedWatts))

        return RouteResult(
            text: result.text,
            tier: .tier3Cloud,
            model: config.tier3Model,
            backend: "cloud",
            latencyMs: result.latencyMs,
            tokensUsed: result.tokensUsed,
            estimatedWatts: result.estimatedWatts,
            verified: false,
            candidatesEvaluated: 1,
            reflectionTriggered: false
        )
    }

    // MARK: - Backend management

    private func backend(for name: String) -> LLMBackend {
        queue.sync {
            if let existing = backends[name] { return existing }
            let b: LLMBackend
            switch name {
            case "ollama":
                b = HTTPBackend(baseURL: config.backendURLs["ollama"] ?? "http://127.0.0.1:11434/v1",
                                model: config.tier1Model,
                                backendName: "ollama")
            case "strata":
                b = HTTPBackend(baseURL: config.backendURLs["strata"] ?? "http://127.0.0.1:8082",
                                model: config.tier2Model,
                                backendName: "strata")
            case "mlx":
                do {
                    b = try MLXBackend(model: config.tier2Model)
                } catch {
                    b = HTTPBackend(baseURL: config.backendURLs["mlx"] ?? "http://127.0.0.1:8081",
                                    model: config.tier2Model,
                                    backendName: "mlx")
                }
            case "cloud":
                b = HTTPBackend(baseURL: config.backendURLs["cloud"] ?? "https://api.openai.com/v1",
                                model: config.tier3Model,
                                backendName: "cloud")
            default:
                b = HTTPBackend(baseURL: config.backendURLs[name] ?? "http://127.0.0.1:8080",
                                model: config.tier2Model,
                                backendName: name)
            }
            backends[name] = b
            return b
        }
    }

    // MARK: - Helpers

    private func doraUniquenessWeights(traces: [String]) -> [Float] {
        // Simple lexical diversity: Jaccard similarity on word bigrams
        let sets = traces.map { Set($0.lowercased().split(separator: " ").map { String($0) }) }
        var weights: [Float] = Array(repeating: 1.0, count: traces.count)

        for i in 0..<traces.count {
            var similaritySum: Float = 0.0
            for j in 0..<traces.count where i != j {
                let intersection = Float(sets[i].intersection(sets[j]).count)
                let union = Float(sets[i].union(sets[j]).count)
                similaritySum += union > 0 ? intersection / union : 0.0
            }
            weights[i] = 1.0 / max(similaritySum, 0.01)
        }

        // Normalize so mean weight is 1.0
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
        // Softmax
        let maxScore = scores.max() ?? 0.0
        let expScores = scores.map { exp($0 - maxScore) }
        let sumExp = expScores.reduce(0, +)
        return expScores.map { $0 / sumExp }
    }

    private func _looksComplete(_ text: String) -> Bool {
        guard !text.isEmpty, text.count >= 5 else { return false }
        let stripped = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if ".!?".contains(stripped.last ?? " ") { return true }
        if stripped.contains("####") || stripped.lowercased().contains("answer:") {
            return true
        }
        return false
    }

    private func _extractStatedAnswer(_ trace: String) -> String? {
        let pattern = #"####\s*(-?\d+(?:\.\d+)?)"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let m = regex.firstMatch(in: trace, range: NSRange(trace.startIndex..<trace.endIndex, in: trace)),
              let range = Range(m.range(at: 1), in: trace) else {
            return nil
        }
        return String(trace[range])
    }

    private func _extractMathExpression(_ trace: String) -> String? {
        guard let pattern = try? NSRegularExpression(pattern: #"([\d\s\+\-\*\/\(\)\.]+)"#),
              let m = pattern.firstMatch(in: trace, range: NSRange(trace.startIndex..<trace.endIndex, in: trace)),
              let range = Range(m.range(at: 1), in: trace) else {
            return nil
        }
        return String(trace[range]).trimmingCharacters(in: .whitespaces)
    }

    private func statsUpdate(key: String, delta: Int) {
        statsQueue.async {
            self.statsData[key, default: 0] += delta
        }
    }

    private func _statsUpdate(key: String, delta: Int) {
        statsQueue.async {
            self.statsData[key, default: 0] += delta
        }
    }

    public func stats() -> [String: Int] {
        return statsQueue.sync { statsData }
    }

    public func memorySnapshot() -> [String: Any] {
        return memoryGuard.snapshot()
    }

    public func thermalSnapshot() -> [String: Any] {
        let snap = thermalMonitor.update(watts: 0.0)
        return [
            "watts": snap.watts,
            "overBudget": snap.overBudget,
            "violations": thermalMonitor.violationCount
        ]
    }
}

// MARK: - MemoryGuard (simplified inline)

public final class MemoryGuard {
    public let budgetBytes: Int
    public let warnBytes: Int
    private let lock = NSLock()
    private var peakRSS: Int = 0
    private var violations: Int = 0

    public init(budgetBytes: Int = 2_500 * 1024 * 1024,
                warnBytes: Int = 2_000 * 1024 * 1024) {
        self.budgetBytes = budgetBytes
        self.warnBytes = warnBytes
    }

    public func check() throws {
        let rss = Self.readRSS()
        lock.lock()
        if rss > peakRSS { peakRSS = rss }
        let overBudget = rss > budgetBytes
        lock.unlock()
        if overBudget {
            violations += 1
            throw MemoryBudgetExceeded(
                message: "RSS \(rss / 1_048_576) MB exceeds \(budgetBytes / 1_048_576) MB budget"
            )
        }
    }

    public var peakRSSMB: Double { Double(peakRSS) / 1_048_576.0 }
    public var violationCount: Int { violations }

    private static func readRSS() -> Int {
        #if os(macOS)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                          task_flavor_t(MACH_TASK_BASIC_INFO),
                          $0,
                          &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.resident_size)
        #else
        return 0
        #endif
    }
}

public struct MemorySnapshot: Sendable {
    public let rssBytes: Int
    public let rssMB: Double
    public let overBudget: Bool
    public let timestamp: Date
}

extension MemoryGuard {
    public func snapshot() -> [String: Any] {
        lock.lock()
        let rss = Self.readRSS()
        let over = rss > budgetBytes
        lock.unlock()
        return [
            "rssMB": Double(rss) / 1_048_576.0,
            "overBudget": over,
            "peakRSSMB": peakRSSMB,
            "violations": violations
        ]
    }
}

public final class ThermalMonitor {
    public let budgetWatts: Float
    private let lock = NSLock()
    private var ewmaWatts: Float?
    private var violations: Int = 0

    public init(budgetWatts: Float = 5.0) {
        self.budgetWatts = budgetWatts
    }

    public func update(watts: Float) -> ThermalSnapshot {
        lock.lock()
        if ewmaWatts == nil {
            ewmaWatts = watts
        } else {
            ewmaWatts = 0.3 * watts + 0.7 * ewmaWatts!
        }
        let over = ewmaWatts! > budgetWatts
        if over { violations += 1 }
        let snap = ThermalSnapshot(watts: ewmaWatts!, overBudget: over, timestamp: Date())
        lock.unlock()
        return snap
    }

    public var currentWatts: Float? { ewmaWatts }
    public var violationCount: Int { violations }
}

public struct ThermalSnapshot: Sendable {
    public let watts: Float
    public let overBudget: Bool
    public let timestamp: Date
}

public enum MemoryBudgetExceeded: Error, Sendable {
    case exceeded(message: String)
}

extension MemoryBudgetExceeded: CustomStringConvertible {
    public var description: String {
        switch self {
        case .exceeded(let msg): return msg
        }
    }
    public init(message: String) {
        self = .exceeded(message: message)
    }
}