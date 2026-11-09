//
//  BackendProtocol.swift
//  MoatKit
//
//  Abstracts LLM inference backends so MoatKit never writes custom Metal shaders.
//  Backends: MLX (Apple Silicon), llama.cpp (C API), Ollama/Strata (HTTP).
//

import Foundation

public enum BackendError: Error {
    case unavailable(String)
    case generationFailed(String)
    case modelNotLoaded
}

public struct GenerationResult: Sendable {
    public let text: String
    public let tokensUsed: Int
    public let latencyMs: Double
    public let estimatedWatts: Double
    public let backend: String
    public let model: String
    public let finishReason: String
}

/// Abstract backend for LLM inference.
public protocol LLMBackend: Sendable {
    var backendName: String { get }
    var modelName: String { get }
    func isAvailable() async -> Bool
    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ) async throws -> GenerationResult
}

/// MLX backend — Apple's on-device ML framework.
/// NOTE: This is a stub placeholder. Full MLX integration requires the exact
/// MLX Swift LM API which is under active development. The router architecture
/// works with any LLMBackend implementation.
public final class MLXBackend: LLMBackend, @unchecked Sendable {
    public let backendName = "mlx"
    public let modelName: String
    private var isLoaded = false

    public init(model: String = "mlx-community/Qwen3-4B-4bit") {
        self.modelName = model
        // MLX model loading is deferred to first generate() call
        // to avoid blocking init. The actual model loading uses MLX LM's
        // Swift API which is still being stabilized.
        self.isLoaded = true // stub: assume available
    }

    public func isAvailable() async -> Bool {
        return isLoaded
    }

    public func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ) async throws -> GenerationResult {
        // Stub: return a placeholder response. Replace with real MLX inference.
        let responseText = "[MLX backend: \(modelName) - stub response for '\(prompt.prefix(30))'...]"
        let latencyMs: Double = 50.0
        let tokensUsed = max(responseText.split(separator: " ").count, 1)
        let estimatedWatts = Double(tokensUsed) * 0.05
        return GenerationResult(
            text: responseText,
            tokensUsed: tokensUsed,
            latencyMs: latencyMs,
            estimatedWatts: estimatedWatts,
            backend: backendName,
            model: modelName,
            finishReason: "stop"
        )
    }
}

/// OpenAI-compatible HTTP backend (llama.cpp server, mlx-lm.server, Strata, Ollama).
public final class HTTPBackend: LLMBackend, @unchecked Sendable {
    public let backendName: String
    public let modelName: String
    private let baseURL: URL
    private let session = URLSession(configuration: .default)

    public init(baseURL: String, model: String, backendName: String) {
        self.baseURL = URL(string: baseURL.rstrip("/"))!
        self.modelName = model
        self.backendName = backendName
    }

    public func isAvailable() async -> Bool {
        var req = URLRequest(url: baseURL.appendingPathComponent("/v1/models"))
        req.timeoutInterval = 3.0
        do {
            let (_, resp) = try await session.data(for: req)
            return (resp as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    public func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ) async throws -> GenerationResult {
        let payload: [String: Any] = [
            "model": modelName,
            "messages": [["role": "user", "content": prompt]],
            "max_tokens": maxTokens,
            "temperature": temperature,
            "top_p": topP,
            "stream": false
        ]
        var req = URLRequest(url: baseURL.appendingPathComponent("/v1/chat/completions"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: payload)
        req.timeoutInterval = 30.0

        let t0 = Date()
        let (data, _) = try await session.data(for: req)
        let latencyMs = Date().timeIntervalSince(t0) * 1000.0

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let choice = (json["choices"] as! [[String: Any]])[0]
        let message = choice["message"] as! [String: Any]
        let text = message["content"] as! String
        let usage = json["usage"] as? [String: Any] ?? [:]
        let tokensUsed = (usage["completion_tokens"] as? Int) ?? text.split(separator: " ").count

        return GenerationResult(
            text: text,
            tokensUsed: tokensUsed,
            latencyMs: latencyMs,
            estimatedWatts: Double(tokensUsed) * 0.05,
            backend: backendName,
            model: modelName,
            finishReason: (choice["finish_reason"] as? String) ?? "stop"
        )
    }
}

private extension String {
    func rstrip(_ char: Character = "/") -> String {
        return String(self.reversed().drop(while: { $0 == char }).reversed())
    }
}
