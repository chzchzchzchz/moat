//
// Project Antigravity — Step-Level Reflection Controller for Tool-Calling Agents
// Implements PRM-gated tool-calling reflection thresholds (tau = 0.75)
//

import Foundation

/// Represents a tool definition available to the Agent
public struct AgentTool: Sendable {
    public let name: String
    public let description: String
    public let jsonSchema: String

    public init(name: String, description: String, jsonSchema: String) {
        self.name = name
        self.description = description
        self.jsonSchema = jsonSchema
    }
}

/// Result of an agent step execution
public struct AgentStepResult: Sendable {
    public let selectedTrace: String
    public let verifierScore: Float
    public let reflectionTriggered: Bool
    public let executedToolCall: String?
    public let tokensUsed: Int
    public let latencyMs: Double
}

/// Agent Controller for tool calling with reflection thresholds
public final class AgentController: @unchecked Sendable {
    public let engine: AntigravityEngine
    public let reflectionThreshold: Float

    public init(engine: AntigravityEngine, reflectionThreshold: Float = 0.75) {
        self.engine = engine
        self.reflectionThreshold = reflectionThreshold
    }

    /// Execute a tool-calling reasoning step with PRM threshold reflection
    public func executeStep(
        promptTokens: [Int32],
        availableTools: [AgentTool],
        maxTokens: Int = 60
    ) async throws -> AgentStepResult {
        // Run parallel Best-of-N reasoning query through AntigravityEngine
        let result = try await engine.reason(
            promptTokens: promptTokens,
            maxTokens: maxTokens,
            temperature: 0.7,
            topP: 0.9
        )

        // Parse tool call if present in best trace
        var toolCall: String? = nil
        if result.bestTraceText.contains("tool_call") || result.bestTraceText.contains("Action:") {
            toolCall = result.bestTraceText
        }

        return AgentStepResult(
            selectedTrace: result.bestTraceText,
            verifierScore: result.verifierScore,
            reflectionTriggered: result.reflectionTriggered,
            executedToolCall: toolCall,
            tokensUsed: result.totalTokensGenerated,
            latencyMs: result.totalLatencyMs
        )
    }
}
