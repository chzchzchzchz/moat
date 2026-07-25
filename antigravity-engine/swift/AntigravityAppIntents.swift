//
// Project Antigravity — Native App Intents Integration (iOS 16+ / macOS 13+)
// Direct system integration for Siri, Spotlight, and hardware Action Button
// Zero serialization overhead directly into local runtime memory.
//

import Foundation
import AppIntents

/// App Intent for Siri, Spotlight, and Action Button triggers
@available(iOS 16.0, macOS 13.0, *)
public struct AntigravityReasoningIntent: AppIntent {
    public static var title: LocalizedStringResource = "Verify Offline Logic & Reasoning"
    public static var description = IntentDescription("Executes N=8 parallel offline reasoning rollouts using Antigravity local inference engine.")

    @Parameter(title: "Prompt Text", description: "User query or text payload to analyze")
    public var prompt: String

    @Parameter(title: "Parallel Channels", default: 8)
    public var parallelChannels: Int

    public init() {
        self.prompt = ""
        self.parallelChannels = 8
    }

    public init(prompt: String, parallelChannels: Int = 8) {
        self.prompt = prompt
        self.parallelChannels = parallelChannels
    }

    public func perform() async throws -> some IntentResult & ReturnsValue<String> {
        // Instantiate zero-config engine wrapper
        let config = EngineConfig(parallelChannels: parallelChannels)
        let engine = try AntigravityEngine(config: config)

        // Execute local offline Best-of-N query
        let result = try await engine.generate(prompt: prompt)

        return .result(value: result.bestTraceText)
    }
}

/// Intent Query Provider for Spotlight Search & Siri Suggestions
@available(iOS 16.0, macOS 13.0, *)
public struct AntigravityIntentQuery: EntityQuery {
    public init() {}

    public func entities(for identifiers: [String]) async throws -> [String] {
        return identifiers
    }
}
