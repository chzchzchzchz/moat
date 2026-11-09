//
//  MoatKitDemoApp.swift
//  MoatKitDemo
//
//  Cognitive Router demo app showing the 3-tier architecture.
//  Demonstrates: Tier 1 reflex, Tier 2 reasoner, Tier 3 cloud fallback.
//

import SwiftUI
import MoatKit

@main
struct MoatKitDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - Demo View

struct ContentView: View {
    @State private var input = ""
    @State private var results: [DemoResult] = []
    @State private var isProcessing = false
    @State private var showingStats = false
    @State private var router: CognitiveRouter?

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // Header
                VStack(alignment: .leading, spacing: 8) {
                    Text("Moat Cognitive Router")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    Text("Apple Silicon + iOS Cognitive Router")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()

                // Input
                VStack(alignment: .leading, spacing: 8) {
                    Text("Enter a query:")
                        .font(.headline)
                    TextField("Type your question here...", text: $input)
                        .textFieldStyle(.roundedBorder)
                        .submitLabel(.send)
                        .onSubmit {
                            Task { await runQuery() }
                        }
                }
                .padding(.horizontal)

                // Tier selector
                VStack(alignment: .leading, spacing: 8) {
                    Text("Force tier (optional):")
                        .font(.headline)
                    Picker("Tier", selection: $selectedTier) {
                        Text("Auto").tag(TierOption.auto)
                        Text("Tier 1 Reflex").tag(TierOption.tier1)
                        Text("Tier 2 Reasoner").tag(TierOption.tier2)
                        Text("Tier 3 Cloud").tag(TierOption.tier3)
                    }
                    .pickerStyle(.segmented)
                }
                .padding(.horizontal)

                // Button
                Button(action: { Task { await runQuery() } }) {
                    HStack {
                        if isProcessing {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isProcessing ? "Processing..." : "Run Query")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(isProcessing)
                .padding(.horizontal)

                // Results
                if !results.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Results")
                            .font(.headline)
                            .frame(maxWidth: .infinity, alignment: .leading)

                        ForEach(results) { result in
                            ResultCard(result: result)
                        }
                    }
                    .padding(.horizontal)
                }

                // Stats
                if let stats = router?.stats() {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Total: \(stats["total"] ?? 0)")
                            Spacer()
                            Text("T1: \(stats["tier1"] ?? 0)")
                            Spacer()
                            Text("T2: \(stats["tier2"] ?? 0)")
                            Spacer()
                            Text("T3: \(stats["tier3"] ?? 0)")
                        }
                        HStack {
                            Text("Verified: \(stats["verified"] ?? 0)")
                            Spacer()
                            Text("Reflection: \(stats["reflection"] ?? 0)")
                            Spacer()
                            Text("CB Hits: \(stats["circuitBreaker"] ?? 0)")
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                    }
                    .padding(.horizontal)
                }

                Spacer()
            }
        }
    }

    @State private var selectedTier: TierOption = .auto

    private func runQuery() async {
        isProcessing = true

        let router = CognitiveRouter()
        let result = route_sync(
            prompt: input,
            maxTokens: 256,
            temperature: 0.7,
            topP: 0.9,
            forceTier: selectedTier.tier,
            router: router
        )

        let resultItem = DemoResult(
            text: result.text,
            tier: result.tier,
            model: result.model,
            latencyMs: result.latencyMs,
            tokensUsed: result.tokensUsed,
            estimatedWatts: result.estimatedWatts,
            verified: result.verified
        )

        results.append(resultItem)
        isProcessing = false
    }
}

struct ResultCard: View {
    let result: DemoResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(result.tier.rawValue.capitalized)
                    .font(.caption)
                    .fontWeight(.bold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(tierColor(result.tier).opacity(0.2))
                    .foregroundColor(tierColor(result.tier))
                    .cornerRadius(6)

                Spacer()

                Text("\(result.latencyMs, specifier: "%.0f") ms")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(result.text)
                .font(.body)
                .foregroundColor(.primary)

            HStack {
                Text("\(result.tokensUsed) tokens")
                Text("•")
                Text("\(result.estimatedWatts, specifier: "%.2f")W")
                Text("•")
                Text(result.verified ? "✓ Verified" : "Unverified")
            }
            .font(.caption2)
            .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }

    private func tierColor(_ tier: Tier) -> Color {
        switch tier {
        case .tier1Reflex: return .green
        case .tier2Reasoner: return .blue
        case .tier3Cloud: return .orange
        }
    }
}

struct DemoResult: Identifiable {
    let id = UUID()
    let text: String
    let tier: Tier
    let model: String
    let latencyMs: Double
    let tokensUsed: Int
    let estimatedWatts: Double
    let verified: Bool
}

enum TierOption: String, CaseIterable {
    case auto = "auto"
    case tier1 = "tier1"
    case tier2 = "tier2"
    case tier3 = "tier3"

    var tier: Tier? {
        switch self {
        case .auto: return nil
        case .tier1: return .tier1Reflex
        case .tier2: return .tier2Reasoner
        case .tier3: return .tier3Cloud
        }
    }
}