//
// Project Antigravity — Developer Kit Single-Page SwiftUI Demo App
// Real-time camera/image visual problem input, N=8 parallel rollout execution view,
// verifier scoring metrics, and step-level reflection state display.
//

import SwiftUI

public struct ContentView: View {
    @StateObject private var viewModel = DemoViewModel()

    public init() {}

    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header Banner
                    headerBanner

                    // Model Weight Download Card
                    weightDownloadCard

                    // Visual Input / Camera Section
                    visualInputCard

                    // Prompt Input Section
                    promptInputCard

                    // Execution Controls
                    executionButton

                    // Live Rollout & Reflection Output View
                    if viewModel.isRunningReasoning {
                        progressView
                    }

                    if !viewModel.bestTraceOutput.isEmpty {
                        resultsView
                    }
                }
                .padding()
            }
            .navigationTitle("Antigravity Engine")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: - Subviews

    private var headerBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "cpu.fill")
                    .font(.title2)
                    .foregroundColor(.blue)
                Text("Apple Silicon Edge Engine")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                Text("N=8 Channels")
                    .font(.caption)
                    .bold()
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.blue.opacity(0.15))
                    .cornerRadius(6)
            }
            Text("On-device parallel reasoning with Metal GPU acceleration & CoreML ANE vision encoding.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(uiColor: .secondarySystemBackground))
        .cornerRadius(12)
    }

    private var weightDownloadCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: viewModel.isModelCached ? "checkmark.circle.fill" : "arrow.down.circle")
                    .foregroundColor(viewModel.isModelCached ? .green : .orange)
                Text(viewModel.isModelCached ? "TinyLlama-1.1B Weights Cached" : "Model Weights Required (~1.1 GB)")
                    .font(.subheadline)
                    .bold()
                Spacer()
                if !viewModel.isModelCached && !viewModel.isDownloading {
                    Button("Download") {
                        Task {
                            await viewModel.downloadWeights()
                        }
                    }
                    .font(.caption)
                    .buttonStyle(.borderedProminent)
                }
            }

            if viewModel.isDownloading {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: viewModel.downloadProgressFraction)
                    HStack {
                        Text("\(Int(viewModel.downloadProgressFraction * 100))%")
                            .font(.caption2)
                        Spacer()
                        Text(viewModel.downloadSpeedText)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding()
        .background(Color(uiColor: .tertiarySystemBackground))
        .cornerRadius(10)
    }

    private var visualInputCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Visual Problem Input (CoreML Vision)")
                .font(.subheadline)
                .bold()

            HStack(spacing: 15) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.gray.opacity(0.2))
                        .frame(width: 80, height: 80)

                    if viewModel.selectedImage != nil {
                        Image(systemName: "doc.richtext.fill")
                            .font(.largeTitle)
                            .foregroundColor(.blue)
                    } else {
                        Image(systemName: "camera.fill")
                            .font(.title)
                            .foregroundColor(.gray)
                    }
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(viewModel.selectedImage == nil ? "No image attached" : "Math problem image attached")
                        .font(.caption)
                        .bold()
                    Text("SigLIP / CLIP Vision Encoder generates 256 embedding vectors for ANE acceleration.")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Button(viewModel.selectedImage == nil ? "Attach Problem Image" : "Remove Image") {
                        if viewModel.selectedImage == nil {
                            // Dummy CGImage creation for demo
                            viewModel.selectedImage = createDummyCGImage()
                        } else {
                            viewModel.selectedImage = nil
                        }
                    }
                    .font(.caption2)
                }
            }
        }
        .padding()
        .background(Color(uiColor: .secondarySystemBackground))
        .cornerRadius(12)
    }

    private var promptInputCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Reasoning Prompt")
                .font(.subheadline)
                .bold()
            TextField("Enter prompt", text: $viewModel.promptText)
                .textFieldStyle(.roundedBorder)
                .font(.body)
        }
        .padding()
        .background(Color(uiColor: .secondarySystemBackground))
        .cornerRadius(12)
    }

    private var executionButton: some View {
        Button(action: {
            Task {
                await viewModel.runReasoning()
            }
        }) {
            HStack {
                Image(systemName: "bolt.fill")
                Text(viewModel.isRunningReasoning ? "Reasoning in progress..." : "Run Parallel N=8 Reasoning")
                    .bold()
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.isRunningReasoning ? Color.gray : Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .disabled(viewModel.isRunningReasoning)
    }

    private var progressView: some View {
        VStack(spacing: 12) {
            ProgressView()
                .scaleEffect(1.2)
            Text(viewModel.currentStatus)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private var resultsView: some View {
        VStack(alignment: .leading, spacing: 15) {
            // Metrics Bar
            HStack {
                metricTile(title: "TTFT", value: String(format: "%.1f ms", viewModel.ttftMs))
                metricTile(title: "Latency", value: String(format: "%.1f ms", viewModel.totalLatencyMs))
                metricTile(title: "Speed", value: String(format: "%.0f tok/s", viewModel.throughputTokPerSec))
                metricTile(title: "PRM Score", value: String(format: "%.2f", viewModel.verifierScore))
            }

            // Step-Level Reflection Badge
            HStack {
                Image(systemName: viewModel.reflectionTriggered ? "exclamationmark.triangle.fill" : "checkmark.seal.fill")
                    .foregroundColor(viewModel.reflectionTriggered ? .orange : .green)
                Text(viewModel.reflectionTriggered ? "Step-Level Reflection Triggered (tau < 0.75)" : "Best-of-N Candidate Verified (tau >= 0.75)")
                    .font(.caption)
                    .bold()
            }
            .padding(8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(viewModel.reflectionTriggered ? Color.orange.opacity(0.15) : Color.green.opacity(0.15))
            .cornerRadius(8)

            // Best Verified Trace Output
            VStack(alignment: .leading, spacing: 6) {
                Text("Verified Solution Output")
                    .font(.headline)
                Text(viewModel.bestTraceOutput)
                    .font(.system(.body, design: .monospaced))
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.black.opacity(0.05))
                    .cornerRadius(8)
            }

            // Parallel Channel Comparison Drawer
            DisclosureGroup("Parallel Channel Traces (N=\(viewModel.candidatesEvaluated))") {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(0..<viewModel.channelTraces.count, id: \.self) { idx in
                        Text(viewModel.channelTraces[idx])
                            .font(.caption2)
                            .padding(6)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(idx == viewModel.bestCandidateIndex ? Color.blue.opacity(0.1) : Color.gray.opacity(0.1))
                            .cornerRadius(6)
                    }
                }
            }
            .font(.caption)
        }
        .padding()
        .background(Color(uiColor: .secondarySystemBackground))
        .cornerRadius(12)
    }

    private func metricTile(title: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption)
                .bold()
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(Color(uiColor: .tertiarySystemBackground))
        .cornerRadius(8)
    }

    private func createDummyCGImage() -> CGImage? {
        let width = 224
        let height = 224
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawData = [UInt8](repeating: 128, count: width * height * 4)
        return rawData.withUnsafeMutableBytes { ptr -> CGImage? in
            guard let context = CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            ) else { return nil }
            return context.makeImage()
        }
    }
}
