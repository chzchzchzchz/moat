import SwiftUI
import AntigravityEngine

public struct ContentView: View {
    @StateObject private var viewModel = TherapistViewModel()

    public init() {}

    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    headerBanner
                    privacyBadge
                    modelStatusCard
                    patientAndTemplateSelector
                    recordingControlCard
                    transcriptInputCard
                    executionButton
                    
                    if viewModel.isAnalyzing {
                        progressView
                    }
                    
                    if !viewModel.analysisOutput.isEmpty {
                        resultsCard
                    }
                }
                .padding()
            }
            .navigationTitle("Therapy Note Assistant")
        }
    }

    private var headerBanner: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: "stethoscope")
                    .font(.title2)
                    .foregroundColor(.teal)
                Text("Clinical Summarizer Pro")
                    .font(.headline)
                    .foregroundColor(.primary)
                Spacer()
                Text("v2.5 Enterprise")
                    .font(.caption2)
                    .padding(4)
                    .background(Color.teal.opacity(0.15))
                    .cornerRadius(4)
            }
            Text("Automated, HIPAA-compliant session documentation powered by Apple Silicon Edge AI.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }

    private var privacyBadge: some View {
        HStack {
            Image(systemName: "lock.shield.fill")
                .font(.title3)
                .foregroundColor(.green)
            VStack(alignment: .leading, spacing: 2) {
                Text("Zero-Trust Privacy Vault Active")
                    .font(.subheadline)
                    .bold()
                    .foregroundColor(.green)
                Text("Audio, ASR, and reasoning are strictly local (0 bytes network). Class A Secure Enclave storage.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(Color.green.opacity(0.08))
        .cornerRadius(10)
    }

    private var modelStatusCard: some View {
        Group {
            if !viewModel.isModelLoaded {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Model Weights Required")
                            .font(.subheadline)
                            .bold()
                            .foregroundColor(.orange)
                        Spacer()
                    }
                    Text("Inference requires model weights on Apple Silicon. Click below to download TinyLlama 1.1B (~1.1GB) into the sandboxed vault, or place weights in ~/moat/models/tinyllama/.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    if viewModel.isDownloadingModel {
                        ProgressView(value: viewModel.downloadProgress)
                        Text(viewModel.currentStatus)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    } else {
                        Button(action: {
                            Task {
                                await viewModel.downloadAndLoadModel()
                            }
                        }) {
                            HStack {
                                Image(systemName: "arrow.down.circle.fill")
                                Text("Download TinyLlama 1.1B Model")
                                    .bold()
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                    }
                }
                .padding()
                .background(Color.orange.opacity(0.08))
                .cornerRadius(10)
            }
        }
    }

    private var patientAndTemplateSelector: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Patient:")
                    .font(.subheadline)
                    .bold()
                TextField("Patient ID", text: $viewModel.patientIdentifier)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .frame(maxWidth: 160)
                Spacer()
            }

            HStack {
                Text("Format:")
                    .font(.subheadline)
                    .bold()
                Picker("Template", selection: $viewModel.selectedTemplate) {
                    ForEach(ClinicalTemplateType.allCases, id: \.self) { tmpl in
                        Text(tmpl.rawValue).tag(tmpl)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                Spacer()
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.08))
        .cornerRadius(10)
    }

    private var recordingControlCard: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Audio Capture")
                        .font(.subheadline)
                        .bold()
                    Text(viewModel.isRecording ? "Recording live on-device..." : "Dictate session or edit text below")
                        .font(.caption)
                        .foregroundColor(viewModel.isRecording ? .red : .secondary)
                }
                Spacer()
                Button(action: {
                    Task {
                        await viewModel.toggleRecording()
                    }
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.fill")
                            .foregroundColor(viewModel.isRecording ? .red : .white)
                        Text(viewModel.isRecording ? "Stop Dictation" : "Record Live")
                            .bold()
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(viewModel.isRecording ? Color.red : Color.teal)
                    .cornerRadius(8)
                }
            }

            if viewModel.isRecording || !viewModel.sessionTranscript.isEmpty {
                AudioWaveformView(levels: viewModel.audioLevels, isRecording: viewModel.isRecording)
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.08))
        .cornerRadius(10)
    }

    private var transcriptInputCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Session Dialogue")
                    .font(.subheadline)
                    .bold()
                Spacer()
                Text("\(viewModel.sessionTranscript.count) chars")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            TextEditor(text: $viewModel.sessionTranscript)
                .frame(minHeight: 120)
                .padding(4)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                )
                .font(.system(.footnote, design: .monospaced))
        }
        .padding()
        .background(Color.secondary.opacity(0.08))
        .cornerRadius(12)
    }

    private var executionButton: some View {
        Button(action: {
            Task {
                await viewModel.runAnalysis()
            }
        }) {
            HStack {
                Image(systemName: "wand.and.stars")
                Text(viewModel.isAnalyzing ? "Synthesizing Locally..." : "Generate Clinical Document (Offline)")
                    .bold()
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(viewModel.isAnalyzing ? Color.gray : Color.teal)
            .foregroundColor(.white)
            .cornerRadius(10)
        }
        .disabled(viewModel.isAnalyzing || viewModel.isRecording)
    }

    private var progressView: some View {
        VStack(spacing: 8) {
            ProgressView()
            Text(viewModel.currentStatus)
                .font(.caption)
                .foregroundColor(.teal)
                .multilineTextAlignment(.center)
        }
        .padding()
    }

    private var resultsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "doc.text.fill")
                    .foregroundColor(.teal)
                Text(viewModel.selectedTemplate.rawValue)
                    .font(.headline)
                Spacer()
                if viewModel.isEncryptedInKeychain {
                    HStack(spacing: 4) {
                        Image(systemName: "lock.fill")
                            .font(.caption2)
                            .foregroundColor(.green)
                        Text("Encrypted")
                            .font(.caption2)
                            .foregroundColor(.green)
                    }
                    .padding(4)
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(4)
                }
            }
            Divider()
            Text(viewModel.analysisOutput)
                .font(.system(.caption, design: .monospaced))
                .padding(.vertical, 4)
            
            HStack {
                Button(action: {
                    viewModel.exportPDF()
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "arrow.down.doc.fill")
                        Text("Export Encrypted PDF")
                            .font(.subheadline)
                            .bold()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.teal.opacity(0.15))
                    .foregroundColor(.teal)
                    .cornerRadius(8)
                }

                Spacer()
                Text(viewModel.currentStatus)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.08))
        .cornerRadius(12)
    }
}
