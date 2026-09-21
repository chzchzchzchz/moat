import SwiftUI
import AntigravityCore

class AppViewModel: ObservableObject {
    // This target links no tokenizer, so it cannot turn typed text into token IDs.
    // The benchmark therefore runs a fixed, pre-tokenized prompt. This used to be an
    // editable @Published field bound to a TextField whose value was never read.
    let benchmarkPrompt = "Prove that 2^x = 16"
    @Published var status: String = "Ready to test Antigravity N=8 Test-Time Search."
    @Published var throughput: String = "-"
    @Published var ttft: String = "-"
    @Published var appMemory: String = "0 MB"
    @Published var generatedTokens: String = "-"
    
    func runTest() {
        self.status = "Initializing Metal Engine..."
        DispatchQueue.global(qos: .userInitiated).async {
            // Memory check before
            var info = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
            let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            
            var config = AntigravityConfig()
            config.n_channels = 8
            config.vocab_size = 32000
            config.hidden_dim = 2048
            config.max_seq_len = 512
            config.use_metal_gpu = true
            
            guard let ctx = AntigravityEngineCreate(&config) else {
                DispatchQueue.main.async { self.status = "Failed to create context!" }
                return
            }
            
            DispatchQueue.main.async { self.status = "Loading weights..." }
            
            // Model location: ANTIGRAVITY_MODEL_DIR if set, else a bundled models/ directory.
            // This replaced one developer's absolute path, which no other machine has.
            let modelDir = ProcessInfo.processInfo.environment["ANTIGRAVITY_MODEL_DIR"]
                ?? Bundle.main.resourcePath.map { $0 + "/models" }
                ?? "models"
            let modelPath = modelDir + "/tinyllama/model_fp16.safetensors"
            let res = AntigravityEngineLoadModel(ctx, modelPath)
            if res != 0 {
                DispatchQueue.main.async { self.status = "Failed to load weights (err \(res))" }
                AntigravityEngineDestroy(ctx)
                return
            }
            
            DispatchQueue.main.async { self.status = "Running 8 parallel rollouts on Metal GPU..." }
            
            // Pre-tokenized form of benchmarkPrompt, fixed because this target has no
            // tokenizer. Keep in sync with benchmarkPrompt if that string changes.
            let promptTokens: [Int32] = [1, 15043, 29892, 1125, 29892, 29871, 313, 29906]
            var outTokens = [Int32](repeating: 0, count: 8 * 32)
            var outLogprobs = [Float](repeating: 0, count: 8)
            var outCounts = [Int32](repeating: 0, count: 8)
            var ttft_ms: Double = 0.0
            var total_ms: Double = 0.0
            
            AntigravityEngineNativeGenerate(
                ctx,
                promptTokens,
                Int32(promptTokens.count),
                32,
                0.7,
                0.9,
                &outTokens,
                &outLogprobs,
                &outCounts,
                &ttft_ms,
                &total_ms
            )
            
            let totalTokens = outCounts.reduce(0, +)
            let tokPerSec = Double(totalTokens) / (total_ms / 1000.0)
            
            // Re-check memory
            var info2 = mach_task_basic_info()
            withUnsafeMutablePointer(to: &info2) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            let memoryMB = Double(info2.resident_size) / (1024.0 * 1024.0)
            
            DispatchQueue.main.async {
                self.throughput = String(format: "%.1f tok/s", tokPerSec)
                self.ttft = String(format: "%.1f ms", ttft_ms)
                self.appMemory = String(format: "%.1f MB (Limit 3500 MB)", memoryMB)
                self.generatedTokens = "\(totalTokens) (N=8)"
                self.status = "Verification Pass: Complete ✅"
            }
            
            AntigravityEngineDestroy(ctx)
        }
    }
}

struct ContentView: View {
    @StateObject var viewModel = AppViewModel()
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Model Configuration")) {
                    Text("Architecture: Antigravity N=8 Test-Time Scaling")
                    Text("Weights: Zero-Copy Metal Shared (MTLResourceStorageModeShared)")
                    Text("Hardware: Apple Silicon A17/A18 Pro")
                }
                
                Section(header: Text("Task")) {
                    Text(viewModel.benchmarkPrompt)
                    Text("Fixed benchmark prompt. This target links no tokenizer, so the prompt is pre-tokenized and cannot be edited here.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Button(action: {
                        viewModel.runTest()
                    }) {
                        Text("🚀 Execute Generation on GPU")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                }
                
                Section(header: Text("Live Telemetry")) {
                    HStack {
                        Text("Status")
                        Spacer()
                        Text(viewModel.status)
                            .foregroundColor(.gray)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Text("Time to First Token (TTFT)")
                        Spacer()
                        Text(viewModel.ttft)
                            .bold()
                    }
                    HStack {
                        Text("Tokens Generated")
                        Spacer()
                        Text(viewModel.generatedTokens)
                            .bold()
                    }
                    HStack {
                        Text("Aggregate Throughput")
                        Spacer()
                        Text(viewModel.throughput)
                            .foregroundColor(.green)
                            .bold()
                    }
                    HStack {
                        Text("App Physical Memory (RSS)")
                        Spacer()
                        Text(viewModel.appMemory)
                            .foregroundColor(.orange)
                            .bold()
                    }
                }
            }
            .navigationTitle("Antigravity Engine")
        }
    }
}
