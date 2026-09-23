import Foundation

// Copying a minimal mock of the Bridging Header functionality since we are compiling manually
@_silgen_name("AntigravityEngineCreate")
func AntigravityEngineCreate(_ config: UnsafeRawPointer) -> UnsafeMutableRawPointer?

@_silgen_name("AntigravityEngineLoadModel")
func AntigravityEngineLoadModel(_ ctx: UnsafeMutableRawPointer, _ path: UnsafePointer<CChar>) -> Int32

@_silgen_name("AntigravityEngineNativeGenerate")
func AntigravityEngineNativeGenerate(
    _ ctx: UnsafeMutableRawPointer,
    _ prompt_tokens: UnsafePointer<Int32>,
    _ prompt_len: Int32,
    _ max_new_tokens: Int32,
    _ temperature: Float,
    _ top_p: Float,
    _ out_tokens: UnsafeMutablePointer<Int32>,
    _ out_logprobs: UnsafeMutablePointer<Float>,
    _ out_token_counts: UnsafeMutablePointer<Int32>,
    _ out_ttft_ms: UnsafeMutablePointer<Double>,
    _ out_total_ms: UnsafeMutablePointer<Double>
) -> Int32

@_silgen_name("AntigravityEngineNativeMCTSGenerate")
func AntigravityEngineNativeMCTSGenerate(
    _ ctx: UnsafeMutableRawPointer,
    _ prompt_tokens: UnsafePointer<Int32>,
    _ prompt_len: Int32,
    _ mcts_config: UnsafeRawPointer?,
    _ out_tokens: UnsafeMutablePointer<Int32>,
    _ out_result: UnsafeMutableRawPointer?
) -> Int32

@_silgen_name("AntigravityEngineDestroy")
func AntigravityEngineDestroy(_ ctx: UnsafeMutableRawPointer)

func getPhysFootprint() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kerr == KERN_SUCCESS ? Double(info.resident_size) / (1024.0 * 1024.0) : 0.0
}

func runSwiftAppTest() {
    print("======================================================================")
    print("📱 Swift Native iOS App Logic Test (CLI Mode)")
    print("======================================================================")
    
    // Proper struct alignment
    struct Config {
        var n_channels: Int32 = 8
        var vocab_size: Int32 = 32000
        var hidden_dim: Int32 = 2048
        var max_seq_len: Int32 = 512
        var use_metal_gpu: Bool = true
    }
    var config = Config()
    
    print("[Swift] Initializing Metal Engine context...")
    guard let ctx = withUnsafePointer(to: &config, { ptr in
        AntigravityEngineCreate(UnsafeRawPointer(ptr))
    }) else {
        print("❌ Failed to create context")
        return
    }
    
    let modelDir = ProcessInfo.processInfo.environment["ANTIGRAVITY_MODEL_DIR"] ?? "models"
    let modelPath = modelDir + "/tinyllama/model_fp16.safetensors"
    print("[Swift] Loading weights from \(modelPath)")
    
    let res = modelPath.withCString { cstr in
        AntigravityEngineLoadModel(ctx, cstr)
    }
    
    if res != 0 {
        print("❌ Failed to load weights (err \(res))")
        return
    }
    
    print("[Swift] Physical Memory (RSS) after load: \(String(format: "%.1f", getPhysFootprint())) MB")
    
    let promptTokens: [Int32] = [1, 15043, 29892, 1125, 29892, 29871, 313, 29906] // "Prove that"
    var outTokens = [Int32](repeating: 0, count: 8 * 32)
    var outLogprobs = [Float](repeating: 0, count: 8)
    var outCounts = [Int32](repeating: 0, count: 8)
    var ttft_ms: Double = 0.0
    var total_ms: Double = 0.0
    
    print("\n--- 1. Parallel N=8 Rollout on Metal GPU ---")
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
    
    print("✅ Parallel Generation Complete!")
    print("  • Time To First Token: \(String(format: "%.2f", ttft_ms)) ms")
    print("  • Total Time: \(String(format: "%.2f", total_ms)) ms")
    print("  • Throughput: \(String(format: "%.1f", tokPerSec)) tok/s (N=8)")
    print("  • Peak Memory RSS: \(String(format: "%.1f", getPhysFootprint())) MB")
    
    print("\n--- 2. Native Chunk-Based MCTS Search with Branch Pruning ---")
    struct MCTSConfigC {
        var chunk_tokens: Int32 = 16
        var num_chunks: Int32 = 3
        var branches_per_chunk: Int32 = 4
        var temperature: Float = 0.8
        var top_p: Float = 0.9
    }
    struct MCTSResultC {
        var total_tokens_generated: Int32 = 0
        var total_tokens_evaluated: Int32 = 0
        var chunks_expanded: Int32 = 0
        var execution_wall_time_ms: Double = 0.0
        var best_score: Float = 0.0
    }
    
    var mctsConfig = MCTSConfigC()
    var mctsResult = MCTSResultC()
    var mctsTokens = [Int32](repeating: 0, count: 128)
    
    let mctsStatus = withUnsafePointer(to: &mctsConfig) { cfgPtr in
        withUnsafeMutablePointer(to: &mctsResult) { resPtr in
            AntigravityEngineNativeMCTSGenerate(
                ctx,
                promptTokens,
                Int32(promptTokens.count),
                UnsafeRawPointer(cfgPtr),
                &mctsTokens,
                UnsafeMutableRawPointer(resPtr)
            )
        }
    }
    
    if mctsStatus == 0 {
        print("✅ Native MCTS Search Complete!")
        print("  • Winning Tokens Generated: \(mctsResult.total_tokens_generated)")
        print("  • Total Tokens Evaluated Across Trees: \(mctsResult.total_tokens_evaluated)")
        print("  • Chunks Expanded: \(mctsResult.chunks_expanded)")
        print("  • Best Verification Score: \(String(format: "%.2f", mctsResult.best_score))")
        print("  • Total MCTS Time: \(String(format: "%.2f", mctsResult.execution_wall_time_ms)) ms")
    } else {
        print("❌ MCTS Search returned error \(mctsStatus)")
    }
    
    AntigravityEngineDestroy(ctx)
}

runSwiftAppTest()
