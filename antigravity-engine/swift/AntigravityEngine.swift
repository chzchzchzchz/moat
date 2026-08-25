import Foundation
import JavaScriptCore

// C-API Struct Definitions

public struct AntigravityMCTSConfig {
    public var n_channels: Int32
    public var expansion_depth: Int32
    public var num_simulations: Int32
    public var temperature: Float
    public var top_p: Float
}


public struct AntigravityMCTSResult {
    public var total_tokens_generated: Int32
    public var total_tokens_evaluated: Int32
    public var chunks_expanded: Int32
    public var best_score: Float
    public var execution_wall_time_ms: Double
}

@_silgen_name("AntigravityEngineCreate")
func AntigravityEngineCreate(_ config: UnsafeRawPointer) -> UnsafeMutableRawPointer?

@_silgen_name("AntigravityEngineLoadModel")
func AntigravityEngineLoadModel(_ ctx: UnsafeMutableRawPointer, _ path: UnsafePointer<CChar>) -> Int32

@_silgen_name("AntigravityEngineDestroy")
func AntigravityEngineDestroy(_ ctx: UnsafeMutableRawPointer)

@_silgen_name("AntigravityEngineNativeMCTSGenerate")
func AntigravityEngineNativeMCTSGenerate(_ ctx: UnsafeMutableRawPointer, _ prompt_tokens: UnsafePointer<Int32>, _ prompt_len: Int32, _ config: UnsafePointer<AntigravityMCTSConfig>, _ out_tokens: UnsafeMutablePointer<Int32>, _ out_result: UnsafeMutablePointer<AntigravityMCTSResult>) -> Int32

public enum StorageMode {
    case shared
    case privateMode
}

public enum SearchMode {
    case firstFinishSearch
    case exhaustiveSearch
}

public struct AntigravityConfig {
    public let maxMemoryAllocBytes: Int64
    public let storageMode: StorageMode
    public let useSpeculativeDecoding: Bool

    public init(maxMemoryAllocBytes: Int64, storageMode: StorageMode, useSpeculativeDecoding: Bool) {
        self.maxMemoryAllocBytes = maxMemoryAllocBytes
        self.storageMode = storageMode
        self.useSpeculativeDecoding = useSpeculativeDecoding
    }
}

public enum ExecutorType {
    case javascriptCore
    case nativeSwift
}

public enum VerificationResult {
    case verified(reward: Float)
    case failed(penalty: Float)
}

public struct VerificationContract {
    public let name: String
    public let executor: ExecutorType
    public let hook: (String, [String: Any]) -> VerificationResult
    
    public init(name: String, executor: ExecutorType, hook: @escaping (String, [String: Any]) -> VerificationResult) {
        self.name = name
        self.executor = executor
        self.hook = hook
    }
}

public struct AgentResponse {
    public let text: String
    public let prmScore: Float
    public let ttft: Double
}

public struct AntigravityNativeConfig {
    public var n_channels: Int32 = 8
    public var vocab_size: Int32 = 32000
    public var hidden_dim: Int32 = 2048
    public var max_seq_len: Int32 = 2048
    public var use_metal_gpu: Bool = true
}

public final class AntigravityEngine {
    private var engineHandle: UnsafeMutableRawPointer?
    private let config: AntigravityConfig

    public init(config: AntigravityConfig) throws {
        self.config = config
        // Actually instantiate the real C-API backend with properly aligned 32-bit struct
        var nativeCfg = AntigravityNativeConfig(
            n_channels: 8,
            vocab_size: 32000,
            hidden_dim: 2048,
            max_seq_len: 2048,
            use_metal_gpu: true
        )
        self.engineHandle = withUnsafePointer(to: &nativeCfg) { ptr in
            return AntigravityEngineCreate(ptr)
        }
    }

    public func loadTargetModel(url: URL) async throws {
        guard let handle = engineHandle else { return }
        url.path.withCString { cPath in
            _ = AntigravityEngineLoadModel(handle, cPath)
        }
    }
    
    deinit {
        if let handle = engineHandle {
            AntigravityEngineDestroy(handle)
        }
    }
    
    internal func getHandle() -> UnsafeMutableRawPointer? {
        return engineHandle
    }
}

public final class Agent {
    private let engine: AntigravityEngine
    private let systemPrompt: String
    private let searchBudget: Int
    private let verifiers: [VerificationContract]
    
    public init(engine: AntigravityEngine, systemPrompt: String, searchBudget: Int, verifiers: [VerificationContract]) {
        self.engine = engine
        self.systemPrompt = systemPrompt
        self.searchBudget = searchBudget
        self.verifiers = verifiers
    }
    
    public func generate(prompt: String, mode: SearchMode) async throws -> AgentResponse {
        guard let handle = engine.getHandle() else {
            throw NSError(domain: "EngineError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid engine handle"])
        }
        
        // Convert prompt string to primitive token IDs (using a basic ascii mapping for demonstration)
        let promptTokens: [Int32] = prompt.utf8.map { Int32($0) }
        var outTokens = [Int32](repeating: 0, count: 512)
        
        var mctsConfig = AntigravityMCTSConfig(
            n_channels: Int32(self.searchBudget),
            expansion_depth: 3,
            num_simulations: 4,
            temperature: 0.8,
            top_p: 0.9
        )
        var mctsRes = AntigravityMCTSResult(
            total_tokens_generated: 0,
            total_tokens_evaluated: 0,
            chunks_expanded: 0,
            best_score: 0.0,
            execution_wall_time_ms: 0.0
        )
        
        let ret = promptTokens.withUnsafeBufferPointer { promptPtr in
            outTokens.withUnsafeMutableBufferPointer { outPtr in
                AntigravityEngineNativeMCTSGenerate(
                    handle,
                    promptPtr.baseAddress!,
                    Int32(promptTokens.count),
                    &mctsConfig,
                    outPtr.baseAddress!,
                    &mctsRes
                )
            }
        }
        
        // Decode the generated tokens back to a string
        var finalCode = ""
        for token in outTokens {
            if token == 0 || token == 2 { break } // Stop at EOS or padding
            if let scalar = UnicodeScalar(UInt32(token)) {
                finalCode.append(Character(scalar))
            }
        }
        
        var score: Float = mctsRes.best_score
        
        for verifier in verifiers {
            if verifier.executor == .javascriptCore || verifier.executor == .nativeSwift {
                let res = verifier.hook(finalCode, [:])
                switch res {
                case .verified(let r): score += r
                case .failed(let p): score -= p
                }
            }
        }
        
        return AgentResponse(text: finalCode, prmScore: score, ttft: mctsRes.execution_wall_time_ms)
    }
}
