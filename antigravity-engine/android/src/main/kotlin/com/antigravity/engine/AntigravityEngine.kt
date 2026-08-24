package com.antigravity.engine

import java.io.File

enum class StorageMode {
    SHARED, // Zero-copy Unified Memory
    PRIVATE // Discrete GPU memory
}

enum class SearchMode {
    FIRST_FINISH_SEARCH, // TOPS: Stop immediately upon verified success
    EXHAUSTIVE_SEARCH // Run all N branches
}

data class AntigravityConfig(
    val maxMemoryAllocBytes: Long,
    val storageMode: StorageMode,
    val useSpeculativeDecoding: Boolean
)

enum class ExecutorType {
    V8,
    NATIVE_KOTLIN
}

sealed class VerificationResult {
    data class Verified(val reward: Float) : VerificationResult()
    data class Failed(val penalty: Float) : VerificationResult()
}

data class VerificationContract(
    val name: String,
    val executor: ExecutorType,
    val hook: (String, Map<String, Any>) -> VerificationResult
)

data class AgentResponse(
    val text: String,
    val prmScore: Float,
    val ttft: Double
)

class AntigravityEngine(private val config: AntigravityConfig) {
    private var engineHandle: Long = 0

    init {
        // Native JNI initialization mapping to Vulkan backend
        // engineHandle = nativeCreateEngine(config.maxMemoryAllocBytes, ...)
    }

    suspend fun loadTargetModel(modelFile: File) {
        // nativeLoadTargetModel(engineHandle, modelFile.absolutePath)
    }

    suspend fun loadDraftModel(modelFile: File) {
        // nativeLoadDraftModel(engineHandle, modelFile.absolutePath)
    }

    // external fun nativeCreateEngine(...): Long
    // ...
}

class Agent(
    private val engine: AntigravityEngine,
    private val systemPrompt: String,
    private val searchBudget: Int,
    private val verifiers: List<VerificationContract>
) {
    suspend fun generate(prompt: String, mode: SearchMode): AgentResponse {
        // JNI call to native generateMCTS
        return AgentResponse("Dummy Response", 1.0f, 24.5)
    }
}
