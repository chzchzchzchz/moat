//
// Project Antigravity — Swift SDK Unit Test Suite
// Tests Swift-only components (config, weight manager, vision encoder patch math)
// C++ Metal engine is tested separately via Python pytest and native test_c_api_client
//

import XCTest
@testable import AntigravityEngine

final class AntigravityEngineTests: XCTestCase {

    // MARK: - EngineConfig Tests

    func testEngineConfigStrictDefaults() {
        let config = EngineConfig.strict4GBFootprint
        XCTAssertEqual(config.memoryLimitBytes, 4500 * 1024 * 1024)
        XCTAssertEqual(config.parallelChannels, 8)
        XCTAssertEqual(config.reflectionThreshold, 0.75)
        XCTAssertEqual(config.vocabSize, 32000)
        XCTAssertEqual(config.hiddenDim, 2048)
        XCTAssertTrue(config.useMetalGPU)
    }

    func testEngineConfigCustomInit() {
        let config = EngineConfig(
            memoryLimitBytes: 2000 * 1024 * 1024,
            parallelChannels: 4,
            reflectionThreshold: 0.5,
            vocabSize: 50000,
            hiddenDim: 4096,
            useMetalGPU: false
        )
        XCTAssertEqual(config.parallelChannels, 4)
        XCTAssertEqual(config.vocabSize, 50000)
        XCTAssertEqual(config.hiddenDim, 4096)
        XCTAssertFalse(config.useMetalGPU)
    }

    // MARK: - VisionEncoder Tests

    func testVisionEncoderPatchCount224() {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 2048, patchSize: 14, imageSize: 224)
        // 224 / 14 = 16, 16 * 16 = 256
        XCTAssertEqual(encoder.patchCount, 256)
    }

    func testVisionEncoderPatchCount384() {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 2048, patchSize: 16, imageSize: 384)
        // 384 / 16 = 24, 24 * 24 = 576
        XCTAssertEqual(encoder.patchCount, 576)
    }

    func testVisionEncoderFallbackEncoding() throws {
        let encoder = VisionEncoder(modelURL: nil, hiddenDim: 64, patchSize: 14, imageSize: 28)
        // 28 / 14 = 2, 2 * 2 = 4 patches
        XCTAssertEqual(encoder.patchCount, 4)

        // Create a tiny 28x28 CGImage for testing
        let width = 28, height = 28
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var rawData = [UInt8](repeating: 128, count: width * height * 4)
        let context = CGContext(
            data: &rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        let image = context.makeImage()!

        let embeddings = try encoder.encode(image: image)
        XCTAssertEqual(embeddings.count, 4 * 64, "Expected 4 patches * 64 hidden_dim = 256 floats")
        // No NaN or Inf values
        XCTAssertFalse(embeddings.contains(where: { $0.isNaN || $0.isInfinite }))
    }

    // MARK: - AgentTool Tests

    func testAgentToolSchema() {
        let tool = AgentTool(name: "calculator", description: "Math API", jsonSchema: "{\"type\":\"object\"}")
        XCTAssertEqual(tool.name, "calculator")
        XCTAssertEqual(tool.description, "Math API")
        XCTAssertEqual(tool.jsonSchema, "{\"type\":\"object\"}")
    }

    // MARK: - WeightManager Tests

    func testWeightManagerStorageDirectory() {
        let manager = WeightManager.shared
        let dir = manager.storageDirectory
        XCTAssertTrue(dir.path.contains("AntigravityEngine"))
    }

    func testWeightManagerLocalURLs() {
        let manager = WeightManager.shared
        let reasonerURL = manager.localURL(for: .reasoner1B)
        let verifierURL = manager.localURL(for: .verifier1_5B)
        XCTAssertTrue(reasonerURL.lastPathComponent.contains("tinyllama"))
        XCTAssertTrue(verifierURL.lastPathComponent.contains("skywork"))
    }

    func testWeightManagerModelNotDownloaded() {
        let manager = WeightManager.shared
        // Custom model type should not be downloaded
        let customType = AntigravityModelType.custom(name: "test_nonexistent", remoteURL: URL(string: "https://example.com/x.safetensors")!)
        XCTAssertFalse(manager.isModelDownloaded(type: customType))
    }

    // MARK: - DownloadProgress Tests

    func testDownloadProgressStruct() {
        let progress = DownloadProgress(
            modelType: .reasoner1B,
            fractionCompleted: 0.75,
            bytesDownloaded: 825_000_000,
            totalBytesExpected: 1_100_000_000,
            speedBytesPerSec: 30_000_000,
            state: .downloading
        )
        XCTAssertEqual(progress.fractionCompleted, 0.75, accuracy: 0.001)
        XCTAssertEqual(progress.bytesDownloaded, 825_000_000)
        XCTAssertEqual(progress.state, .downloading)
    }

    func testDownloadProgressCompletedState() {
        let progress = DownloadProgress(
            modelType: .verifier1_5B,
            fractionCompleted: 1.0,
            bytesDownloaded: 2_880_000_000,
            totalBytesExpected: 2_880_000_000,
            speedBytesPerSec: 0,
            state: .completed
        )
        XCTAssertEqual(progress.state, .completed)
        XCTAssertEqual(progress.fractionCompleted, 1.0)
    }

    // MARK: - AntigravityModelType Tests

    func testModelTypeDefaultFileNames() {
        XCTAssertEqual(AntigravityModelType.reasoner1B.defaultFileName, "tinyllama_1.1b_model.safetensors")
        XCTAssertEqual(AntigravityModelType.verifier1_5B.defaultFileName, "skywork_prm_1.5b_model.safetensors")
    }

    func testModelTypeExpectedBytes() {
        XCTAssertEqual(AntigravityModelType.reasoner1B.expectedByteCount, 1_100_000_000)
        XCTAssertEqual(AntigravityModelType.verifier1_5B.expectedByteCount, 2_880_000_000)
    }

    // MARK: - AntigravityError Tests

    func testErrorDescriptions() {
        let memError = AntigravityError.memoryBudgetExceeded(requestedBytes: 5_000_000_000, ceilingBytes: 4_500_000_000)
        XCTAssertNotNil(memError.errorDescription)
        XCTAssertTrue(memError.errorDescription!.contains("Memory budget exceeded"))

        let loadError = AntigravityError.modelLoadingFailed(reason: "file not found")
        XCTAssertTrue(loadError.errorDescription!.contains("file not found"))

        let execError = AntigravityError.executionFailed(reason: "GPU timeout")
        XCTAssertTrue(execError.errorDescription!.contains("GPU timeout"))

        let visionError = AntigravityError.visionEncodingFailed(reason: "bad image")
        XCTAssertTrue(visionError.errorDescription!.contains("bad image"))
    }
}
