//
// Project Antigravity — App-Side Weight Download & Local Cache Manager
// Manages streaming, caching, and lifecycle of model weights (Safetensors / GGUF)
// in sandboxed iOS Application Support / Documents directories.
//

import Foundation
import CAntigravityEngine

/// Representation of downloadable model types for Antigravity Engine
public enum AntigravityModelType: Hashable, Sendable {
    case reasoner1B
    case verifier1_5B
    case custom(name: String, remoteURL: URL)

    public var defaultFileName: String {
        switch self {
        case .reasoner1B:
            return "tinyllama_1.1b_model.safetensors"
        case .verifier1_5B:
            return "skywork_prm_1.5b_model.safetensors"
        case .custom(let name, _):
            return "\(name).safetensors"
        }
    }

    public var defaultRemoteURL: URL {
        switch self {
        case .reasoner1B:
            return URL(string: "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors")!
        case .verifier1_5B:
            return URL(string: "https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/resolve/main/model.safetensors")!
        case .custom(_, let remoteURL):
            return remoteURL
        }
    }

    public var expectedByteCount: Int64 {
        switch self {
        case .reasoner1B:
            return 1_100_000_000 // ~1.1 GB
        case .verifier1_5B:
            return 2_880_000_000 // ~2.88 GB
        case .custom:
            return 0
        }
    }
}

/// Download Progress update emitted during model streaming
public struct DownloadProgress: Sendable {
    public enum State: Sendable, Equatable {
        case idle
        case downloading
        case completed
        case failed(String)
    }

    public let modelType: AntigravityModelType
    public let fractionCompleted: Double
    public let bytesDownloaded: Int64
    public let totalBytesExpected: Int64
    public let speedBytesPerSec: Double
    public let state: State

    public init(
        modelType: AntigravityModelType,
        fractionCompleted: Double,
        bytesDownloaded: Int64,
        totalBytesExpected: Int64,
        speedBytesPerSec: Double,
        state: State
    ) {
        self.modelType = modelType
        self.fractionCompleted = fractionCompleted
        self.bytesDownloaded = bytesDownloaded
        self.totalBytesExpected = totalBytesExpected
        self.speedBytesPerSec = speedBytesPerSec
        self.state = state
    }
}

/// Manager class for downloading and caching Safetensors model weights on iOS / macOS
public final class WeightManager: NSObject, @unchecked Sendable {
    public static let shared = WeightManager()

    private let fileManager = FileManager.default
    private let sessionQueue = OperationQueue()

    public var storageDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let modelsDir = appSupport.appendingPathComponent("AntigravityEngine/Models", isDirectory: true)
        if !fileManager.fileExists(atPath: modelsDir.path) {
            try? fileManager.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        }
        return modelsDir
    }

    public override init() {
        super.init()
        sessionQueue.maxConcurrentOperationCount = 2
    }

    /// Check if model is already downloaded locally in sandbox
    public func isModelDownloaded(type: AntigravityModelType) -> Bool {
        let fileURL = localURL(for: type)
        guard fileManager.fileExists(atPath: fileURL.path) else { return false }
        do {
            let attrs = try fileManager.attributesOfItem(atPath: fileURL.path)
            if let size = attrs[.size] as? Int64, size > 1024 * 1024 { // Must be at least > 1MB
                return true
            }
        } catch {
            return false
        }
        return false
    }

    /// Get local sandboxed URL for a given model type
    public func localURL(for type: AntigravityModelType) -> URL {
        return storageDirectory.appendingPathComponent(type.defaultFileName)
    }

    /// Delete downloaded local weights
    public func removeLocalModel(type: AntigravityModelType) throws {
        let url = localURL(for: type)
        if fileManager.fileExists(atPath: url.path) {
            try fileManager.removeItem(at: url)
        }
    }

    /// Stream download model weights from HF CDN / custom server directly to local disk
    public func downloadModel(
        type: AntigravityModelType,
        customURL: URL? = nil
    ) -> AsyncStream<DownloadProgress> {
        let remoteURL = customURL ?? type.defaultRemoteURL
        let destinationURL = localURL(for: type)
        let expectedBytes = type.expectedByteCount

        return AsyncStream { continuation in
            let sessionConfig = URLSessionConfiguration.default
            let delegate = SingleFileDownloadDelegate(
                modelType: type,
                destinationURL: destinationURL,
                totalBytesExpected: expectedBytes,
                continuation: continuation
            )

            let session = URLSession(configuration: sessionConfig, delegate: delegate, delegateQueue: self.sessionQueue)
            let task = session.downloadTask(with: remoteURL)
            task.resume()

            continuation.onTermination = { _ in
                task.cancel()
                session.finishTasksAndInvalidate()
            }
        }
    }

    /// Prepare and load model weights into the Antigravity C++ Metal Engine
    public func prepareAndLoad(
        modelType: AntigravityModelType,
        into engine: AntigravityEngine
    ) async throws {
        let destinationURL = localURL(for: modelType)
        if !isModelDownloaded(type: modelType) {
            // Stream download
            for await progress in downloadModel(type: modelType) {
                if case .failed(let err) = progress.state {
                    throw AntigravityError.modelLoadingFailed(reason: "Download failed for \(modelType.defaultFileName): \(err)")
                }
            }
        }

        guard isModelDownloaded(type: modelType) else {
            throw AntigravityError.modelLoadingFailed(reason: "Model file not available at \(destinationURL.path)")
        }

        try engine.loadModel(at: destinationURL.path)
    }
}

// MARK: - URLSession Download Delegate Helper

private final class SingleFileDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let modelType: AntigravityModelType
    private let destinationURL: URL
    private var totalBytesExpected: Int64
    private let continuation: AsyncStream<DownloadProgress>.Continuation
    private var startTime: Date?
    private var lastBytesRead: Int64 = 0

    init(
        modelType: AntigravityModelType,
        destinationURL: URL,
        totalBytesExpected: Int64,
        continuation: AsyncStream<DownloadProgress>.Continuation
    ) {
        self.modelType = modelType
        self.destinationURL = destinationURL
        self.totalBytesExpected = totalBytesExpected
        self.continuation = continuation
        super.init()
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        if startTime == nil {
            startTime = Date()
        }

        if totalBytesExpectedToWrite > 0 {
            totalBytesExpected = totalBytesExpectedToWrite
        }

        let duration = Date().timeIntervalSince(startTime ?? Date())
        let speed = duration > 0 ? Double(totalBytesWritten) / duration : 0.0
        let fraction = totalBytesExpected > 0 ? Double(totalBytesWritten) / Double(totalBytesExpected) : 0.0

        let progress = DownloadProgress(
            modelType: modelType,
            fractionCompleted: fraction,
            bytesDownloaded: totalBytesWritten,
            totalBytesExpected: totalBytesExpected,
            speedBytesPerSec: speed,
            state: .downloading
        )
        continuation.yield(progress)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: destinationURL.path) {
                try fileManager.removeItem(at: destinationURL)
            }
            try fileManager.moveItem(at: location, to: destinationURL)

            let finalAttrs = try fileManager.attributesOfItem(atPath: destinationURL.path)
            let finalSize = (finalAttrs[.size] as? Int64) ?? 0

            let finalProgress = DownloadProgress(
                modelType: modelType,
                fractionCompleted: 1.0,
                bytesDownloaded: finalSize,
                totalBytesExpected: finalSize,
                speedBytesPerSec: 0,
                state: .completed
            )
            continuation.yield(finalProgress)
            continuation.finish()
        } catch {
            let errProgress = DownloadProgress(
                modelType: modelType,
                fractionCompleted: 0,
                bytesDownloaded: 0,
                totalBytesExpected: totalBytesExpected,
                speedBytesPerSec: 0,
                state: .failed(error.localizedDescription)
            )
            continuation.yield(errProgress)
            continuation.finish()
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            let errProgress = DownloadProgress(
                modelType: modelType,
                fractionCompleted: 0,
                bytesDownloaded: 0,
                totalBytesExpected: totalBytesExpected,
                speedBytesPerSec: 0,
                state: .failed(error.localizedDescription)
            )
            continuation.yield(errProgress)
            continuation.finish()
        }
    }
}
