//
// Project Antigravity — Public Swift API Wrapper & Engine SDK
// Apple Silicon Edge Engine (iOS / macOS, M1-M4, A17 Pro, A18 Pro)
//

import Foundation
#if canImport(Speech)
import Speech
import AVFoundation
#endif

/// On-device speech recognition service for air-gapped clinical transcription.
/// Uses Apple's SFSpeechRecognizer with `requiresOnDeviceRecognition = true`
/// to guarantee zero network egress.
public actor SpeechService {
    public enum State: Sendable {
        case idle
        case recording
        case transcribing
        case error(String)
    }

    public struct TranscriptionSegment: Sendable {
        public let text: String
        public let startTime: TimeInterval
        public let endTime: TimeInterval
        public let confidence: Float
    }

#if canImport(Speech)
    // Properties
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine
#endif
    public private(set) var state: State = .idle
    public private(set) var segments: [TranscriptionSegment] = []
    public private(set) var currentTranscript: String = ""
    private var audioLevelHandler: (@Sendable (Float) -> Void)?

    public init(locale: Locale = Locale(identifier: "en-US")) {
#if canImport(Speech)
        self.speechRecognizer = SFSpeechRecognizer(locale: locale)
        self.audioEngine = AVAudioEngine()
#endif
    }

    public func setAudioLevelHandler(_ handler: (@Sendable (Float) -> Void)?) {
        self.audioLevelHandler = handler
    }

    public func dispatchAudioLevel(_ level: Float) {
        self.audioLevelHandler?(level)
    }

    // Request speech recognition authorization
    public func requestAuthorization() async -> Bool {
#if canImport(Speech)
        return await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { authStatus in
                continuation.resume(returning: authStatus == .authorized)
            }
        }
#else
        return false
#endif
    }

    // Check if on-device recognition is available
    public var isOnDeviceAvailable: Bool {
#if canImport(Speech)
        return speechRecognizer?.supportsOnDeviceRecognition ?? false
#else
        return false
#endif
    }

    // Start live microphone recording and on-device transcription
    public func startRecording() throws {
#if canImport(Speech)
#if os(iOS)
        let memoryFootprint = os_proc_available_memory()
        if memoryFootprint < 50 * 1024 * 1024 {
            print("Warning: Low memory available for speech recognition")
        }
#endif

        if let recognitionTask = recognitionTask {
            recognitionTask.cancel()
            self.recognitionTask = nil
        }

        #if os(iOS)
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        #endif

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw NSError(domain: "SpeechService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unable to create request"])
        }

        recognitionRequest.requiresOnDeviceRecognition = true
        recognitionRequest.shouldReportPartialResults = true

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        let localRequest = self.recognitionRequest

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, when) in
            localRequest?.append(buffer)

            if let channelData = buffer.floatChannelData?[0] {
                let frames = Int(buffer.frameLength)
                if frames > 0 {
                    var sumSquares: Float = 0.0
                    for i in 0..<frames {
                        sumSquares += channelData[i] * channelData[i]
                    }
                    let rms = sqrt(sumSquares / Float(frames))
                    let level = min(1.0, max(0.02, rms * 6.0))
                    // Bind to a let before the Task. Referencing the weak-captured
                    // `self` var from inside concurrently-executing code is a warning
                    // today and an error under Swift 6. The recognitionTask closure
                    // below already does this correctly.
                    if let self {
                        Task { await self.dispatchAudioLevel(level) }
                    }
                }
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
        state = .recording

        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            Task {
                await self.handleRecognition(result: result, error: error)
            }
        }
#else
        state = .error("Speech framework not available")
#endif
    }

#if canImport(Speech)
    private func handleRecognition(result: SFSpeechRecognitionResult?, error: Error?) {
        if let result = result {
            self.currentTranscript = result.bestTranscription.formattedString
            self.segments = result.bestTranscription.segments.map {
                TranscriptionSegment(
                    text: $0.substring,
                    startTime: $0.timestamp,
                    endTime: $0.timestamp + $0.duration,
                    confidence: $0.confidence
                )
            }
        }
        if error != nil {
            self.audioEngine.stop()
            self.audioEngine.inputNode.removeTap(onBus: 0)
            self.recognitionRequest = nil
            self.recognitionTask = nil
            self.state = .error(error?.localizedDescription ?? "Unknown error")
        }
    }
#endif

    // Stop recording and finalize transcription
    public func stopRecording() -> String {
#if canImport(Speech)
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        state = .idle
#endif
        return currentTranscript
    }

    // Transcribe pre-recorded audio file (for batch processing)
    public func transcribeFile(at url: URL) async throws -> [TranscriptionSegment] {
#if canImport(Speech)
        guard let recognizer = speechRecognizer else { return [] }
        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true

        return try await withCheckedThrowingContinuation { continuation in
            recognizer.recognitionTask(with: request) { result, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                if let result = result, result.isFinal {
                    let segs = result.bestTranscription.segments.map {
                        TranscriptionSegment(
                            text: $0.substring,
                            startTime: $0.timestamp,
                            endTime: $0.timestamp + $0.duration,
                            confidence: $0.confidence
                        )
                    }
                    continuation.resume(returning: segs)
                }
            }
        }
#else
        return []
#endif
    }

    // Get estimated memory footprint
    public var estimatedMemoryBytes: Int64 {
        return 50 * 1024 * 1024 // ~50MB for system-managed speech models
    }
}
