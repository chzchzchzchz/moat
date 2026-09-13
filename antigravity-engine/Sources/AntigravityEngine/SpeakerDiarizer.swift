//
// Project Antigravity — Public Swift API Wrapper & Engine SDK
// Apple Silicon Edge Engine (iOS / macOS, M1-M4, A17 Pro, A18 Pro)
//

import Foundation
import Accelerate

/// Lightweight 2-speaker diarizer for clinical sessions.
/// Uses voice enrollment with spectral feature matching to distinguish
/// therapist from patient without external ML model dependencies.
public struct SpeakerDiarizer: Sendable {
    public enum SpeakerRole: String, Sendable, Codable {
        case therapist = "Therapist"
        case patient = "Patient"
        case unknown = "Unknown"
    }

    public struct DiarizedSegment: Sendable {
        public let speaker: SpeakerRole
        public let text: String
        public let startTime: TimeInterval
        public let endTime: TimeInterval
    }

    // Enrolled therapist voice embedding (spectral centroid, energy, ZCR, spectral rolloff)
    private var therapistEmbedding: [Float]?
    private let similarityThreshold: Float

    public init(similarityThreshold: Float = 0.70) {
        self.similarityThreshold = similarityThreshold
    }

    // Enroll therapist voice from a short audio sample
    // Extracts spectral features using Accelerate vDSP FFT
    public mutating func enrollTherapistVoice(audioSamples: [Float], sampleRate: Int = 16000) {
        self.therapistEmbedding = extractEmbedding(audioSamples: audioSamples, sampleRate: sampleRate)
    }

    // Extract speaker embedding from audio chunk
    // Features: spectral centroid, energy RMS, zero-crossing rate, spectral rolloff
    public func extractEmbedding(audioSamples: [Float], sampleRate: Int = 16000) -> [Float] {
        guard !audioSamples.isEmpty else { return [0, 0, 0, 0] }
        
        var rms: Float = 0
        vDSP_rmsqv(audioSamples, 1, &rms, vDSP_Length(audioSamples.count))
        
        var zcr: Float = 0
        var previousSign = audioSamples[0] >= 0
        for i in 1..<audioSamples.count {
            let currentSign = audioSamples[i] >= 0
            if currentSign != previousSign {
                zcr += 1
                previousSign = currentSign
            }
        }
        zcr /= Float(audioSamples.count)
        
        // Simple FFT for spectral features
        let log2n = vDSP_Length(log2(Float(audioSamples.count)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return [0, 0, 0, 0] }
        defer { vDSP_destroy_fftsetup(fftSetup) }
        
        let n = 1 << log2n
        var realP = [Float](repeating: 0, count: n/2)
        var imagP = [Float](repeating: 0, count: n/2)
        
        var centroid: Float = 0
        var rolloff: Float = 0
        
        realP.withUnsafeMutableBufferPointer { realPtr in
            imagP.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)
                
                audioSamples.withUnsafeBufferPointer { samplesPtr in
                    samplesPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: n/2) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(n/2))
                    }
                }
                
                vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
                
                var magnitudes = [Float](repeating: 0, count: n/2)
                vDSP_zvmags(&splitComplex, 1, &magnitudes, 1, vDSP_Length(n/2))
                
                var sumMags: Float = 0
                var sumWeightedMags: Float = 0
                for i in 0..<n/2 {
                    let freq = Float(i) * Float(sampleRate) / Float(n)
                    sumMags += magnitudes[i]
                    sumWeightedMags += magnitudes[i] * freq
                }
                centroid = sumMags > 0 ? sumWeightedMags / sumMags : 0
                
                var currentSum: Float = 0
                for i in 0..<n/2 {
                    currentSum += magnitudes[i]
                    if currentSum >= 0.85 * sumMags {
                        rolloff = Float(i) * Float(sampleRate) / Float(n)
                        break
                    }
                }
            }
        }
        
        var embedding: [Float] = [centroid, rms, zcr, rolloff]
        
        var sumSquares: Float = 0
        vDSP_svesq(embedding, 1, &sumSquares, vDSP_Length(embedding.count))
        let norm = sqrt(sumSquares)
        if norm > 0 {
            vDSP_vsdiv(embedding, 1, [norm], &embedding, 1, vDSP_Length(embedding.count))
        }
        
        return embedding
    }

    // Classify a segment's speaker
    public func identifySpeaker(segmentEmbedding: [Float]) -> SpeakerRole {
        guard let therapist = therapistEmbedding else { return .unknown }
        let similarity = cosineSimilarity(a: therapist, b: segmentEmbedding)
        return similarity >= similarityThreshold ? .therapist : .patient
    }

    // Cosine similarity between two embedding vectors
    public func cosineSimilarity(a: [Float], b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dotProduct: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        return dotProduct
    }

    // Diarize a full transcript with time-aligned audio chunks
    public func diarize(
        segments: [SpeechService.TranscriptionSegment],
        audioChunks: [[Float]],
        sampleRate: Int = 16000
    ) -> [DiarizedSegment] {
        var results: [DiarizedSegment] = []
        let count = min(segments.count, audioChunks.count)
        
        for i in 0..<count {
            let segment = segments[i]
            let chunk = audioChunks[i]
            let embedding = extractEmbedding(audioSamples: chunk, sampleRate: sampleRate)
            let speaker = identifySpeaker(segmentEmbedding: embedding)
            
            results.append(DiarizedSegment(
                speaker: speaker,
                text: segment.text,
                startTime: segment.startTime,
                endTime: segment.endTime
            ))
        }
        
        return results
    }
}
