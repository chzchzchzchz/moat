//
// Project Antigravity — CoreML Vision Embedding Encoder
// Encodes input CGImage / CVPixelBuffer into vision patch embedding vectors
// for multimodal models (Llama 3.2 Vision / Gemma 4 / MiniCPM-V)
// via CoreML, which prefers the Apple Neural Engine.
//
// Real embeddings require a CoreML vision model. Without one this type cannot
// produce them, and by default it refuses rather than substituting something
// that merely has the right shape. See allowsNonSemanticFallback.
//

import Foundation
import CoreGraphics
import ImageIO
import Accelerate
import CoreML
import Vision

/// Vision patch encoder using Apple Neural Engine / CoreML
public final class VisionEncoder: @unchecked Sendable {
    public let hiddenDim: Int
    public let patchSize: Int
    public let imageSize: Int
    public let compiledModelURL: URL?

    /// When no CoreML model is available, allow encode(image:) to return the
    /// patch colour-average descriptor described on that method.
    ///
    /// Those vectors are NOT vision embeddings: they carry no semantic content, and a
    /// multimodal model fed them will describe something unrelated to the image. Leave
    /// this false unless you are deliberately exercising the plumbing (shapes, buffer
    /// sizes, dispatch) and not the output.
    public let allowsNonSemanticFallback: Bool

    private var mlModel: MLModel?

    /// True when a CoreML vision model is loaded and real embeddings are possible.
    public var canProduceEmbeddings: Bool { mlModel != nil }

    public init(
        modelURL: URL? = nil,
        hiddenDim: Int = 2048,
        patchSize: Int = 14,
        imageSize: Int = 224,
        allowsNonSemanticFallback: Bool = false
    ) {
        self.hiddenDim = hiddenDim
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.compiledModelURL = modelURL
        self.allowsNonSemanticFallback = allowsNonSemanticFallback

        if let url = modelURL {
            let config = MLModelConfiguration()
            config.computeUnits = .all // Prefers Apple Neural Engine (ANE) + GPU
            self.mlModel = try? MLModel(contentsOf: url, configuration: config)
        }
    }

    /// Number of patches generated for an image of dimensions (imageSize x imageSize)
    public var patchCount: Int {
        let side = imageSize / patchSize
        return side * side
    }

    /// Encode CGImage into flattened FP32 patch embeddings [patchCount * hiddenDim].
    ///
    /// With a CoreML model loaded this runs that model. Without one it throws, unless
    /// `allowsNonSemanticFallback` is set, in which case it returns a descriptor built
    /// from each patch's mean RGB tiled across the hidden dimension with a positional
    /// ramp. That descriptor has the correct shape and no semantic content.
    public func encode(image: CGImage) throws -> [Float] {
        guard patchCount > 0 else {
            throw AntigravityError.visionEncodingFailed(reason: "Invalid patch dimensions")
        }

        let width = image.width
        let height = image.height
        guard width > 0 && height > 0 else {
            throw AntigravityError.visionEncodingFailed(reason: "Invalid image dimensions: \(width)x\(height)")
        }

        // --- CoreML Execution Path (if MLModel is loaded) ---
        if let model = mlModel {
            guard let pixelBuffer = createResizedPixelBuffer(
                from: image,
                targetSize: CGSize(width: imageSize, height: imageSize)
            ) else {
                throw AntigravityError.visionEncodingFailed(
                    reason: "Could not build a \(imageSize)x\(imageSize) pixel buffer for CoreML input"
                )
            }
            do {
                let featureProvider = try MLDictionaryFeatureProvider(dictionary: ["pixel_values": pixelBuffer])
                let prediction = try model.prediction(from: featureProvider)
                guard let multiArray = prediction.featureValue(for: "patch_embeddings")?.multiArrayValue else {
                    throw AntigravityError.visionEncodingFailed(
                        reason: "CoreML vision model produced no 'patch_embeddings' output"
                    )
                }
                let expected = patchCount * hiddenDim
                guard multiArray.count >= expected else {
                    throw AntigravityError.visionEncodingFailed(
                        reason: "CoreML vision model returned \(multiArray.count) values, expected \(expected)"
                    )
                }
                var result = [Float](repeating: 0.0, count: expected)
                let ptr = multiArray.dataPointer.bindMemory(to: Float.self, capacity: expected)
                for i in 0..<expected {
                    result[i] = ptr[i]
                }
                return result
            } catch let error as AntigravityError {
                throw error
            } catch {
                throw AntigravityError.visionEncodingFailed(
                    reason: "CoreML vision model prediction failed: \(error.localizedDescription)"
                )
            }
        }

        // --- No CoreML model: patch colour averages, NOT vision embeddings ---
        guard allowsNonSemanticFallback else {
            throw AntigravityError.visionEncodingFailed(
                reason: "No CoreML vision model is loaded, so real patch embeddings cannot be "
                      + "produced. Supply modelURL, or set allowsNonSemanticFallback to accept "
                      + "shape-correct patch colour averages that carry no semantic content."
            )
        }

        var embeddings = [Float](repeating: 0.0, count: patchCount * hiddenDim)
        let side = imageSize / patchSize

        // Extract raw pixel data from CGImage
        guard let dataProvider = image.dataProvider,
              let pixelData = dataProvider.data,
              let ptr = CFDataGetBytePtr(pixelData) else {
            // Returning the zero-filled buffer here would look like a successful encode.
            throw AntigravityError.visionEncodingFailed(reason: "Could not read image pixel data")
        }

        let imgWidth = image.width
        let imgHeight = image.height
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow

        for p in 0..<patchCount {
            let row = p / side
            let col = p % side
            let startX = (col * imgWidth) / side
            let startY = (row * imgHeight) / side
            let endX = min(startX + max(1, imgWidth / side), imgWidth)
            let endY = min(startY + max(1, imgHeight / side), imgHeight)

            // Compute mean RGB color for the spatial patch
            var rSum: Float = 0
            var gSum: Float = 0
            var bSum: Float = 0
            var sampleCount: Float = 0

            for y in startY..<endY {
                for x in startX..<endX {
                    let offset = y * bytesPerRow + x * bytesPerPixel
                    if offset + 2 < CFDataGetLength(pixelData) {
                        rSum += Float(ptr[offset]) / 255.0
                        gSum += Float(ptr[offset + 1]) / 255.0
                        bSum += Float(ptr[offset + 2]) / 255.0
                        sampleCount += 1.0
                    }
                }
            }

            let meanR = sampleCount > 0 ? rSum / sampleCount : 0.5
            let meanG = sampleCount > 0 ? gSum / sampleCount : 0.5
            let meanB = sampleCount > 0 ? bSum / sampleCount : 0.5

            for k in 0..<hiddenDim {
                let channelWeight = (k % 3 == 0) ? meanR : ((k % 3 == 1) ? meanG : meanB)
                let posWeight = Float(p) / Float(patchCount)
                embeddings[p * hiddenDim + k] = (channelWeight * 0.8 + posWeight * 0.2) - 0.5
            }
        }

        return embeddings
    }

    /// Helper to convert CGImage to CVPixelBuffer scaled to targetSize for CoreML input
    private func createResizedPixelBuffer(from image: CGImage, targetSize: CGSize) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(targetSize.width),
            Int(targetSize.height),
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: Int(targetSize.width),
            height: Int(targetSize.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        )

        context?.draw(image, in: CGRect(origin: .zero, size: targetSize))
        CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))

        return buffer
    }
}
