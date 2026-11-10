//  Qwen34BEngine.swift
//  MoatKit
//
//  Qwen3-4B engine using MLX Swift - native Qwen3 architecture support.
//  Uses MLX Swift LM (https://github.com/ml-explore/mlx-swift-lm) which has
//  native Qwen3 support via Qwen3Model.
//  This is the recommended path for Apple Silicon (M1/M2/M3/M4) and iOS
//  deployment - no custom Metal kernels needed, Apple's Metal backend
//  handles everything efficiently.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Qwen3-4B engine using MLX Swift - native Qwen3 architecture support.
public final class Qwen34BEngine: Sendable {
    public let container: ModelContainer

    /// Load Qwen3-4B from a local directory containing config.json,
    /// model.safetensors (or shards), and tokenizer.json.
    public init(modelDir: URL) async throws {
        let tokenizerLoader = LocalDirectoryTokenizerLoader(directory: modelDir)

        let container = try await LLMModelFactory.shared.loadContainer(
            from: modelDir,
            using: tokenizerLoader
        )
        self.container = container

        print("Qwen3-4B loaded from \(modelDir.path)")
    }

    /// Human-readable model name.
    public var modelName: String {
        "Qwen3-4B"
    }

    /// Generate text from a prompt.
    public func generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> String {
        let tokenIds = await container.encode(prompt)
        return try await generate(
            tokenIds: tokenIds,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP
        )
    }

    /// Generate text from pre-computed token IDs.
    public func generate(
        tokenIds: [Int],
        maxTokens: Int = 256,
        temperature: Float = 0.7,
        topP: Float = 0.9
    ) async throws -> String {
        let tokens = MLXArray(tokenIds)
        let input = LMInput(tokens: tokens)

        let parameters = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: 0,
            minP: 0.0,
            repetitionPenalty: nil,
            prefillStepSize: 512,
            seed: nil
        )

        let stream = try await container.generate(input: input, parameters: parameters)

        var generatedText = ""
        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
            case .info, .toolCall:
                break
            }
        }

        return generatedText
    }
}

// MARK: - Local Directory Tokenizer Loader

/// A `TokenizerLoader` that reads tokenizer.json from a local directory.
/// Implements byte-level BPE matching the Qwen3 HuggingFace tokenizer.
private final class LocalDirectoryTokenizerLoader: TokenizerLoader {
    private let directory: URL

    init(directory: URL) {
        self.directory = directory
    }

    func load(from url: URL) async throws -> any Tokenizer {
        let tokenizerURL = url.appending(component: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw NSError(
                domain: "LocalDirectoryTokenizerLoader", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "tokenizer.json not found at \(tokenizerURL.path)"]
            )
        }
        let data = try Data(contentsOf: tokenizerURL)
        return try Qwen3ByteLevelBPETokenizer(jsonData: data)
    }
}

/// Byte-level BPE tokenizer for Qwen3 models.
///
/// Matches the HuggingFace Qwen3 tokenizer:
/// 1. NFC normalization
/// 2. Regex pre-tokenization (GPT-2 style pattern)
/// 3. Byte-level encoding: each byte maps to a Unicode character
/// 4. BPE merges from vocab/merges in tokenizer.json
/// 5. Token-to-ID lookup from vocab
private struct Qwen3ByteLevelBPETokenizer: Tokenizer {
    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let merges: [(String, String)]
    private let mergeRanks: [String: Int]

    init(jsonData: Data) throws {
        let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] ?? [:]

        let modelDict = json["model"] as? [String: Any] ?? [:]
        let vocabDict = modelDict["vocab"] as? [String: Int] ?? [:]
        self.vocab = vocabDict
        self.idToToken = Dictionary(uniqueKeysWithValues: vocabDict.map { ($1, $0) })

        let mergesArray = modelDict["merges"] as? [[String]] ?? []
        var merges: [(String, String)] = []
        var ranks: [String: Int] = [:]
        for (rank, pair) in mergesArray.enumerated() {
            if pair.count == 2 {
                let left = pair[0], right = pair[1]
                merges.append((left, right))
                ranks[left + right] = rank
            }
        }
        self.merges = merges
        self.mergeRanks = ranks
    }

    // MARK: - Tokenizer protocol

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        // NFC normalize
        let normalized = text.precomposedStringWithCanonicalMapping

        // Pre-tokenization: match non-whitespace sequences (\S+).
        // For all pieces after the first, prepend U+0120 ("Ġ", the byte-level
        // encoding of space) to match HuggingFace's ByteLevel pre-tokenizer.
        let pattern = #"\S+"#
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let range = NSRange(normalized.startIndex..<normalized.endIndex, in: normalized)
        let matches = regex.matches(in: normalized, options: [], range: range)

        var allTokens: [String] = []
        for (idx, match) in matches.enumerated() {
            var piece = String(normalized[Range(match.range, in: normalized)!])
            // Prepend a space to all non-first pieces so that byte-level
            // encoding maps it to "Ġ" (U+0120), matching HF's ByteLevel
            // pre-tokenizer which marks word boundaries this way.
            if idx > 0 {
                piece = " " + piece
            }
            let byteTokens = byteLevelEncode(piece)
            let bpeTokens = applyBPE(byteTokens)
            allTokens.append(contentsOf: bpeTokens)
        }

        return allTokens.compactMap { vocab[$0] }
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        let tokens = tokenIds.compactMap { idToToken[$0] }
        // Merge byte-level tokens back into UTF-8
        var byteValues: [UInt8] = []
        for token in tokens {
            byteValues.append(contentsOf: tokenToBytes(token))
        }
        return String(bytes: byteValues, encoding: .utf8) ?? ""
    }

    func convertTokenToId(_ token: String) -> Int? {
        vocab[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        idToToken[id]
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text = messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        return encode(text: text, addSpecialTokens: true)
    }

    // MARK: - Byte-level encoding

    /// Maps each byte (0-255) to a Unicode character, matching the GPT-2 byte encoder.
    private func byteLevelEncode(_ text: String) -> [String] {
        let bytes = Array(text.utf8)
        return bytes.map { byteToUnicode[$0]! }
    }

    /// Converts a byte-level token back to its raw byte value.
    private func tokenToBytes(_ token: String) -> [UInt8] {
        guard let first = token.unicodeScalars.first else { return [] }
        let value = Int(first.value)
        if value < 256 {
            return [UInt8(value)]
        }
        // Extended bytes: 256 + n maps to byte n
        let n = value - 256
        if n >= 0 && n < 256 {
            return [UInt8(n)]
        }
        return []
    }

    // MARK: - BPE

    private func applyBPE(_ tokens: [String]) -> [String] {
        if tokens.count == 1 { return tokens }

        var tokens = tokens
        while true {
            var bestRank = Int.max
            var bestIdx = -1

            for i in 0..<tokens.count - 1 {
                let pair = tokens[i] + tokens[i + 1]
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }

            if bestIdx == -1 { break }

            tokens.replaceSubrange(bestIdx...bestIdx + 1, with: [tokens[bestIdx] + tokens[bestIdx + 1]])
        }

        return tokens
    }
}

// MARK: - Byte-to-Unicode mapping

/// GPT-2 byte encoder: maps each byte 0-255 to a Unicode character.
/// Matches the HuggingFace `ByteLevel` pre-tokenizer.
private let byteToUnicode: [UInt8: String] = {
    var bs: [UInt8] = []
    bs.append(contentsOf: Array(UInt8(33)...UInt8(126)))   // ! to ~
    bs.append(contentsOf: Array(UInt8(161)...UInt8(172)))  // ¡ to ¬
    bs.append(contentsOf: Array(UInt8(174)...UInt8(255)))  // ® to ÿ

    var cs: [Int] = bs.map { Int($0) }
    var n = 0
    for b in 0..<256 {
        if !bs.contains(UInt8(b)) {
            bs.append(UInt8(b))
            cs.append(256 + n)
            n += 1
        }
    }

    var dict: [UInt8: String] = [:]
    for i in 0..<bs.count {
        dict[bs[i]] = String(UnicodeScalar(cs[i])!)
    }
    return dict
}()
