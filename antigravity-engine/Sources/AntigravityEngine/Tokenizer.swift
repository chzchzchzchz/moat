import Foundation

/// Fast on-device BPE Tokenizer for LLM inference on Apple Silicon.
/// Loads standard Hugging Face `tokenizer.json` including merge rules
/// and performs proper Byte-Pair Encoding with zero external dependencies.
public final class AntigravityTokenizer: @unchecked Sendable {
    private var vocab: [String: Int32] = [:]
    private var idToToken: [Int32: String] = [:]
    private var merges: [(String, String)] = []
    private var mergeRanks: [String: Int] = [:]
    private var bosTokenId: Int32 = 1
    private var eosTokenId: Int32 = 2
    private var padTokenId: Int32 = 0
    private var addedTokens: Set<String> = []
    private let queue = DispatchQueue(label: "org.antigravity.tokenizer", qos: .userInitiated)
    private var hasMerges: Bool { !merges.isEmpty }

    public var vocabSize: Int {
        return vocab.count
    }

    /// Initialize with an optional path to `tokenizer.json`.
    /// If no file is provided or file is not found, initializes a byte-level fallback vocabulary.
    public init(tokenizerJSONURL: URL? = nil) {
        if let url = tokenizerJSONURL, FileManager.default.fileExists(atPath: url.path) {
            loadFromJSON(url: url)
        } else {
            initializeByteLevelFallback()
        }
    }

    private func loadFromJSON(url: URL) {
        do {
            let data = try Data(contentsOf: url)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let model = json["model"] as? [String: Any],
                  let rawVocab = model["vocab"] as? [String: Int] else {
                initializeByteLevelFallback()
                return
            }

            // Load vocabulary
            for (token, id) in rawVocab {
                let id32 = Int32(id)
                vocab[token] = id32
                idToToken[id32] = token
            }

            // Load BPE merge rules (critical for real tokenization)
            if let rawMerges = model["merges"] as? [String] {
                for (rank, mergeStr) in rawMerges.enumerated() {
                    let parts = mergeStr.split(separator: " ", maxSplits: 1)
                    if parts.count == 2 {
                        let pair = (String(parts[0]), String(parts[1]))
                        merges.append(pair)
                        mergeRanks[mergeStr] = rank
                    }
                }
            }

            // Load added tokens (special tokens like <s>, </s>, <pad>)
            if let addedTokensList = json["added_tokens"] as? [[String: Any]] {
                for tokenInfo in addedTokensList {
                    if let content = tokenInfo["content"] as? String,
                       let id = tokenInfo["id"] as? Int {
                        let id32 = Int32(id)
                        vocab[content] = id32
                        idToToken[id32] = content
                        addedTokens.insert(content)

                        // Detect special token IDs
                        if content == "<s>" || content == "<|begin_of_text|>" || content == "<|startoftext|>" {
                            bosTokenId = id32
                        } else if content == "</s>" || content == "<|end_of_text|>" || content == "<|endoftext|>" ||
                                    content == "<|im_end|>" || content == "<|eot_id|>" {
                            eosTokenId = id32
                        } else if content == "<pad>" || content == "<|padding|>" {
                            padTokenId = id32
                        }
                    }
                }
            }

            if merges.isEmpty {
                print("[AntigravityTokenizer] WARNING: tokenizer.json loaded vocab (\(vocab.count) tokens) but has 0 merge rules. Encoding will use character-level fallback.")
            } else {
                print("[AntigravityTokenizer] Loaded \(vocab.count) tokens with \(merges.count) BPE merge rules.")
            }

        } catch {
            print("[AntigravityTokenizer] Failed to load tokenizer.json: \(error.localizedDescription)")
            initializeByteLevelFallback()
        }
    }

    private func initializeByteLevelFallback() {
        // Standard ASCII & byte-level tokens
        for b in 0...255 {
            let token = String(UnicodeScalar(UInt8(b)))
            let id = Int32(b + 3)
            vocab[token] = id
            idToToken[id] = token
        }
        vocab["<pad>"] = 0
        idToToken[0] = "<pad>"
        vocab["<s>"] = 1
        idToToken[1] = "<s>"
        vocab["</s>"] = 2
        idToToken[2] = "</s>"
    }

    // MARK: - BPE Core Algorithm

    /// Apply BPE merges to a list of tokens until no more merges can be applied.
    private func applyBPE(tokens: [String]) -> [String] {
        guard hasMerges else { return tokens }
        var current = tokens
        
        while current.count > 1 {
            // Find the highest-priority (lowest rank) merge pair
            var bestRank = Int.max
            var bestIdx = -1
            
            for i in 0..<(current.count - 1) {
                let pairKey = "\(current[i]) \(current[i + 1])"
                if let rank = mergeRanks[pairKey], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            
            // No more applicable merges
            if bestIdx == -1 { break }
            
            // Merge the pair
            let merged = current[bestIdx] + current[bestIdx + 1]
            var newTokens: [String] = []
            var i = 0
            while i < current.count {
                if i == bestIdx {
                    newTokens.append(merged)
                    i += 2  // skip the merged pair
                } else {
                    newTokens.append(current[i])
                    i += 1
                }
            }
            current = newTokens
        }
        
        return current
    }

    /// Convert a word into initial BPE tokens.
    /// For SentencePiece-style tokenizers, the word is prefixed with ▁ (U+2581).
    /// For GPT-style byte-level BPE, each byte is mapped to its unicode representation.
    /// For byte-level fallback (no tokenizer.json), each character maps directly.
    private func wordToInitialTokens(_ word: String) -> [String] {
        if !hasMerges {
            // Byte-level fallback: each character maps directly to vocab
            return word.map { String($0) }
        }
        
        // Check if vocab uses SentencePiece-style ▁ prefix (like LLaMA/TinyLlama)
        let usesSentencePiece = vocab.keys.contains(where: { $0.hasPrefix("▁") })
        
        if usesSentencePiece {
            // SentencePiece: each character is a token, unicode bytes as fallback
            return word.map { String($0) }
        } else {
            // GPT-style byte-level BPE: map each byte to its GPT-2 byte representation
            return word.utf8.map { byteToGPT2Token($0) }
        }
    }

    /// Map a byte value to its GPT-2 byte-level BPE unicode token.
    private func byteToGPT2Token(_ byte: UInt8) -> String {
        // GPT-2 byte-level mapping: printable ASCII bytes map to themselves,
        // non-printable bytes map to unicode range starting at U+0100
        let b = Int(byte)
        if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
            return String(UnicodeScalar(b)!)
        } else {
            // Map 0x00-0x20, 0x7F-0xA0, 0xAD to U+0100+offset
            return String(UnicodeScalar(b + 256)!)
        }
    }

    /// Pre-tokenize text by splitting on whitespace boundaries and punctuation,
    /// preserving the leading space as part of the token (SentencePiece ▁ convention).
    private func preTokenize(_ text: String) -> [String] {
        // Byte-level fallback: preserve each word and space as separate chunks
        if !hasMerges {
            var chunks: [String] = []
            var current = ""
            for char in text {
                if char == " " || char == "\n" || char == "\t" || char == "\r" {
                    if !current.isEmpty {
                        chunks.append(current)
                        current = ""
                    }
                    chunks.append(String(char))
                } else {
                    current.append(char)
                }
            }
            if !current.isEmpty { chunks.append(current) }
            return chunks
        }
        
        let usesSentencePiece = vocab.keys.contains(where: { $0.hasPrefix("▁") })
        
        var words: [String] = []
        var current = ""
        
        for (i, char) in text.enumerated() {
            if char == " " || char == "\n" || char == "\t" || char == "\r" {
                if !current.isEmpty {
                    words.append(current)
                    current = ""
                }
                if usesSentencePiece {
                    // In SentencePiece, spaces become ▁ prefix on the next word
                    if char == " " {
                        current = "▁"
                    }
                } else {
                    // In GPT-style, space is part of the word
                    current = String(char)
                }
            } else {
                if i == 0 && usesSentencePiece {
                    // SentencePiece: add ▁ at start of text
                    current = "▁" + String(char)
                } else {
                    current.append(char)
                }
            }
        }
        if !current.isEmpty && current != "▁" {
            words.append(current)
        }
        
        return words
    }

    // MARK: - Public API

    /// Encode human text into token IDs using BPE merge rules.
    public func encode(text: String, addBOS: Bool = true) -> [Int32] {
        return queue.sync {
            var tokens: [Int32] = []
            if addBOS {
                tokens.append(bosTokenId)
            }

            let words = preTokenize(text)
            
            for word in words {
                // First check if the whole word is in vocab (common for short/frequent words)
                if let exactId = vocab[word] {
                    tokens.append(exactId)
                    continue
                }
                
                // Split word into initial character/byte tokens and apply BPE merges
                let initialTokens = wordToInitialTokens(word)
                let mergedTokens = applyBPE(tokens: initialTokens)
                
                // Map merged subword tokens to IDs
                for subword in mergedTokens {
                    if let id = vocab[subword] {
                        tokens.append(id)
                    } else {
                        // Byte-level fallback for unknown subwords
                        for byte in subword.utf8 {
                            let byteStr = String(UnicodeScalar(byte))
                            tokens.append(vocab[byteStr] ?? Int32(byte + 3))
                        }
                    }
                }
            }
            return tokens
        }
    }

    /// Decode token IDs back into readable text.
    public func decode(tokens: [Int32], skipSpecialTokens: Bool = true) -> String {
        return queue.sync {
            var result = ""
            for id in tokens {
                if skipSpecialTokens {
                    if id == bosTokenId || id == eosTokenId || id == padTokenId { continue }
                    if let token = idToToken[id], addedTokens.contains(token) { continue }
                }

                if let token = idToToken[id] {
                    // Replace SentencePiece space marker '▁' with actual space
                    let clean = token.replacingOccurrences(of: "▁", with: " ")
                    result.append(clean)
                } else if id >= 3 && id <= 258 {
                    let byte = UInt8(id - 3)
                    result.append(String(UnicodeScalar(byte)))
                }
            }
            // Trim only leading space (SentencePiece adds ▁ at start)
            if result.hasPrefix(" ") {
                result = String(result.dropFirst())
            }
            return result
        }
    }
}
