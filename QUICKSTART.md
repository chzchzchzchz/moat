# Quickstart Guide: Project Antigravity 🛸

> Integrate hardware-accelerated, offline Best-of-N LLM inference into your iOS/macOS app in under 30 minutes.

---

## 1. Installation

### Swift Package Manager (SPM) / Xcode
1. Drag `AntigravityEngine.xcframework` into your Xcode project.
2. Under **Frameworks, Libraries, and Embedded Content**, set to **Embed & Sign**.
3. Add `Info.plist` key to enforce kernel-level network air-gapping:
   ```xml
   <key>NSAppTransportSecurity</key>
   <dict>
       <key>NSAllowsArbitraryLoads</key>
       <false/>
   </dict>
   ```

---

## 2. Hello World (Swift Integration)

```swift
import AntigravityEngine

// Initialize engine with strict 4.5GB iOS memory entitlement ceiling
let engine = try AntigravityEngine(config: .strict4GBFootprint)

// Run offline Best-of-N reasoning query
let result = try await engine.generate(
    prompt: "Prove that 2^n > n^2 for all n >= 5",
    maxTokens: 50
)

print("Best Reasoning Trace:\n\(result.bestTraceText)")
print("Verifier Score: \(result.verifierScore)")
print("Token Savings: \(result.tokenSavingsPercentage)%")
```

---

## 3. Python Integration

```python
from orchestrator import AntigravityEngine

engine = AntigravityEngine(n_channels=8)
result = engine.run_best_of_n_query("Prove 2^n > n^2 for n >= 5")

print(result['best_trace'])
```

---

## 4. Local API Server (OpenAI Compatible)

Start local HTTP server on port 8080:
```bash
python3 antigravity-engine/run_server.py --port 8080
```

Query via `curl`:
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "antigravity-1.5b-tts",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```
