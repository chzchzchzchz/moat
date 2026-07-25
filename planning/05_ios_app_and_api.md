# Architectural Document 05: iOS App Architecture & OpenAI-Compatible Local Server

## 1. System Integration Architecture

Project Antigravity exposes its accelerated hardware execution pipeline via a lightweight, drop-in local server on iOS:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             iOS Application / Client                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP POST /v1/chat/completions
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                Local API Gateway (Swift / Network.framework)                │
│                        Listening on localhost:8080                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ JSON Request Payload
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Antigravity Local Engine Bus                           │
│  ┌───────────────────────┐   ┌───────────────────┐   ┌───────────────────┐  │
│  │ Batched GEMM Generator│   │  LUT Softmax Kernel│   │ List-Wise Verifier│  │
│  │ (Metal / simdgroup)   │──►│ (Exp Gather Table)│──►│ (PRM Sequential)  │  │
│  └───────────────────────┘   └───────────────────┘   └───────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Selected Best-of-N Output
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OpenAI SSE Response Stream                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. OpenAI API Compatibility Layer (`/v1/chat/completions`)

### 2.1 Endpoint Specification
- **Method:** `POST`
- **URL:** `http://localhost:8080/v1/chat/completions` (or Unix domain socket `/tmp/antigravity.sock`).
- **Headers:** `Content-Type: application/json`

### 2.2 Request Payload Format
```json
{
  "model": "antigravity-qwen2.5-1.5b-tts",
  "messages": [
    {"role": "system", "content": "You are a helpful math reasoning assistant."},
    {"role": "user", "content": "Solve for x: 3x + 5 = 20"}
  ],
  "temperature": 0.7,
  "stream": true,
  "n_parallel_rollouts": 8,
  "verifier_threshold": 0.75
}
```

### 2.3 Server-Sent Events (SSE) Streaming Response
```text
data: {"id":"chatcmpl-ag123","object":"chat.completion.chunk","created":1785000000,"model":"antigravity-1.5b","choices":[{"index":0,"delta":{"content":"To solve "},"finish_reason":null}]}

data: {"id":"chatcmpl-ag123","object":"chat.completion.chunk","created":1785000000,"model":"antigravity-1.5b","choices":[{"index":0,"delta":{"content":"for x..."},"finish_reason":null}]}

data: [DONE]
```

---

## 3. Performance & Verification Benchmarking Harness

### 3.1 `benchmark.py` Automated Test Suite
The repository includes a comprehensive benchmark harness to profile and verify all claims:

```bash
python3 benchmark.py --device mac --batch_sizes 1,2,4,8,16 --model_path models/qwen2.5-1.5b-int4.bin
```

### 3.2 Key Metrics Recorded
1. **GEMM Acceleration Ratio:** $\frac{\text{Latency}(N \text{ sequential single decodes})}{\text{Latency}(\text{Batched } N \text{ parallel decode})}$. (Target: $\ge 3.0\times$).
2. **Accuracy Gain:** Solution correctness rate on GSM8K/MATH standard benchmark sets vs. 1-pass greedy decode (Target: $+15\%$ absolute gain).
3. **Memory Footprint Peak:** Measured peak allocation in MB (Target: $< 4500 \text{ MB}$).
4. **Token Reduction Ratio:** Measured token savings from adaptive reflection (Target: $> 35\%$).
