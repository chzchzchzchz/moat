# Architectural Document 06: Double-Blind Security & Cryptographic Isolation

## 1. Network Air-Gapping (OS Entitlement Lock)

### 1.1 Info.plist Network Kill Switch
Strip ALL outbound network capabilities at the compiler level:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <false/>
    <!-- No exceptions. Zero external domains. -->
</dict>
```

Even if a third-party SDK or Apple background process attempts to phone home from within the app sandbox, the iOS kernel kills the network request. This provides **mathematically verifiable offline execution**.

---

## 2. Kill the HTTP Localhost — Use Secure IPC Instead

### 2.1 Why localhost:8080 is Dangerous
Any malicious app on the device can port-scan localhost and intercept unencrypted HTTP traffic. This is unacceptable for Apple-acquisition-grade security.

### 2.2 iOS Production Architecture: App Extension + Encrypted Shared Container
- Build Antigravity as an **App Extension** (custom keyboard, Share extension, or App Intent).
- Use **App Groups** with encrypted shared containers for IPC.
- Flow:
  1. Client app drops encrypted prompt into shared container.
  2. Antigravity engine wakes, processes, drops encrypted result back.
  3. Client app reads and decrypts.

### 2.3 macOS Development: XPC Service
- Use Apple's native **XPC Service** for inter-process communication.
- XPC verifies the cryptographic signature of the requesting app before accepting connections.

---

## 3. Secure Enclave Prompt Encryption

### 3.1 Asymmetric Keypair Generation
```
Antigravity Engine ──► Secure Enclave ──► Generate P256 Keypair
                                          ├── Public Key (shared with client apps)
                                          └── Private Key (NEVER leaves Secure Enclave)
```

### 3.2 Encryption Flow
1. Client app encrypts prompt using Antigravity's **public key**.
2. Encrypted prompt sent to Antigravity via App Group shared container.
3. Secure Enclave decrypts prompt directly into **volatile RAM** (never touches NAND flash).
4. Generated output encrypted using client app's public key before return.
5. **Plaintext prompt and response NEVER touch persistent storage.**

---

## 4. Metal Buffer Sanitization

### 4.1 Post-Inference Memory Scrubbing
The moment the verifier selects the Best-of-N output and encrypts it:

```swift
// Immediately zero-fill all candidate Metal buffers
func sanitizeMetalBuffers(_ buffers: [MTLBuffer]) {
    for buffer in buffers {
        let ptr = buffer.contents()
        memset(ptr, 0, buffer.length)
        // Force memory barrier to prevent compiler optimization of the zeroing
        __asm__ __volatile__("" ::: "memory")
    }
}
```

**Mandatory scrub targets:**
- All N candidate trace Metal buffers
- KV-cache Metal buffers for all rollout channels
- Intermediate attention score buffers
- Dequantized weight activation buffers (transient)
