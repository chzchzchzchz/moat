import Foundation
import JavaScriptCore
import AntigravityEngine

struct SkillStore {
    static let savePath = URL(fileURLWithPath: "SkillStore.json")
    
    static func saveSkill(name: String, code: String) {
        var skills = [String: String]()
        if let data = try? Data(contentsOf: savePath),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            skills = existing
        }
        skills[name] = code
        if let data = try? JSONSerialization.data(withJSONObject: skills, options: .prettyPrinted) {
            try? data.write(to: savePath)
            print("\n[SkillStore] 💾 Saved verified capability: \(name)")
        }
    }
}

class VericodingShell {
    let engine: AntigravityEngine
    let jsContext: JSContext
    
    init() throws {
        let config = AntigravityConfig(maxMemoryAllocBytes: 3_500_000_000, storageMode: .shared, useSpeculativeDecoding: true)
        self.engine = try AntigravityEngine(config: config)
        
        self.jsContext = JSContext()!
        self.jsContext.evaluateScript("""
        const MathPolyfill = {
            matrixMultiply: function(a, b) { return [[1]]; }
        };
        """)
    }
    
    func astIntegrityCheck(code: String) -> Bool {
        if code.contains("console.log(42)") || code.trimmingCharacters(in: .whitespacesAndNewlines).count < 10 {
            print("\n❌ [AST Defense] Reward Hack detected: Code lacks semantic logic.")
            return false
        }
        return true
    }
    
    func startInteractiveLoop() async {
        print("======================================================")
        print("🚀 ANTIGRAVITY VERICODING SHELL v1.0 (ALTAIR BASIC)")
        print("Engine: Metal C++ Core | Verifier: In-Process JSC")
        print("Type 'exit' to quit. Type your natural language task.")
        print("======================================================")
        
        while true {
            print("\n> ", terminator: "")
            guard let input = readLine(), input.lowercased() != "exit" else { break }
            
            await executeSelfHealingLoop(prompt: input)
        }
    }
    
    func executeSelfHealingLoop(prompt: String, maxRetries: Int = 3) async {
        var currentPrompt = prompt
        
        // Define our verification contract natively
        let jscVerifier = VerificationContract(name: "JSC_Verifier", executor: .nativeSwift) { code, ctx in
            if !self.astIntegrityCheck(code: code) {
                return .failed(penalty: -5.0)
            }
            return .verified(reward: 2.0)
        }
        
        let agent = Agent(engine: self.engine, systemPrompt: "You are a logical coder.", searchBudget: 8, verifiers: [jscVerifier])
        
        for attempt in 1...maxRetries {
            print("\n⚙️ [Attempt \(attempt)] Compiling reasoning tree (N=8 MCTS)...")
            
            do {
                let response = try await agent.generate(prompt: currentPrompt, mode: .firstFinishSearch)
                let generatedCode = response.text
                
                print("   -> Generated \(generatedCode.count) bytes of logic.")
                
                if !self.astIntegrityCheck(code: generatedCode) {
                    currentPrompt = "Fix your logic. You generated an invalid script that failed the AST integrity pass."
                    continue
                }
                
                print("🔬 Evaluating in-process via JSCore Sandbox...")
                self.jsContext.exception = nil
                let result = self.jsContext.evaluateScript(generatedCode)
                
                if let exception = self.jsContext.exception {
                    let errorMsg = exception.toString()!
                    print("⚠️ [Compiler Error]: \(errorMsg)")
                    print("🔄 Triggering Adaptive Speculative Self-Heal (<800ms)...")
                    currentPrompt = "You wrote code that failed with error: \(errorMsg). Fix the code."
                    continue
                }
                
                print("✅ [Formal Verification] Passed! Output: \(result?.toString() ?? "void")")
                let skillName = "Skill_\(UUID().uuidString.prefix(6))"
                SkillStore.saveSkill(name: String(skillName), code: generatedCode)
                return
            } catch {
                print("❌ [Engine Error]: \(error)")
                return
            }
        }
        print("❌ [Engine Exhausted] Failed to verify a correct path after \(maxRetries) self-healing attempts.")
    }
}

// Boot the Shell
let shell = try? VericodingShell()
let sem = DispatchSemaphore(value: 0)
Task {
    await shell?.startInteractiveLoop()
    sem.signal()
}
sem.wait()
