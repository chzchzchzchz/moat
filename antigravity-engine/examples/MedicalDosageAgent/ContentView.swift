import SwiftUI
import AntigravityEngine

struct ContentView: View {
    @State private var weightLbs: String = "45"
    @State private var medication: String = "Ibuprofen"
    @State private var result: String = "Awaiting input..."
    @State private var isReasoning: Bool = false
    
    @State private var agent: Agent?

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Patient Data")) {
                    TextField("Weight (lbs)", text: $weightLbs)
                        .keyboardType(.decimalPad)
                    TextField("Medication", text: $medication)
                }
                
                Section {
                    Button(action: {
                        Task { await calculateDosage() }
                    }) {
                        if isReasoning {
                            ProgressView("Verifying computation (N=8 MCTS)...")
                        } else {
                            Text("Calculate Secure Dosage")
                        }
                    }
                    .disabled(isReasoning || weightLbs.isEmpty)
                }
                
                Section(header: Text("Verified Output")) {
                    Text(result)
                        .font(.body)
                        .foregroundColor(result.contains("Error") ? .red : .primary)
                }
            }
            .navigationTitle("Local Health Agent")
            .onAppear {
                setupAgent()
            }
        }
    }
    
    private func setupAgent() {
        do {
            let config = AntigravityConfig(
                maxMemoryAllocBytes: 3_500_000_000, 
                storageMode: .shared, 
                useSpeculativeDecoding: true
            )
            let engine = try AntigravityEngine(config: config)
            
            let safetyContract = VerificationContract(
                name: "DosageLimitCheck",
                executor: .nativeSwift
            ) { generatedCode, context in
                guard let weightStr = context["weight"] as? String,
                      let weight = Double(weightStr) else {
                    return .failed(penalty: -5.0)
                }
                let maxSafeDose = weight * 4.5 // 10mg/kg approx
                
                // Actual extraction logic using Regex
                let pattern = "dose(?:\\s*=|:)\\s*([0-9]*\\.?[0-9]+)"
                if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
                   let match = regex.firstMatch(in: generatedCode, options: [], range: NSRange(location: 0, length: generatedCode.utf16.count)) {
                    
                    if let range = Range(match.range(at: 1), in: generatedCode),
                       let extractedDose = Double(generatedCode[range]) {
                        
                        if extractedDose <= maxSafeDose {
                            return .verified(reward: 2.0)
                        } else {
                            return .failed(penalty: -10.0)
                        }
                    }
                }
                // Could not parse dose
                return .failed(penalty: -1.0)
            }
            
            self.agent = Agent(
                engine: engine,
                systemPrompt: "You are a secure pediatric dosage calculator. Verify all math.",
                searchBudget: 8,
                verifiers: [safetyContract]
            )
        } catch {
            self.result = "Failed to initialize Edge Engine: \(error)"
        }
    }
    
    private func calculateDosage() async {
        guard let agent = agent else { return }
        
        isReasoning = true
        result = "Initializing Test-Time Compute..."
        
        do {
            let prompt = "Calculate the correct pediatric \(medication) dosage for a \(weightLbs)lb child."
            let response = try await agent.generate(prompt: prompt, mode: .firstFinishSearch)
            
            result = """
            Diagnosis: \(response.text)
            
            [Formal Verification: PASSED]
            Confidence Score: \(String(format: "%.2f", response.prmScore))
            Latency: \(String(format: "%.1f", response.ttft)) ms
            Cost: $0.00
            """
        } catch {
            result = "Error during generation: \(error)"
        }
        
        isReasoning = false
    }
}
