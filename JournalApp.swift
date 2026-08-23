import SwiftUI
import AppKit

struct ContentView: View {
    @State private var entryText: String = "I've been feeling really overwhelmed with work lately. I keep trying to organize my tasks but by the end of the day, I feel like I accomplished nothing. I'm just exhausted."
    @State private var insight: String = ""
    @State private var isAnalyzing: Bool = false
    @State private var analysisTime: String = ""

    var body: some View {
        VStack(spacing: 20) {
            Text("Private Edge Journal")
                .font(.largeTitle)
                .bold()
                .padding(.top)

            VStack(alignment: .leading) {
                Text("New Entry")
                    .font(.headline)
                    .foregroundColor(.gray)
                
                TextEditor(text: $entryText)
                    .padding(4)
                    .background(Color.white)
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.gray.opacity(0.3)))
                    .frame(height: 150)
            }
            .padding(.horizontal)

            Button(action: analyzeEntry) {
                if isAnalyzing {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                } else {
                    Text("🧠 Reflect (Local Edge AI)")
                        .bold()
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.purple)
            .foregroundColor(.white)
            .cornerRadius(12)
            .padding(.horizontal)
            .disabled(isAnalyzing || entryText.isEmpty)

            VStack(alignment: .leading) {
                Text("AI Insight (100% Private)")
                    .font(.headline)
                    .foregroundColor(.gray)
                
                ScrollView {
                    Text(insight.isEmpty ? "Your insight will appear here. No data ever leaves your device." : insight)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.purple.opacity(0.1))
                        .cornerRadius(8)
                }
                .frame(height: 150)
                
                if !analysisTime.isEmpty {
                    Text(analysisTime)
                        .font(.caption)
                        .foregroundColor(.green)
                }
            }
            .padding(.horizontal)
            
            Spacer()
        }
        .frame(width: 450, height: 600)
        .background(Color(NSColor.windowBackgroundColor))
        .onAppear { analyzeEntry() }
    }

    func analyzeEntry() {
        isAnalyzing = true
        insight = "Generating multiple thoughts and scoring them locally..."
        let startTime = Date()
        
        let url = URL(string: "http://127.0.0.1:8000/reflect")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let payload: [String: String] = ["entry": entryText]
        request.httpBody = try? JSONEncoder().encode(payload)
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                isAnalyzing = false
                if let error = error {
                    insight = "Error: \(error.localizedDescription)"
                    return
                }
                
                if let data = data,
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let reflection = json["reflection"] as? String {
                    let elapsed = Date().timeIntervalSince(startTime)
                    self.insight = reflection
                    self.analysisTime = String(format: "Generated and scored 4 trajectories in %.1f seconds using Qwen3.5 4B.", elapsed)
                } else {
                    insight = "Failed to parse response."
                }
            }
        }.resume()
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        let contentView = ContentView()
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 450, height: 600),
            styleMask: [.titled, .closable, .miniaturizable, .fullSizeContentView],
            backing: .buffered, defer: false)
        window.center()
        window.title = "Private Edge Journal"
        window.contentView = NSHostingView(rootView: contentView)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

let app = NSApplication.shared
NSApp.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
app.run()
