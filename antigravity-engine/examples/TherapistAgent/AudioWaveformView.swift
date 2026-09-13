import SwiftUI

/// Live pulsating audio waveform visualizer for clinical dictation
public struct AudioWaveformView: View {
    public let levels: [Float]
    public let isRecording: Bool

    public init(levels: [Float], isRecording: Bool) {
        self.levels = levels
        self.isRecording = isRecording
    }

    public var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<levels.count, id: \.self) { idx in
                let normalized = CGFloat(levels[idx])
                let barHeight = max(4.0, normalized * 38.0)

                RoundedRectangle(cornerRadius: 2)
                    .fill(
                        LinearGradient(
                            gradient: Gradient(colors: isRecording ? [.red.opacity(0.8), .orange] : [.teal.opacity(0.6), .teal]),
                            startPoint: .bottom,
                            endPoint: .top
                        )
                    )
                    .frame(width: 4, height: barHeight)
                    .animation(.easeInOut(duration: 0.1), value: normalized)
            }
        }
        .frame(height: 42)
        .padding(.horizontal, 8)
        .background(Color.secondary.opacity(0.06))
        .cornerRadius(8)
    }
}
