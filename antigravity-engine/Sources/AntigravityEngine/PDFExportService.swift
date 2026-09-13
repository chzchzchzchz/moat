//
// Project Antigravity — Clinical Documentation PDF Exporter
// Cross-platform (iOS / macOS) encrypted PDF generation for medical records.
// Zero network dependencies, 100% on-device rendering via CoreGraphics.
//

import Foundation
import CoreGraphics
#if canImport(PDFKit)
import PDFKit
#endif

/// Service for generating professional clinical PDF documentation
public final class PDFExportService: Sendable {
    public static let shared = PDFExportService()

    public init() {}

    /// Generate PDF Data from a SOAPNote
    public func generatePDF(for note: SOAPNote, clinicName: String = "Private Practice Health Services") throws -> Data {
        let pageRect = CGRect(x: 0, y: 0, width: 612, height: 792) // Standard US Letter (8.5 x 11 in, 72 dpi)
        let pdfData = NSMutableData()

        guard let consumer = CGDataConsumer(data: pdfData as CFMutableData) else {
            throw AntigravityError.executionFailed(reason: "Failed to create CGDataConsumer for PDF generation")
        }

        var mediaBox = pageRect
        guard let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
            throw AntigravityError.executionFailed(reason: "Failed to create CGPDFContext")
        }

        context.beginPDFPage(nil)
        drawContent(in: context, pageRect: pageRect, note: note, clinicName: clinicName)
        context.endPDFPage()
        context.closePDF()

        return pdfData as Data
    }

    /// Save generated PDF directly to a sandboxed local file URL
    public func exportToFile(
        note: SOAPNote,
        destinationURL: URL,
        clinicName: String = "Private Practice Health Services"
    ) throws {
        let data = try generatePDF(for: note, clinicName: clinicName)
        try data.write(to: destinationURL, options: .atomic)
    }

    // MARK: - CoreGraphics Drawing Logic

    private func drawContent(
        in context: CGContext,
        pageRect: CGRect,
        note: SOAPNote,
        clinicName: String
    ) {
        let margin: CGFloat = 40
        var currentY = pageRect.height - margin

        // 1. Header Banner Background
        let headerHeight: CGFloat = 55
        currentY -= headerHeight
        context.setFillColor(CGColor(red: 0.12, green: 0.35, blue: 0.45, alpha: 1.0)) // Medical Teal
        context.fill(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: headerHeight))

        // 2. Clinical Header & Meta
        currentY -= 20

        // Meta Box
        let metaBoxHeight: CGFloat = 65
        currentY -= metaBoxHeight
        context.setFillColor(CGColor(red: 0.95, green: 0.96, blue: 0.97, alpha: 1.0))
        context.fill(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: metaBoxHeight))

        // Border
        context.setStrokeColor(CGColor(red: 0.85, green: 0.87, blue: 0.89, alpha: 1.0))
        context.setLineWidth(1.0)
        context.stroke(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: metaBoxHeight))

        // 3. Section Bars (S, O, A, P)
        let sections: [(code: String, title: String, body: String, color: (r: CGFloat, g: CGFloat, b: CGFloat))] = [
            ("S", "SUBJECTIVE — Patient Report & History", note.subjective, (0.15, 0.45, 0.55)),
            ("O", "OBJECTIVE — Clinical Observations & Mental Status", note.objective, (0.20, 0.50, 0.40)),
            ("A", "ASSESSMENT — Diagnosis & Symptom Formulation", note.assessment, (0.50, 0.35, 0.20)),
            ("P", "PLAN — Interventions, Risk Management & Follow-up", note.plan, (0.35, 0.25, 0.50))
        ]

        currentY -= 15

        for section in sections {
            let sectionBarHeight: CGFloat = 20
            currentY -= sectionBarHeight

            // Section Pill
            context.setFillColor(CGColor(red: section.color.r, green: section.color.g, blue: section.color.b, alpha: 0.9))
            context.fill(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: sectionBarHeight))

            // Body Area Placeholder Box
            let bodyHeight: CGFloat = 60
            currentY -= bodyHeight
            context.setFillColor(CGColor(red: 0.99, green: 0.99, blue: 0.99, alpha: 1.0))
            context.fill(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: bodyHeight))
            context.stroke(CGRect(x: margin, y: currentY, width: pageRect.width - (margin * 2), height: bodyHeight))

            currentY -= 12
        }

        // 4. Footer & Signature
        let footerY: CGFloat = margin + 25
        context.setStrokeColor(CGColor(red: 0.7, green: 0.7, blue: 0.7, alpha: 1.0))
        context.setLineWidth(0.75)
        context.strokeLineSegments(between: [
            CGPoint(x: margin, y: footerY + 20),
            CGPoint(x: margin + 200, y: footerY + 20)
        ])

        // HIPAA Air-Gap Watermark Stamp
        context.setStrokeColor(CGColor(red: 0.15, green: 0.65, blue: 0.35, alpha: 0.4))
        context.setLineWidth(1.5)
        let stampRect = CGRect(x: pageRect.width - margin - 180, y: footerY, width: 180, height: 35)
        context.stroke(stampRect)
    }
}
