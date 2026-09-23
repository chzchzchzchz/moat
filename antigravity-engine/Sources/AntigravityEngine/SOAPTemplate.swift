//
// Project Antigravity — Clinical Documentation & SOAP Note Engine
// Standardized on-device medical note formatting, extraction, and template parsing.
// Designed for HIPAA-compliant, air-gapped clinical documentation.
//

import Foundation

/// Standard clinical documentation template types
public enum ClinicalTemplateType: String, CaseIterable, Sendable, Codable {
    case soap = "SOAP Note"
    case intake = "Intake Assessment"
    case progress = "Progress Note"
    case dap = "DAP Note (Data, Assessment, Plan)"

    public var description: String {
        switch self {
        case .soap:
            return "Standard Subjective, Objective, Assessment, Plan format."
        case .intake:
            return "Comprehensive diagnostic intake with symptom inventory and psychiatric history."
        case .progress:
            return "Session-by-session milestone tracking and intervention updates."
        case .dap:
            return "Data, Assessment, and Plan notation for psychotherapy and counseling."
        }
    }
}

/// Structured representation of an encrypted clinical note
public struct SOAPNote: Sendable, Codable, Identifiable {
    public let id: UUID
    public var patientIdentifier: String
    public var sessionDate: Date
    public var clinicianName: String
    public var templateType: ClinicalTemplateType
    
    // Core SOAP Sections
    public var subjective: String
    public var objective: String
    public var assessment: String
    public var plan: String
    
    // Auxiliary Clinical Metadata
    public var identifiedSymptoms: [String]
    public var riskAssessment: String
    public var diagnosticImpressions: [String]
    public var billingCodes: [String]
    public var rawGeneratedText: String

    public init(
        id: UUID = UUID(),
        patientIdentifier: String = "Anonymous Patient",
        sessionDate: Date = Date(),
        clinicianName: String = "Attending Clinician",
        templateType: ClinicalTemplateType = .soap,
        subjective: String = "",
        objective: String = "",
        assessment: String = "",
        plan: String = "",
        identifiedSymptoms: [String] = [],
        // A missing risk evaluation means risk was NOT assessed, which is not the same as
        // risk being absent. Defaulting to a negative finding would place an unverified
        // "no self-harm risk" assertion into a clinical record.
        riskAssessment: String = "NOT ASSESSED — no risk evaluation was produced for this session. Requires clinician review.",
        diagnosticImpressions: [String] = [],
        billingCodes: [String] = ["90837 - Psychotherapy 53+ min"],
        rawGeneratedText: String = ""
    ) {
        self.id = id
        self.patientIdentifier = patientIdentifier
        self.sessionDate = sessionDate
        self.clinicianName = clinicianName
        self.templateType = templateType
        self.subjective = subjective
        self.objective = objective
        self.assessment = assessment
        self.plan = plan
        self.identifiedSymptoms = identifiedSymptoms
        self.riskAssessment = riskAssessment
        self.diagnosticImpressions = diagnosticImpressions
        self.billingCodes = billingCodes
        self.rawGeneratedText = rawGeneratedText
    }

    /// Formats note into professional markdown / text document
    public var formattedReport: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short

        var symptomsText = "None explicitly documented"
        if !identifiedSymptoms.isEmpty {
            symptomsText = identifiedSymptoms.map { "  • " + $0 }.joined(separator: "\n")
        }

        return """
        ================================================================================
        CLINICAL PSYCHOTHERAPY DOCUMENTATION — STRICTLY CONFIDENTIAL
        ================================================================================
        Patient Identifier : \(patientIdentifier)
        Date & Time        : \(formatter.string(from: sessionDate))
        Clinician          : \(clinicianName)
        Template           : \(templateType.rawValue)
        Billing Codes      : \(billingCodes.joined(separator: ", "))
        --------------------------------------------------------------------------------
        
        [S] SUBJECTIVE:
        \(subjective.trimmingCharacters(in: .whitespacesAndNewlines))

        [O] OBJECTIVE:
        \(objective.trimmingCharacters(in: .whitespacesAndNewlines))

        [A] ASSESSMENT:
        \(assessment.trimmingCharacters(in: .whitespacesAndNewlines))
        
        Identified Symptoms:
        \(symptomsText)
        
        Risk Assessment:
        \(riskAssessment.trimmingCharacters(in: .whitespacesAndNewlines))

        [P] PLAN:
        \(plan.trimmingCharacters(in: .whitespacesAndNewlines))

        --------------------------------------------------------------------------------
        Digitally compiled on-device via Antigravity Engine (Zero Cloud Egress)
        Electronically signed: \(clinicianName)
        ================================================================================
        """
    }
}

/// Clinical Note Generation & Parsing Engine
public struct SOAPTemplateEngine: Sendable {
    public init() {}

    /// Construct structured reasoning prompt for the model
    public func buildPrompt(
        transcript: String,
        patientId: String = "Patient",
        template: ClinicalTemplateType = .soap
    ) -> String {
        return """
        You are a clinical documentation specialist. Below is a verbatim psychotherapy session transcript.
        Transform this dialogue into an objective, professional, HIPAA-compliant \(template.rawValue).
        
        Structure your response with the following explicit sections:
        [SUBJECTIVE]
        Detailed report of the patient's self-reported feelings, stressors, sleep patterns, somatic complaints, and recent life events.
        
        [OBJECTIVE]
        Clinician observations regarding client affect, cognitive flow, verbal tone, thought process, and engagement.
        
        [ASSESSMENT]
        Clinical synthesis of symptoms, underlying psychological patterns, stress reactivity, and formal symptom inventory.
        
        [PLAN]
        Actionable interventions, homework/coping strategies assigned, frequency of continued therapy, and risk mitigation plan.
        
        [SYMPTOMS]
        Comma-separated list of identified clinical symptoms.
        
        [RISK]
        Direct evaluation of self-harm, suicidality, or homicidal ideation.
        
        Transcript:
        \(transcript)
        """
    }

    /// Parses raw model generation text into a structured SOAPNote
    public func parseModelOutput(
        _ output: String,
        patientId: String = "Patient",
        clinician: String = "Attending Clinician",
        template: ClinicalTemplateType = .soap
    ) -> SOAPNote {
        var subjective = ""
        var objective = ""
        var assessment = ""
        var plan = ""
        var symptoms: [String] = []
        // Stays this way unless the model actually emitted a [RISK] section. See the
        // note on SOAPNote.init: absence of an evaluation is not a negative finding.
        var risk = "NOT ASSESSED — no risk evaluation was produced for this session. Requires clinician review."

        let pattern = #"(?i)\[(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|SYMPTOMS|RISK)\]\s*([\s\S]*?)(?=(?:\[(?:SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN|SYMPTOMS|RISK)\])|\z)"#
        
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let nsOutput = output as NSString
            let matches = regex.matches(in: output, options: [], range: NSRange(location: 0, length: nsOutput.length))
            
            for match in matches {
                guard match.numberOfRanges >= 3 else { continue }
                let tag = nsOutput.substring(with: match.range(at: 1)).uppercased()
                let content = nsOutput.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                
                switch tag {
                case "SUBJECTIVE":
                    if !content.isEmpty { subjective = content }
                case "OBJECTIVE":
                    if !content.isEmpty { objective = content }
                case "ASSESSMENT":
                    if !content.isEmpty { assessment = content }
                case "PLAN":
                    if !content.isEmpty { plan = content }
                case "SYMPTOMS":
                    symptoms = content.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
                case "RISK":
                    if !content.isEmpty { risk = content }
                default:
                    break
                }
            }
        }

        // Fallback: if model generated text without structured section headers,
        // provide honest pending-review defaults for unformatted sections
        if subjective.isEmpty && objective.isEmpty && assessment.isEmpty && plan.isEmpty {
            subjective = output.trimmingCharacters(in: .whitespacesAndNewlines)
            objective = "Clinical observations pending clinician review."
            assessment = "Clinical assessment pending clinician review."
            plan = "Treatment plan pending clinician review."
        }

        return SOAPNote(
            patientIdentifier: patientId,
            sessionDate: Date(),
            clinicianName: clinician,
            templateType: template,
            subjective: subjective,
            objective: objective,
            assessment: assessment,
            plan: plan,
            identifiedSymptoms: symptoms,
            riskAssessment: risk,
            rawGeneratedText: output
        )
    }
}
