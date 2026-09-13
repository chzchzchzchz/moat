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
        riskAssessment: String = "No immediate self-harm or acute safety risks reported.",
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
        \(riskAssessment)

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
        var risk = "No acute risk factors identified during session."

        let sections = output.components(separatedBy: "[")
        
        for section in sections {
            let trimmed = section.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasPrefix("SUBJECTIVE]") {
                subjective = trimmed.replacingOccurrences(of: "SUBJECTIVE]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("OBJECTIVE]") {
                objective = trimmed.replacingOccurrences(of: "OBJECTIVE]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("ASSESSMENT]") {
                assessment = trimmed.replacingOccurrences(of: "ASSESSMENT]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("PLAN]") {
                plan = trimmed.replacingOccurrences(of: "PLAN]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            } else if trimmed.hasPrefix("SYMPTOMS]") {
                let text = trimmed.replacingOccurrences(of: "SYMPTOMS]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                symptoms = text.split(separator: ",").map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            } else if trimmed.hasPrefix("RISK]") {
                risk = trimmed.replacingOccurrences(of: "RISK]", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        // Fallback if model did not use exact [SECTION] headers
        if subjective.isEmpty && objective.isEmpty && assessment.isEmpty && plan.isEmpty {
            subjective = output
            objective = "Client engaged in session via local transcription interface."
            assessment = "Session review documented."
            plan = "Continue treatment plan as scheduled."
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
