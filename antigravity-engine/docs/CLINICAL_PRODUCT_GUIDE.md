# Antigravity Clinical Assistant: Turnkey Therapy Documentation
### Zero-Cloud Clinical Documentation Product Manual (v2.5.0)

---

## 1. Product Overview

The Antigravity Clinical Assistant is a native macOS and iOS application created specifically for independent psychotherapists, clinical psychologists, and psychiatric practices who require 100% private, zero-cloud medical documentation.

### Core Value Proposition
- **Guaranteed Privacy Immunity**: All speech-to-text, clinical reasoning, and database storage occur on the device's Apple Silicon neural engine and GPU. Zero bytes are transmitted to any cloud server. No Business Associate Agreement (BAA) is required with third-party cloud AI vendors because no data ever traverses the internet.
- **Biometric Class A Hardware Encryption**: Records are encrypted with hardware-backed AES-256 keys derived through the Apple Keychain Data Protection (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`) and require Face ID / Touch ID authentication.
- **Longitudinal Memory**: Built-in encrypted SQLite database tracks cross-session patient progress, historical assessment plans, and standardized psychometric metrics (PHQ-9, GAD-7).
- **One-Click Vector PDF Export**: Instantly exports professional clinical documentation designed with HIPAA considerations ready for electronic health record (EHR) upload or physical filing.

---

## 2. Feature Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ANTIGRAVITY CLINICAL VAULT                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│     Audio Capture     │ │   Clinical Engine     │ │   Encrypted Storage   │
├───────────────────────┤ ├───────────────────────┤ ├───────────────────────┤
│ • On-device Apple ASR │ │ • Edge LLM Reasoning │ │ • SQLite Database     │
│ • vDSP Audio Waveform │ │ • SOAP, DAP, Intake   │ │ • Keychain Data Key   │
│ • Diarization Heuristic││ • Section Extraction  │ │ • Biometric Lock      │
│ • 0 Network Outbound  │ │ • Psychometric Metrics│ │ • Vector PDF Export   │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
```

---

## 3. Clinical Workflow

### Step 1: Patient Selection & Template Setup
Select the target template format from the picker:
- **SOAP Note**: Standard Subjective, Objective, Assessment, Plan documentation for ongoing therapy.
- **Intake Assessment**: Comprehensive biopsychosocial diagnostic evaluation.
- **Progress Note**: Targeted session-by-session milestone tracking.
- **DAP Note**: Data, Assessment, Plan notation for counseling.

### Step 2: Live Dictation & Waveform Feedback
Press **Record Live** to dictate the session dialogue or therapist session notes. The real-time audio waveform monitors mic input power. Transcription runs on-device via Apple's neural speech engine.

### Step 3: Offline Clinical Document Generation
Click **Generate Clinical Document (Offline)**. Antigravity retrieves the patient's prior 3 session assessments and treatment plans from the local encrypted database to ensure continuity of care, passes the context to the on-device model, and parses the output into structured clinical sections:
- `[SUBJECTIVE]`: Patient reported distress, precipitants, and symptoms.
- `[OBJECTIVE]`: Mental status exam, clinical presentation, motor tone, and affect.
- `[ASSESSMENT]`: Diagnostic impression, ICD-10 categorization, and clinical rationale.
- `[PLAN]`: Targeted interventions, homework assignments, and follow-up cadence.
- `[SYMPTOMS]`: Normalized list of identified symptoms.
- `[RISK]`: Explicit suicide/homicide risk stratification.

### Step 4: Storage & PDF Export
The session is automatically saved to the biometric-gated SQLite vault. Clicking **Export Encrypted PDF** renders a publication-grade, vector clinical report complete with clinic headers and therapist signature lines.

---

## 4. Privacy & Compliance Rationale

| Regulatory Framework | Cloud AI Providers (e.g. OpenAI, Anthropic) | Antigravity Engine Edge Architecture |
| :--- | :--- | :--- |
| **HIPAA Considerations** | Requires signed BAA; potential audit risk on third-party server breaches. | **Exempt from third-party BAA**: 0 WAN data transfer. Data remains on clinician hardware. |
| **Data at Rest** | Stored on multi-tenant vendor cloud databases. | **Class A Keychain Data Protection**: AES-256 keys locked behind Face ID / Touch ID. |
| **Audio Privacy** | Audio recordings uploaded and processed remotely. | **Apple On-Device Speech API**: Zero audio transmission off device. |
| **Availability** | Subject to internet outages and cloud downtime. | **No outbound network calls during inference**: Runs on planes, remote clinics, or during power/network outages. |
