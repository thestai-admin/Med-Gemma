# PrimaCare AI - Video Script (All 4 Tracks)

## Duration
3:00 target

## 0:00-0:20 Problem Statement
"Primary care clinicians face three challenges: delayed radiology support for chest X-rays, a health literacy gap affecting 36% of US adults, and lack of AI infrastructure in resource-limited settings. PrimaCare AI addresses all three with a unified system built on MedGemma and MedSigLIP."

## 0:20-0:50 Architecture
Show 5-agent pipeline diagram:
- IntakeAgent -> ImagingAgent -> ReasoningAgent -> GuidelinesAgent -> EducationAgent
- Orchestrator with lazy loading, profiling, parallel execution, fast mode

Show tiered deployment diagram:
- Edge (CPU): MedSigLIP ONNX INT8 for pneumonia screening
- Cloud (GPU): Full 5-agent pipeline with MedGemma

Narration:
"Five agents coordinate through a central orchestrator. For resource-limited settings, a quantized edge classifier provides CPU-only pneumonia screening, triaging cases that need the full cloud pipeline."

## 0:50-1:30 Demo: Full Pipeline
- Enter clinical history: "65yo male smoker, cough for 2 weeks with fever"
- Upload chest X-ray
- Run full pipeline with `include_education=True`
- Show output sections: structured HPI, imaging analysis, differential diagnosis, guideline recommendations

Narration:
"Each stage is explicit and reviewable. The orchestrator chains structured data between agents - patient context flows to imaging, imaging flows to reasoning, reasoning to guidelines."

## 1:30-1:55 Demo: Patient Education (Novel Task)
- Show education output at basic level: simplified diagnosis, what it means, next steps, glossary
- Switch to detailed level to show the contrast
- Highlight glossary of medical terms

Narration:
"The PatientEducationAgent converts the technical report into language a 6th grader can understand. Three reading levels match diverse health literacy needs. The glossary defines every medical term used."

## 1:55-2:20 Demo: Edge AI
- Show edge classifier running on CPU
- Display classification result: normal vs pneumonia with probabilities
- Show benchmark comparison table: GPU vs Edge (latency, model size, accuracy)

Narration:
"For clinics without GPU infrastructure, we export MedSigLIP to ONNX and quantize to INT8. The edge classifier runs on any CPU, providing fast pneumonia screening that triages cases to the full pipeline."

## 2:20-2:45 Metrics
Display results table:

Binary pneumonia (100 samples):
- Accuracy 76.0%, Recall 98.0%, F1 0.803

Pipeline latency (Kaggle T4):
- Total 111s, suitable for asynchronous decision support

Test coverage:
- 42 tests passing, all using mocks (no GPU needed)

Narration:
"Binary mode achieves 98% recall for pneumonia detection. We include threshold sweep utilities and bootstrap confidence intervals for transparent evaluation."

## 2:45-3:00 Close
"PrimaCare AI is clinician decision support, not autonomous diagnosis. Our contribution is a reproducible 5-agent CXR workflow with patient education for health literacy, edge deployment for resource equity, and transparent evaluation with measurable performance. All code is public."

## Recording Checklist
- [ ] Run notebook with education enabled, capture all 5 stages
- [ ] Show education output at basic and detailed levels
- [ ] Show edge classifier output and benchmark table
- [ ] Display final metrics table from notebook
- [ ] Keep all statements aligned with displayed numbers
- [ ] Total time under 3:00
