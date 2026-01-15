# PrimaCare AI - Video Demo Script (3 Minutes)

## Overview
Total Duration: 3:00
Format: Screen recording with voiceover

---

## SECTION 1: Problem Statement (0:00 - 0:30)

### Visuals
- Title slide: "PrimaCare AI: Multimodal Diagnostic Support"
- Statistics slide with key numbers

### Script
> "Every year, over 500 million patients visit primary care physicians in the United States. These physicians are expected to diagnose conditions across every organ system - but they have only 15 to 20 minutes per patient.
>
> Chest X-rays are one of the most common tests ordered in primary care, but many physicians don't have immediate radiology support. Studies show interpretation errors can reach 20-30%.
>
> Today I'll show you PrimaCare AI - a multi-agent diagnostic support system built on MedGemma that helps primary care physicians make faster, more accurate diagnoses."

---

## SECTION 2: Architecture Overview (0:30 - 1:00)

### Visuals
- Architecture diagram showing 4 agents
- Flow: Patient → IntakeAgent → ImagingAgent → ReasoningAgent → Output

### Script
> "PrimaCare AI uses four specialized agents that mirror how clinicians actually think.
>
> First, the IntakeAgent structures the patient's story - identifying red flags and organizing symptoms into a formal history.
>
> Second, the ImagingAgent uses MedGemma 1.5 4B to analyze chest X-rays - it provides both zero-shot classification with MedSigLIP and detailed findings with MedGemma.
>
> Third, the ReasoningAgent integrates everything - generating a differential diagnosis and recommending next steps.
>
> Finally, an Orchestrator coordinates everything, whether you have just text, just an image, or both."

---

## SECTION 3: Live Demo (1:00 - 2:30)

### Visuals
- Gradio interface
- Upload image, enter text, show results

### Demo Flow

**Step 1: Launch Demo (1:00 - 1:10)**
> "Let's see it in action. Here's our Gradio interface deployed from Kaggle."

**Step 2: Enter Clinical Context (1:10 - 1:30)**
> "I'll enter a typical primary care scenario: A 58-year-old male smoker presenting with two weeks of productive cough and mild shortness of breath."

*Type:* "58 year old male, 30 pack-year smoking history, presenting with productive cough for 2 weeks, mild dyspnea on exertion, no fever"

**Step 3: Upload X-ray (1:30 - 1:45)**
> "Now I'll upload a chest X-ray for analysis."

*Upload sample chest X-ray*

**Step 4: Run Analysis (1:45 - 2:00)**
> "Let's run the full consultation."

*Click analyze button*

**Step 5: Show Results (2:00 - 2:30)**
> "Look at the output - the IntakeAgent has structured the history with red flags highlighted.
>
> The ImagingAgent shows the MedSigLIP classification - and here are the detailed findings from MedGemma.
>
> The ReasoningAgent synthesizes everything into a ranked differential diagnosis with recommended workup - labs, additional imaging, and specialist referral if needed.
>
> This entire analysis took less than a minute."

---

## SECTION 4: Impact & Closing (2:30 - 3:00)

### Visuals
- Impact statistics
- GitHub link
- Closing slide

### Script
> "PrimaCare AI can save physicians over an hour per day by automating history structuring and providing instant X-ray analysis.
>
> It's particularly impactful in resource-limited settings where radiology support isn't immediately available.
>
> The system runs entirely on Kaggle's free T4 GPU, making it accessible to anyone.
>
> All code is available on GitHub. Thank you for watching - let's bring AI-powered diagnostic support to the frontlines of healthcare."

*Show GitHub URL: github.com/thestai-admin/Med-Gemma*

---

## Recording Checklist

- [ ] Test Gradio demo is working
- [ ] Prepare sample chest X-ray image
- [ ] Practice voiceover timing
- [ ] Record screen at 1080p
- [ ] Check audio quality
- [ ] Keep under 3:00

## Technical Notes

- Use 04-agentic-workflow.ipynb to launch Gradio demo
- Ensure Kaggle GPU is enabled
- Have backup screenshots in case of issues
- Consider recording in segments and editing together
