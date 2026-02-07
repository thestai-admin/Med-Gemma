"""
PrimaCare AI - Medical Imaging Analysis Demo

Interactive Gradio demo for the MedGemma Impact Challenge.
Showcases MedGemma's capabilities for primary care diagnostic support.

Features:
- Chest X-ray analysis with multiple prompt types
- Zero-shot classification with MedSigLIP
- Text-based medical Q&A
- Structured report generation

Run locally:
    python app/demo.py

Or on Colab/Kaggle:
    import gradio as gr
    from app.demo import create_demo
    demo = create_demo()
    demo.launch()
"""

import gradio as gr
from PIL import Image
from typing import Optional
import sys
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy loading for GPU environments
_model = None
_classifier = None
_orchestrator = None


def get_model():
    """Lazy load MedGemma model."""
    global _model
    if _model is None:
        from src.model import MedGemma
        print("Loading MedGemma model...")
        _model = MedGemma()
        print("Model loaded!")
    return _model


def get_classifier():
    """Lazy load MedSigLIP classifier."""
    global _classifier
    if _classifier is None:
        from src.model import MedSigLIP
        print("Loading MedSigLIP classifier...")
        _classifier = MedSigLIP()
        print("Classifier loaded!")
    return _classifier


def get_orchestrator():
    """Lazy load PrimaCare orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        from src.agents import PrimaCareOrchestrator
        print("Loading PrimaCare Orchestrator...")
        _orchestrator = PrimaCareOrchestrator()
        print("Orchestrator ready!")
    return _orchestrator


# =============================================================================
# New Feature Functions
# =============================================================================

def run_longitudinal_comparison(
    prior_image: Image.Image,
    current_image: Image.Image,
    clinical_context: str,
    interval: str,
) -> str:
    """Run longitudinal CXR comparison."""
    if prior_image is None or current_image is None:
        return "Please upload both prior and current images."

    orchestrator = get_orchestrator()

    try:
        result = orchestrator.run_longitudinal(
            prior_image=prior_image,
            current_image=current_image,
            clinical_context=clinical_context if clinical_context.strip() else None,
            interval=interval if interval.strip() else None,
        )
        return result.to_prompt_context()
    except Exception as e:
        return f"Error running longitudinal comparison: {str(e)}"


def run_volumetric_analysis(
    ct_files,
    modality: str,
    body_region: str,
    clinical_context: str,
) -> str:
    """Run volumetric CT/MRI analysis."""
    if ct_files is None or len(ct_files) == 0:
        return "Please upload CT/MRI slice images."

    orchestrator = get_orchestrator()

    try:
        from src.agents.volumetric import VolumetricModality

        # Load images from uploaded files
        images = []
        for f in ct_files:
            img = Image.open(f.name)
            images.append(img)

        # Map modality string to enum
        mod_enum = VolumetricModality.CT if modality == "CT" else VolumetricModality.MRI

        result = orchestrator.run_volumetric(
            volume=images,
            modality=mod_enum,
            clinical_context=clinical_context if clinical_context.strip() else None,
            body_region=body_region.lower(),
            num_slices=min(6, len(images)),
        )
        return result.to_prompt_context()
    except Exception as e:
        return f"Error running volumetric analysis: {str(e)}"


def run_ehr_query(
    question: str,
    fhir_json: str,
) -> tuple:
    """Query EHR data."""
    if not question.strip():
        return "Please enter a question.", ""

    if not fhir_json.strip():
        return "Please provide FHIR data (JSON format).", ""

    orchestrator = get_orchestrator()

    try:
        import json
        fhir_bundle = json.loads(fhir_json)

        result = orchestrator.query_ehr(
            question=question,
            fhir_bundle=fhir_bundle,
        )

        # Get patient summary
        summary = orchestrator.get_ehr_patient_summary(fhir_bundle)

        return result.to_prompt_context(), summary
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for FHIR data.", ""
    except Exception as e:
        return f"Error querying EHR: {str(e)}", ""


def run_pathology_analysis(
    image: Image.Image,
    tissue_type: str,
    clinical_context: str,
) -> str:
    """Run pathology image analysis."""
    if image is None:
        return "Please upload a pathology image."

    orchestrator = get_orchestrator()

    try:
        from src.agents.pathology import TissueType

        # Map tissue type string to enum
        tissue_map = {
            "Breast": TissueType.BREAST,
            "Lung": TissueType.LUNG,
            "Colon": TissueType.COLON,
            "Prostate": TissueType.PROSTATE,
            "Skin": TissueType.SKIN,
            "Liver": TissueType.LIVER,
            "Kidney": TissueType.KIDNEY,
            "Lymph Node": TissueType.LYMPH_NODE,
            "General": TissueType.GENERAL,
        }
        tissue_enum = tissue_map.get(tissue_type, TissueType.GENERAL)

        result = orchestrator.run_pathology(
            image_or_wsi=image,
            tissue_type=tissue_enum,
            clinical_context=clinical_context if clinical_context.strip() else None,
            is_wsi=False,
        )
        return result.to_prompt_context()
    except Exception as e:
        return f"Error running pathology analysis: {str(e)}"


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_xray(
    image: Image.Image,
    analysis_type: str,
    custom_prompt: Optional[str] = None,
) -> str:
    """Analyze a chest X-ray with MedGemma."""
    if image is None:
        return "Please upload an image first."

    model = get_model()

    # Select prompt based on analysis type
    prompts = {
        "Describe Findings": "Describe this chest X-ray in detail, including all visible findings.",
        "Bullet Point Findings": "List all findings in this chest X-ray in bullet points.",
        "Differential Diagnosis": "Based on this chest X-ray, provide a differential diagnosis with the most likely conditions.",
        "Structured Report": """Generate a structured radiology report for this chest X-ray:

**TECHNIQUE:**
[Describe imaging technique]

**FINDINGS:**
[Describe cardiac silhouette, lung fields, mediastinum, bones, soft tissues]

**IMPRESSION:**
[Provide concise summary]

**RECOMMENDATIONS:**
[Suggest follow-up if needed]""",
        "Primary Care Review": """As a primary care physician reviewing this X-ray, provide:

1. **Key Findings:** What abnormalities are visible?
2. **Differential Diagnosis:** Most likely conditions in order of probability
3. **Recommended Next Steps:** What follow-up is needed?
4. **Patient Communication:** How would you explain this to the patient?""",
        "Custom Prompt": custom_prompt or "Describe this medical image.",
    }

    prompt = prompts.get(analysis_type, prompts["Describe Findings"])

    try:
        response = model.analyze_image(image, prompt, max_new_tokens=2000)
        return response
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def classify_xray(image: Image.Image) -> str:
    """Zero-shot classification with MedSigLIP."""
    if image is None:
        return "Please upload an image first."

    classifier = get_classifier()

    labels = [
        "normal chest x-ray",
        "pneumonia",
        "pleural effusion",
        "cardiomegaly",
        "pulmonary edema",
        "atelectasis",
        "pneumothorax",
        "consolidation",
    ]

    try:
        results = classifier.classify(image, labels)

        # Format results
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        output = "**Classification Results:**\n\n"
        for label, prob in sorted_results:
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            output += f"{label}: {bar} {prob*100:.1f}%\n"

        return output
    except Exception as e:
        return f"Error classifying image: {str(e)}"


def answer_question(question: str) -> str:
    """Answer a medical question with MedGemma."""
    if not question.strip():
        return "Please enter a question."

    model = get_model()

    try:
        response = model.ask(question, max_new_tokens=1500)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def run_agentic_pipeline(
    chief_complaint: str,
    history: str,
    image: Optional[Image.Image],
    age: str,
    gender: str,
) -> tuple:
    """Run the full agentic diagnostic pipeline."""
    if not chief_complaint.strip():
        return "Please enter a chief complaint.", "", ""

    orchestrator = get_orchestrator()

    try:
        age_int = int(age) if age and age.strip() else None
    except ValueError:
        age_int = None

    gender_str = gender if gender and gender != "Unknown" else None

    try:
        result = orchestrator.run(
            chief_complaint=chief_complaint,
            history=history or "",
            xray_image=image,
            age=age_int,
            gender=gender_str,
        )

        # Format guidelines output
        guidelines_text = ""
        sources_text = ""

        if result.guidelines_result and result.guidelines_result.recommendations:
            gr_result = result.guidelines_result

            # Format recommendations
            guidelines_lines = ["**Evidence-Based Recommendations:**\n"]
            for i, rec in enumerate(gr_result.recommendations, 1):
                guidelines_lines.append(f"**{i}. {rec.recommendation}**")
                if rec.evidence_level:
                    guidelines_lines.append(f"   Evidence Level: {rec.evidence_level}")
                if rec.source_guidelines:
                    guidelines_lines.append(f"   Source: {rec.source_guidelines[0]}")
                guidelines_lines.append("")

            if gr_result.summary:
                guidelines_lines.append(f"**Summary:** {gr_result.summary}")

            guidelines_text = "\n".join(guidelines_lines)

            # Format sources
            if gr_result.retrieved_chunks:
                sources_lines = ["**Retrieved Guideline Sources:**\n"]
                seen = set()
                for chunk in gr_result.retrieved_chunks[:5]:
                    citation = chunk.to_citation()
                    if citation not in seen:
                        sources_lines.append(f"- {citation}")
                        sources_lines.append(f"  *{chunk.content[:200]}...*\n")
                        seen.add(citation)
                sources_text = "\n".join(sources_lines)

        return result.to_report(), guidelines_text, sources_text
    except Exception as e:
        return f"Error running pipeline: {str(e)}", "", ""


def generate_report(
    image: Image.Image,
    patient_age: str,
    patient_gender: str,
    clinical_history: str,
) -> str:
    """Generate a comprehensive radiology report."""
    if image is None:
        return "Please upload an image first."

    model = get_model()

    # Build context-aware prompt
    context_parts = []
    if patient_age:
        context_parts.append(f"Age: {patient_age}")
    if patient_gender:
        context_parts.append(f"Gender: {patient_gender}")
    if clinical_history:
        context_parts.append(f"Clinical History: {clinical_history}")

    context = "\n".join(context_parts) if context_parts else "No clinical context provided."

    prompt = f"""**Patient Information:**
{context}

Please analyze this chest X-ray and generate a comprehensive radiology report.

**TECHNIQUE:**
[Describe the imaging technique and quality]

**COMPARISON:**
[Note if prior studies are available for comparison]

**FINDINGS:**
Provide detailed findings for each anatomical region:
- Lungs and Airways
- Cardiac Silhouette
- Mediastinum
- Pleura
- Bones and Soft Tissues

**IMPRESSION:**
[Concise summary of key findings]

**DIFFERENTIAL DIAGNOSIS:**
[List possible diagnoses in order of likelihood]

**RECOMMENDATIONS:**
[Suggest follow-up imaging or clinical correlation as needed]"""

    try:
        response = model.analyze_image(image, prompt, max_new_tokens=2500)
        return response
    except Exception as e:
        return f"Error generating report: {str(e)}"


# =============================================================================
# Demo Interface
# =============================================================================

def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="PrimaCare AI - Medical Imaging Analysis",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown("""
        # PrimaCare AI: Multimodal Diagnostic Support

        **MedGemma Impact Challenge Demo** | *Agentic Workflow Prize Submission*

        Multi-agent diagnostic support system for primary care physicians.
        Combines patient history structuring, chest X-ray analysis, and clinical reasoning.

        > **Disclaimer:** This is a research demonstration. All outputs require clinical
        > verification by qualified healthcare professionals before any clinical use.
        """)

        with gr.Tabs():
            # Tab 0: Agentic Pipeline (Main Feature)
            with gr.TabItem("Agentic Pipeline"):
                gr.Markdown("""
                ### Multi-Agent Diagnostic Support

                This pipeline uses four AI agents working together:
                1. **Intake Agent** - Structures patient history into HPI format
                2. **Imaging Agent** - Analyzes chest X-ray (if provided)
                3. **Reasoning Agent** - Generates differential diagnosis and recommendations
                4. **Guidelines Agent** - Retrieves evidence-based clinical guidelines (RAG)
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        agent_complaint = gr.Textbox(
                            label="Chief Complaint",
                            placeholder="e.g., Cough for 2 weeks with fever",
                            lines=1,
                        )
                        agent_history = gr.Textbox(
                            label="Clinical History",
                            placeholder="e.g., 65yo male smoker. Productive cough with yellow sputum. Night sweats. 10lb weight loss.",
                            lines=4,
                        )
                        with gr.Row():
                            agent_age = gr.Textbox(
                                label="Age",
                                placeholder="65",
                                scale=1,
                            )
                            agent_gender = gr.Dropdown(
                                choices=["Male", "Female", "Other", "Unknown"],
                                value="Unknown",
                                label="Gender",
                                scale=1,
                            )
                        agent_image = gr.Image(
                            label="Chest X-ray (optional)",
                            type="pil",
                            height=250,
                        )
                        agent_btn = gr.Button("Run Diagnostic Pipeline", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        agent_output = gr.Textbox(
                            label="Clinical Assessment",
                            lines=20,
                            show_copy_button=True,
                        )
                        with gr.Accordion("Evidence-Based Guidelines", open=True):
                            guidelines_output = gr.Markdown(
                                label="Guideline Recommendations",
                            )
                        with gr.Accordion("Retrieved Sources", open=False):
                            sources_output = gr.Markdown(
                                label="Guideline Sources",
                            )

                agent_btn.click(
                    run_agentic_pipeline,
                    inputs=[agent_complaint, agent_history, agent_image, agent_age, agent_gender],
                    outputs=[agent_output, guidelines_output, sources_output],
                )

                # Example cases
                gr.Examples(
                    examples=[
                        ["Cough for 2 weeks with fever", "65 year old male smoker. Started with dry cough, now productive with yellow sputum. Low grade fever. Night sweats. 10 pound weight loss.", "65", "Male"],
                        ["Chest pain for 3 hours", "45 year old female. Substernal pressure radiating to left arm. Started at rest. Diaphoresis.", "45", "Female"],
                        ["Shortness of breath", "70 year old male with history of CHF. Increasing dyspnea on exertion. Orthopnea. Lower extremity edema.", "70", "Male"],
                    ],
                    inputs=[agent_complaint, agent_history, agent_age, agent_gender],
                )

            # Tab 1: Image Analysis
            with gr.TabItem("Chest X-ray Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Chest X-ray",
                            type="pil",
                            height=400,
                        )
                        analysis_type = gr.Dropdown(
                            choices=[
                                "Describe Findings",
                                "Bullet Point Findings",
                                "Differential Diagnosis",
                                "Structured Report",
                                "Primary Care Review",
                                "Custom Prompt",
                            ],
                            value="Primary Care Review",
                            label="Analysis Type",
                        )
                        custom_prompt = gr.Textbox(
                            label="Custom Prompt (optional)",
                            placeholder="Enter your custom prompt here...",
                            visible=False,
                        )
                        analyze_btn = gr.Button("Analyze X-ray", variant="primary")

                    with gr.Column(scale=1):
                        analysis_output = gr.Textbox(
                            label="Analysis Results",
                            lines=20,
                            show_copy_button=True,
                        )

                # Show/hide custom prompt based on selection
                def toggle_custom_prompt(choice):
                    return gr.update(visible=(choice == "Custom Prompt"))

                analysis_type.change(
                    toggle_custom_prompt,
                    inputs=[analysis_type],
                    outputs=[custom_prompt],
                )

                analyze_btn.click(
                    analyze_xray,
                    inputs=[image_input, analysis_type, custom_prompt],
                    outputs=[analysis_output],
                )

            # Tab 2: Classification
            with gr.TabItem("Zero-Shot Classification"):
                with gr.Row():
                    with gr.Column(scale=1):
                        classify_image = gr.Image(
                            label="Upload Chest X-ray",
                            type="pil",
                            height=400,
                        )
                        classify_btn = gr.Button("Classify Image", variant="primary")

                    with gr.Column(scale=1):
                        classify_output = gr.Textbox(
                            label="Classification Results",
                            lines=15,
                            show_copy_button=True,
                        )

                classify_btn.click(
                    classify_xray,
                    inputs=[classify_image],
                    outputs=[classify_output],
                )

            # Tab 3: Report Generation
            with gr.TabItem("Full Report"):
                with gr.Row():
                    with gr.Column(scale=1):
                        report_image = gr.Image(
                            label="Upload Chest X-ray",
                            type="pil",
                            height=300,
                        )
                        patient_age = gr.Textbox(
                            label="Patient Age",
                            placeholder="e.g., 65",
                        )
                        patient_gender = gr.Dropdown(
                            choices=["Male", "Female", "Other", "Unknown"],
                            value="Unknown",
                            label="Patient Gender",
                        )
                        clinical_history = gr.Textbox(
                            label="Clinical History",
                            placeholder="e.g., Cough for 2 weeks, fever, smoker",
                            lines=3,
                        )
                        report_btn = gr.Button("Generate Report", variant="primary")

                    with gr.Column(scale=1):
                        report_output = gr.Textbox(
                            label="Radiology Report",
                            lines=25,
                            show_copy_button=True,
                        )

                report_btn.click(
                    generate_report,
                    inputs=[report_image, patient_age, patient_gender, clinical_history],
                    outputs=[report_output],
                )

            # Tab 4: Medical Q&A
            with gr.TabItem("Medical Q&A"):
                gr.Markdown("""
                Ask medical questions without an image. MedGemma can answer questions about:
                - Radiology findings and interpretations
                - Differential diagnoses
                - Medical conditions and treatments
                - Clinical decision support
                """)

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the classic chest X-ray findings of heart failure?",
                    lines=3,
                )
                ask_btn = gr.Button("Ask MedGemma", variant="primary")
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=15,
                    show_copy_button=True,
                )

                ask_btn.click(
                    answer_question,
                    inputs=[question_input],
                    outputs=[answer_output],
                )

                # Example questions
                gr.Examples(
                    examples=[
                        ["What are the classic findings of pneumonia on a chest X-ray?"],
                        ["A 65-year-old smoker presents with hemoptysis and weight loss. What should be considered?"],
                        ["What is the difference between consolidation and ground-glass opacity?"],
                        ["How do you differentiate between CHF and pneumonia on chest X-ray?"],
                        ["What are the signs of tension pneumothorax?"],
                    ],
                    inputs=[question_input],
                )

            # Tab 5: Longitudinal CXR Comparison (NEW)
            with gr.TabItem("Longitudinal CXR"):
                gr.Markdown("""
                ### Longitudinal Chest X-ray Comparison

                Compare two chest X-rays over time to assess disease progression.
                Upload a prior study and a current study to see interval changes.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        prior_image = gr.Image(
                            label="Prior Study",
                            type="pil",
                            height=250,
                        )
                        current_image = gr.Image(
                            label="Current Study",
                            type="pil",
                            height=250,
                        )
                        with gr.Row():
                            long_context = gr.Textbox(
                                label="Clinical Context",
                                placeholder="e.g., 65yo smoker with cough",
                                lines=2,
                            )
                            long_interval = gr.Textbox(
                                label="Interval",
                                placeholder="e.g., 6 months",
                            )
                        compare_btn = gr.Button("Compare Studies", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        longitudinal_output = gr.Textbox(
                            label="Comparison Results",
                            lines=20,
                            show_copy_button=True,
                        )

                compare_btn.click(
                    run_longitudinal_comparison,
                    inputs=[prior_image, current_image, long_context, long_interval],
                    outputs=[longitudinal_output],
                )

            # Tab 6: CT/MRI Analysis (NEW)
            with gr.TabItem("CT/MRI Analysis"):
                gr.Markdown("""
                ### Volumetric CT/MRI Analysis

                Analyze CT or MRI studies by uploading individual slice images.
                The system will sample representative slices and synthesize findings.

                **Note:** Upload up to 20 slice images (PNG/JPG). The system will
                automatically sample 6 representative slices for analysis.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ct_upload = gr.File(
                            label="Upload CT/MRI Slices",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        with gr.Row():
                            modality_select = gr.Dropdown(
                                choices=["CT", "MRI"],
                                value="CT",
                                label="Modality",
                            )
                            region_select = gr.Dropdown(
                                choices=["Chest", "Abdomen", "Head"],
                                value="Chest",
                                label="Body Region",
                            )
                        vol_context = gr.Textbox(
                            label="Clinical Context",
                            placeholder="e.g., 55yo with abdominal pain",
                            lines=2,
                        )
                        vol_btn = gr.Button("Analyze Volume", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        vol_output = gr.Textbox(
                            label="Volumetric Analysis",
                            lines=20,
                            show_copy_button=True,
                        )

                vol_btn.click(
                    run_volumetric_analysis,
                    inputs=[ct_upload, modality_select, region_select, vol_context],
                    outputs=[vol_output],
                )

            # Tab 7: EHR Navigator (NEW)
            with gr.TabItem("EHR Navigator"):
                gr.Markdown("""
                ### EHR/FHIR Navigator

                Query patient EHR data using natural language questions.
                Paste FHIR-formatted JSON data to analyze patient records.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        ehr_question = gr.Textbox(
                            label="Ask about patient's EHR",
                            placeholder="e.g., What medications is the patient taking?",
                            lines=2,
                        )
                        ehr_json = gr.Textbox(
                            label="FHIR Data (JSON)",
                            placeholder='Paste FHIR Bundle JSON here...\n{\n  "resourceType": "Bundle",\n  "entry": [...]\n}',
                            lines=10,
                        )
                        ehr_btn = gr.Button("Query EHR", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        ehr_output = gr.Textbox(
                            label="Query Results",
                            lines=12,
                            show_copy_button=True,
                        )
                        ehr_summary = gr.Textbox(
                            label="Patient Summary",
                            lines=8,
                            show_copy_button=True,
                        )

                ehr_btn.click(
                    run_ehr_query,
                    inputs=[ehr_question, ehr_json],
                    outputs=[ehr_output, ehr_summary],
                )

                # Example FHIR query
                gr.Examples(
                    examples=[
                        ["What are the patient's active conditions?"],
                        ["List all current medications."],
                        ["Does the patient have any allergies?"],
                        ["What was the most recent lab result?"],
                    ],
                    inputs=[ehr_question],
                )

            # Tab 8: Pathology Analysis (NEW)
            with gr.TabItem("Pathology"):
                gr.Markdown("""
                ### Histopathology Analysis

                Analyze pathology images for microscopic findings.
                Supports H&E stained tissue images from various organ systems.

                **Note:** For whole slide images (.svs, .ndpi), the system requires
                OpenSlide. For demo purposes, upload pre-extracted tiles or
                low-resolution overview images.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        path_image = gr.Image(
                            label="Upload Pathology Image",
                            type="pil",
                            height=300,
                        )
                        tissue_type = gr.Dropdown(
                            choices=["Breast", "Lung", "Colon", "Prostate", "Skin",
                                    "Liver", "Kidney", "Lymph Node", "General"],
                            value="General",
                            label="Tissue Type",
                        )
                        path_context = gr.Textbox(
                            label="Clinical Context",
                            placeholder="e.g., Breast biopsy, suspicious mass on imaging",
                            lines=2,
                        )
                        path_btn = gr.Button("Analyze Pathology", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        path_output = gr.Textbox(
                            label="Pathology Analysis",
                            lines=20,
                            show_copy_button=True,
                        )

                path_btn.click(
                    run_pathology_analysis,
                    inputs=[path_image, tissue_type, path_context],
                    outputs=[path_output],
                )

        gr.Markdown("""
        ---
        **About This Demo**

        Built with [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it)
        and [MedSigLIP](https://huggingface.co/google/medsiglip-448) from Google Health AI.

        *MedGemma Impact Challenge 2026*
        """)

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Check for GPU
    if torch is not None and torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: torch/GPU not detected. Running on CPU will be slow.")

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
