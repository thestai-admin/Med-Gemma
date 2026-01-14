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
import torch
from PIL import Image
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy loading for GPU environments
_model = None
_classifier = None


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

        **MedGemma Impact Challenge Demo**

        This demo showcases MedGemma's capabilities for primary care diagnostic support.
        Upload a chest X-ray to get AI-powered analysis, or ask medical questions.

        > **Disclaimer:** This is a research demonstration. All outputs require clinical
        > verification by qualified healthcare professionals before any clinical use.
        """)

        with gr.Tabs():
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
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU detected. Running on CPU will be slow.")

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
