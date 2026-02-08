"""Tests for ImagingAgent classification modes and decision metadata."""

from unittest.mock import Mock

from PIL import Image

from src.agents.imaging import ImagingAgent


ANALYSIS_TEXT = """**TECHNIQUE:**\nPA\n\n**QUALITY:**\nAdequate\n\n**FINDINGS:**\nNo focal infiltrate\n\n**IMPRESSION:**\nNo acute cardiopulmonary process\n\n**URGENT:**\nNO"""


def _build_classifier():
    classifier = Mock()

    def classify(image, labels):
        if labels == ImagingAgent.CXR_LABELS:
            return {
                label: score
                for label, score in zip(
                    ImagingAgent.CXR_LABELS,
                    [0.12, 0.22, 0.10, 0.06, 0.05, 0.08, 0.03, 0.09, 0.11, 0.14],
                )
            }

        if labels == ImagingAgent.PNEUMONIA_BINARY_LABELS:
            return {
                ImagingAgent.PNEUMONIA_BINARY_LABELS[0]: 0.40,
                ImagingAgent.PNEUMONIA_BINARY_LABELS[1]: 0.60,
            }

        raise AssertionError("Unexpected label set passed to classifier")

    classifier.classify = Mock(side_effect=classify)
    return classifier


def test_analyze_multilabel_mode_forces_classification_when_requested():
    model = Mock()
    model.analyze_image = Mock(return_value=ANALYSIS_TEXT)

    agent = ImagingAgent(model=model, classifier=_build_classifier(), load_classifier=False)
    image = Image.new("RGB", (64, 64))

    analysis = agent.analyze(
        image=image,
        include_classification=True,
        classification_mode="multilabel",
        skip_classification_if_confident=False,
    )

    assert analysis.classification_mode == "multilabel"
    assert analysis.classification_details.get("status") == "completed"
    assert "pneumonia" in analysis.classification_probs


def test_analyze_binary_mode_applies_threshold_decision():
    model = Mock()
    model.analyze_image = Mock(return_value=ANALYSIS_TEXT)

    agent = ImagingAgent(model=model, classifier=_build_classifier(), load_classifier=False)
    image = Image.new("RGB", (64, 64))

    analysis = agent.analyze(
        image=image,
        include_classification=True,
        classification_mode="binary",
        classification_threshold=0.65,
    )

    assert analysis.classification_mode == "binary"
    assert analysis.classification_probs["pneumonia"] == 0.60
    assert analysis.classification_details["is_pneumonia"] is False


def test_analyze_ensemble_mode_returns_combined_probability_and_decision():
    model = Mock()
    model.analyze_image = Mock(side_effect=[ANALYSIS_TEXT, "ABNORMAL"])

    agent = ImagingAgent(model=model, classifier=_build_classifier(), load_classifier=False)
    image = Image.new("RGB", (64, 64))

    analysis = agent.analyze(
        image=image,
        include_classification=True,
        classification_mode="ensemble",
        classification_threshold=0.60,
    )

    assert analysis.classification_mode == "ensemble"
    assert "pneumonia" in analysis.classification_probs
    assert analysis.classification_details["is_pneumonia"] is True
    assert analysis.classification_details["threshold"] == 0.60
