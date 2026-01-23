"""
Tests for Longitudinal CXR Comparison Feature

Tests the sequential processing and memory safety of comparing
two chest X-rays over time.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.agents.imaging import (
    ImagingAgent,
    LongitudinalAnalysis,
    ChangeStatus,
    ImageModality,
)


@pytest.fixture
def mock_model():
    """Create a mock MedGemma model."""
    model = Mock()
    model.analyze_image = Mock(return_value="Test findings from image analysis.")
    model.ask = Mock(return_value="""
**COMPARISON:**
Interval improvement in bilateral lung opacities.

**KEY CHANGES:**
- Resolution of right lower lobe consolidation
- Decreased bilateral pleural effusions

**CHANGE SUMMARY:**
IMPROVED

**URGENT:**
NO
""")
    return model


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))


class TestLongitudinalAnalysisDataclass:
    """Tests for LongitudinalAnalysis dataclass."""

    def test_creation(self):
        """Test creating LongitudinalAnalysis."""
        analysis = LongitudinalAnalysis(
            prior_findings="Prior findings text",
            current_findings="Current findings text",
            comparison="Comparison text",
            change_summary=ChangeStatus.IMPROVED,
        )
        assert analysis.prior_findings == "Prior findings text"
        assert analysis.change_summary == ChangeStatus.IMPROVED

    def test_to_prompt_context(self):
        """Test formatting for downstream prompts."""
        analysis = LongitudinalAnalysis(
            prior_findings="Prior findings",
            current_findings="Current findings",
            comparison="Interval improvement",
            change_summary=ChangeStatus.IMPROVED,
            interval="6 months",
            key_changes=["Resolved consolidation"],
        )
        context = analysis.to_prompt_context()
        assert "LONGITUDINAL IMAGING COMPARISON" in context
        assert "6 months" in context
        assert "IMPROVED" in context
        assert "Resolved consolidation" in context

    def test_urgent_flag(self):
        """Test urgent review flag in output."""
        analysis = LongitudinalAnalysis(
            prior_findings="Prior",
            current_findings="Current",
            comparison="Comparison",
            change_summary=ChangeStatus.WORSENED,
            requires_urgent_review=True,
        )
        context = analysis.to_prompt_context()
        assert "URGENT" in context


class TestChangeStatus:
    """Tests for ChangeStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses exist."""
        assert ChangeStatus.NEW.value == "new"
        assert ChangeStatus.IMPROVED.value == "improved"
        assert ChangeStatus.WORSENED.value == "worsened"
        assert ChangeStatus.UNCHANGED.value == "unchanged"
        assert ChangeStatus.RESOLVED.value == "resolved"


class TestImagingAgentLongitudinal:
    """Tests for ImagingAgent.analyze_longitudinal method."""

    def test_analyze_longitudinal_basic(self, mock_model, sample_image):
        """Test basic longitudinal analysis."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze_longitudinal(
                prior_image=sample_image,
                current_image=sample_image,
                clinical_context="Test context",
                interval="3 months",
            )

        assert isinstance(result, LongitudinalAnalysis)
        assert result.interval == "3 months"
        # Model should be called 3 times: 2 images + 1 synthesis
        assert mock_model.analyze_image.call_count == 2
        assert mock_model.ask.call_count == 1

    def test_sequential_processing(self, mock_model, sample_image):
        """Verify images are processed sequentially, not simultaneously."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)
        call_order = []

        def track_image_call(*args, **kwargs):
            call_order.append('image')
            return "Findings"

        def track_text_call(*args, **kwargs):
            call_order.append('text')
            return "**CHANGE SUMMARY:**\nIMPROVED\n**URGENT:**\nNO"

        mock_model.analyze_image = Mock(side_effect=track_image_call)
        mock_model.ask = Mock(side_effect=track_text_call)

        with patch('torch.cuda.empty_cache'):
            agent.analyze_longitudinal(
                prior_image=sample_image,
                current_image=sample_image,
            )

        # Should be: image, image, text (sequential)
        assert call_order == ['image', 'image', 'text']

    def test_change_status_parsing(self, mock_model, sample_image):
        """Test parsing of different change statuses."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)

        test_cases = [
            ("IMPROVED", ChangeStatus.IMPROVED),
            ("WORSENED", ChangeStatus.WORSENED),
            ("NEW", ChangeStatus.NEW),
            ("RESOLVED", ChangeStatus.RESOLVED),
            ("UNCHANGED", ChangeStatus.UNCHANGED),
        ]

        for status_text, expected_enum in test_cases:
            mock_model.ask.return_value = f"**CHANGE SUMMARY:**\n{status_text}\n**URGENT:**\nNO"

            with patch('torch.cuda.empty_cache'):
                result = agent.analyze_longitudinal(
                    prior_image=sample_image,
                    current_image=sample_image,
                )

            assert result.change_summary == expected_enum, f"Failed for {status_text}"

    def test_urgent_detection(self, mock_model, sample_image):
        """Test detection of urgent findings."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)

        mock_model.ask.return_value = "**CHANGE SUMMARY:**\nWORSENED\n**URGENT:**\nYES"

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze_longitudinal(
                prior_image=sample_image,
                current_image=sample_image,
            )

        assert result.requires_urgent_review is True

    def test_path_input_handling(self, mock_model, tmp_path):
        """Test that file paths are properly handled."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)

        # Create temporary test images
        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        prior_path = tmp_path / "prior.png"
        current_path = tmp_path / "current.png"
        img.save(prior_path)
        img.save(current_path)

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze_longitudinal(
                prior_image=str(prior_path),
                current_image=str(current_path),
            )

        assert isinstance(result, LongitudinalAnalysis)


class TestMemorySafety:
    """Tests for memory safety on T4 GPU."""

    def test_cuda_cache_cleared(self, mock_model, sample_image):
        """Verify torch.cuda.empty_cache is called between operations."""
        agent = ImagingAgent(model=mock_model, load_classifier=False)

        with patch('torch.cuda.empty_cache') as mock_cache:
            agent.analyze_longitudinal(
                prior_image=sample_image,
                current_image=sample_image,
            )

        # Should be called at least twice (after each image analysis)
        assert mock_cache.call_count >= 2
