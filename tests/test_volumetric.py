"""
Tests for Volumetric CT/MRI Analysis Feature

Tests slice sampling, sequential processing, and memory safety
for 3D volumetric imaging analysis.
"""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
import numpy as np

from src.agents.volumetric import (
    VolumetricImagingAgent,
    VolumetricAnalysis,
    VolumetricModality,
    SliceOrientation,
    SamplingStrategy,
    SliceSample,
    SliceFinding,
)


@pytest.fixture
def mock_model():
    """Create a mock MedGemma model."""
    model = Mock()
    model.analyze_image = Mock(return_value="Normal findings at this level.")
    model.ask = Mock(return_value="""
**KEY FINDINGS:**
- Normal lung parenchyma
- No masses or nodules

**IMPRESSION:**
Normal chest CT.

**DIFFERENTIAL DIAGNOSIS:**
1. Normal study

**RECOMMENDATIONS:**
None needed.

**URGENT:**
NO
""")
    return model


@pytest.fixture
def sample_volume():
    """Create a sample 3D volume as list of images."""
    return [
        Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        for _ in range(20)
    ]


@pytest.fixture
def sample_numpy_volume():
    """Create a sample 3D volume as numpy array."""
    return np.zeros((20, 256, 256), dtype=np.uint8)


class TestVolumetricAnalysisDataclass:
    """Tests for VolumetricAnalysis dataclass."""

    def test_creation(self):
        """Test creating VolumetricAnalysis."""
        analysis = VolumetricAnalysis(
            modality=VolumetricModality.CT,
            orientation=SliceOrientation.AXIAL,
            total_slices=100,
            slices_analyzed=6,
            impression="Normal study",
        )
        assert analysis.modality == VolumetricModality.CT
        assert analysis.slices_analyzed == 6

    def test_to_prompt_context(self):
        """Test formatting for downstream prompts."""
        analysis = VolumetricAnalysis(
            modality=VolumetricModality.CT,
            orientation=SliceOrientation.AXIAL,
            total_slices=100,
            slices_analyzed=6,
            key_findings=["No masses", "Normal heart size"],
            impression="Normal CT chest",
        )
        context = analysis.to_prompt_context()
        assert "CT" in context
        assert "6 of 100" in context
        assert "No masses" in context


class TestSliceSampling:
    """Tests for slice sampling functionality."""

    def test_uniform_sampling(self, mock_model, sample_volume):
        """Test uniform slice sampling."""
        agent = VolumetricImagingAgent(model=mock_model)
        samples = agent.sample_slices(
            sample_volume,
            num_slices=6,
            strategy=SamplingStrategy.UNIFORM,
        )

        assert len(samples) == 6
        assert all(isinstance(s, SliceSample) for s in samples)

    def test_max_slices_enforced(self, mock_model, sample_volume):
        """Test that MAX_SLICES is enforced."""
        agent = VolumetricImagingAgent(model=mock_model)

        # Request more than MAX_SLICES
        samples = agent.sample_slices(
            sample_volume,
            num_slices=20,  # More than MAX_SLICES (6)
        )

        assert len(samples) <= agent.MAX_SLICES

    def test_numpy_volume_conversion(self, mock_model, sample_numpy_volume):
        """Test conversion of numpy volume to images."""
        agent = VolumetricImagingAgent(model=mock_model)
        samples = agent.sample_slices(sample_numpy_volume, num_slices=4)

        assert len(samples) == 4
        assert all(isinstance(s.image, Image.Image) for s in samples)

    def test_slice_location_labels(self, mock_model, sample_volume):
        """Test that slices get appropriate location labels."""
        agent = VolumetricImagingAgent(model=mock_model)
        samples = agent.sample_slices(
            sample_volume,
            num_slices=6,
            body_region="chest",
        )

        locations = [s.location for s in samples]
        # Check that location labels are assigned
        assert all(loc is not None for loc in locations)

    def test_empty_volume(self, mock_model):
        """Test handling of empty volume."""
        agent = VolumetricImagingAgent(model=mock_model)
        samples = agent.sample_slices([], num_slices=6)
        assert samples == []


class TestVolumetricAnalysis:
    """Tests for volumetric analysis workflow."""

    def test_analyze_basic(self, mock_model, sample_volume):
        """Test basic volumetric analysis."""
        agent = VolumetricImagingAgent(model=mock_model)

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze(
                volume=sample_volume,
                modality=VolumetricModality.CT,
                clinical_context="Test context",
                body_region="chest",
                num_slices=4,
            )

        assert isinstance(result, VolumetricAnalysis)
        assert result.modality == VolumetricModality.CT
        assert result.slices_analyzed == 4

    def test_sequential_slice_processing(self, mock_model, sample_volume):
        """Verify slices are processed one at a time."""
        agent = VolumetricImagingAgent(model=mock_model)
        call_count = [0]

        def track_call(*args, **kwargs):
            call_count[0] += 1
            return f"Finding {call_count[0]}"

        mock_model.analyze_image = Mock(side_effect=track_call)

        with patch('torch.cuda.empty_cache'):
            agent.analyze(sample_volume, num_slices=4)

        # Should analyze exactly 4 slices
        assert call_count[0] == 4

    def test_mri_modality(self, mock_model, sample_volume):
        """Test MRI modality analysis."""
        agent = VolumetricImagingAgent(model=mock_model)

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze(
                volume=sample_volume,
                modality=VolumetricModality.MRI,
                num_slices=4,
            )

        assert result.modality == VolumetricModality.MRI

    def test_different_body_regions(self, mock_model, sample_volume):
        """Test analysis with different body regions."""
        agent = VolumetricImagingAgent(model=mock_model)

        for region in ["chest", "abdomen", "head"]:
            with patch('torch.cuda.empty_cache'):
                result = agent.analyze(
                    volume=sample_volume,
                    body_region=region,
                    num_slices=4,
                )
            assert isinstance(result, VolumetricAnalysis)


class TestMemorySafety:
    """Tests for memory safety on T4 GPU."""

    def test_cuda_cache_cleared_per_slice(self, mock_model, sample_volume):
        """Verify torch.cuda.empty_cache is called after each slice."""
        agent = VolumetricImagingAgent(model=mock_model)

        with patch('torch.cuda.empty_cache') as mock_cache:
            agent.analyze(sample_volume, num_slices=4)

        # Should be called at least once per slice
        assert mock_cache.call_count >= 4

    def test_max_slices_constant(self):
        """Verify MAX_SLICES is set conservatively for T4."""
        assert VolumetricImagingAgent.MAX_SLICES <= 6


class TestSliceSampleDataclass:
    """Tests for SliceSample dataclass."""

    def test_position_percent(self):
        """Test position percentage calculation."""
        sample = SliceSample(
            image=Image.new('RGB', (256, 256)),
            x=0,
            y=0,
            width=256,
            height=256,
            level=0,
            slice_index=50,
            total_slices=100,
            location="mid lung",
            magnification=1.0,
        )
        assert sample.position_percent == 50.0

    def test_location_string(self):
        """Test location string formatting."""
        sample = SliceSample(
            image=Image.new('RGB', (256, 256)),
            x=100,
            y=200,
            width=256,
            height=256,
            level=0,
            slice_index=10,
            total_slices=100,
            location="upper lung",
            magnification=10.0,
        )
        assert "10.0x" in sample.location_str
