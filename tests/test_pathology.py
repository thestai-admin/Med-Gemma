"""
Tests for Histopathology/WSI Support Feature

Tests tile extraction, sequential processing, and memory safety
for whole slide image analysis.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np

from src.agents.pathology import (
    PathologyAgent,
    PathologyAnalysis,
    TissueType,
    StainType,
    MagnificationLevel,
    TileSample,
    TileFinding,
)


@pytest.fixture
def mock_model():
    """Create a mock MedGemma model."""
    model = Mock()
    model.analyze_image = Mock(return_value="""
**MORPHOLOGY:**
Normal ductal epithelium with mild stromal changes.

**FINDINGS:**
No significant abnormalities identified.

**ABNORMALITY:**
NO
""")
    model.ask = Mock(return_value="""
**MICROSCOPIC DESCRIPTION:**
Normal breast tissue with benign ductal epithelium.

**MORPHOLOGICAL FEATURES:**
- Normal ductal architecture
- Unremarkable stroma

**IMPRESSION:**
Benign breast tissue, no malignancy identified.

**DIFFERENTIAL DIAGNOSIS:**
1. Normal breast tissue
2. Fibrocystic changes

**GRADE:**
N/A

**RECOMMENDED STAINS:**
None needed.

**URGENT:**
NO
""")
    return model


@pytest.fixture
def sample_image():
    """Create a sample pathology image."""
    return Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))


class TestPathologyAnalysisDataclass:
    """Tests for PathologyAnalysis dataclass."""

    def test_creation(self):
        """Test creating PathologyAnalysis."""
        analysis = PathologyAnalysis(
            tissue_type=TissueType.BREAST,
            stain_type=StainType.HE,
            magnification=10.0,
            tiles_analyzed=4,
            overall_impression="Benign tissue",
        )
        assert analysis.tissue_type == TissueType.BREAST
        assert analysis.magnification == 10.0

    def test_to_prompt_context(self):
        """Test formatting for downstream prompts."""
        analysis = PathologyAnalysis(
            tissue_type=TissueType.LUNG,
            stain_type=StainType.HE,
            magnification=20.0,
            tiles_analyzed=4,
            morphological_features=["Alveolar architecture preserved"],
            overall_impression="Normal lung parenchyma",
            differential_diagnosis=["Normal lung tissue"],
        )
        context = analysis.to_prompt_context()
        assert "lung" in context.lower()
        assert "H&E" in context
        assert "Alveolar" in context

    def test_urgent_flag(self):
        """Test urgent review flag in output."""
        analysis = PathologyAnalysis(
            tissue_type=TissueType.BREAST,
            stain_type=StainType.HE,
            magnification=10.0,
            tiles_analyzed=4,
            requires_urgent_review=True,
        )
        context = analysis.to_prompt_context()
        assert "URGENT" in context

    def test_grade_display(self):
        """Test grade is displayed when present."""
        analysis = PathologyAnalysis(
            tissue_type=TissueType.BREAST,
            stain_type=StainType.HE,
            magnification=10.0,
            tiles_analyzed=4,
            grade="Grade 2",
        )
        context = analysis.to_prompt_context()
        assert "Grade 2" in context


class TestTissueType:
    """Tests for TissueType enum."""

    def test_all_tissue_types(self):
        """Verify all expected tissue types exist."""
        assert TissueType.BREAST.value == "breast"
        assert TissueType.LUNG.value == "lung"
        assert TissueType.COLON.value == "colon"
        assert TissueType.PROSTATE.value == "prostate"
        assert TissueType.SKIN.value == "skin"
        assert TissueType.LIVER.value == "liver"
        assert TissueType.KIDNEY.value == "kidney"
        assert TissueType.LYMPH_NODE.value == "lymph_node"
        assert TissueType.GENERAL.value == "general"


class TestStainType:
    """Tests for StainType enum."""

    def test_stain_types(self):
        """Verify stain types exist."""
        assert StainType.HE.value == "H&E"
        assert StainType.IHC.value == "IHC"
        assert StainType.SPECIAL.value == "special"


class TestTileSample:
    """Tests for TileSample dataclass."""

    def test_creation(self):
        """Test creating TileSample."""
        tile = TileSample(
            image=Image.new('RGB', (512, 512)),
            x=1000,
            y=2000,
            width=512,
            height=512,
            level=0,
            magnification=10.0,
        )
        assert tile.x == 1000
        assert tile.magnification == 10.0

    def test_location_str(self):
        """Test location string formatting."""
        tile = TileSample(
            image=Image.new('RGB', (512, 512)),
            x=1000,
            y=2000,
            width=512,
            height=512,
            level=0,
            magnification=10.0,
        )
        assert "(1000, 2000)" in tile.location_str
        assert "10.0x" in tile.location_str


class TestPathologyAgentImageAnalysis:
    """Tests for PathologyAgent single image analysis."""

    def test_analyze_image_basic(self, mock_model, sample_image):
        """Test basic single image analysis."""
        agent = PathologyAgent(model=mock_model)

        result = agent.analyze_image(
            image=sample_image,
            tissue_type=TissueType.BREAST,
            clinical_context="Breast biopsy, suspicious mass",
        )

        assert isinstance(result, PathologyAnalysis)
        assert result.tissue_type == TissueType.BREAST
        assert result.tiles_analyzed == 1

    def test_analyze_image_different_tissues(self, mock_model, sample_image):
        """Test analysis with different tissue types."""
        agent = PathologyAgent(model=mock_model)

        for tissue in [TissueType.LUNG, TissueType.COLON, TissueType.SKIN]:
            result = agent.analyze_image(
                image=sample_image,
                tissue_type=tissue,
            )
            assert result.tissue_type == tissue

    def test_path_input(self, mock_model, tmp_path):
        """Test that file paths are properly handled."""
        agent = PathologyAgent(model=mock_model)

        # Create temporary test image
        img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        img_path = tmp_path / "test_path.png"
        img.save(img_path)

        result = agent.analyze_image(
            image=str(img_path),
            tissue_type=TissueType.GENERAL,
        )

        assert isinstance(result, PathologyAnalysis)


class TestPathologyAgentWSI:
    """Tests for WSI analysis (requires OpenSlide mock)."""

    def test_openslide_check(self, mock_model):
        """Test OpenSlide availability check."""
        agent = PathologyAgent(model=mock_model)
        # Should not raise, just return False if not available
        assert isinstance(agent._openslide_available, bool)

    def test_tile_extraction_without_openslide(self, mock_model):
        """Test that WSI analysis fails gracefully without OpenSlide."""
        agent = PathologyAgent(model=mock_model)

        if not agent._openslide_available:
            with pytest.raises(ImportError):
                agent.extract_tiles("nonexistent.svs")

    def test_max_tiles_constant(self):
        """Verify MAX_TILES is set conservatively for T4."""
        assert PathologyAgent.MAX_TILES <= 4

    def test_tile_size_constant(self):
        """Verify TILE_SIZE is reasonable."""
        assert PathologyAgent.TILE_SIZE == 512


class TestTileFinding:
    """Tests for TileFinding dataclass."""

    def test_creation(self):
        """Test creating TileFinding."""
        finding = TileFinding(
            tile_location="(1000, 2000) at 10x",
            findings="Normal ductal epithelium",
            morphology="Regular cell arrangement",
            has_abnormality=False,
        )
        assert finding.tile_location == "(1000, 2000) at 10x"
        assert finding.has_abnormality is False

    def test_abnormality_detection(self):
        """Test abnormality flag."""
        finding = TileFinding(
            tile_location="(500, 500) at 20x",
            findings="Atypical cells with high N:C ratio",
            has_abnormality=True,
            abnormality_type="atypia",
        )
        assert finding.has_abnormality is True
        assert finding.abnormality_type == "atypia"


class TestSynthesis:
    """Tests for findings synthesis."""

    def test_synthesize_findings(self, mock_model):
        """Test synthesis of tile findings."""
        agent = PathologyAgent(model=mock_model)

        tile_findings = [
            TileFinding(
                tile_location="(0, 0) at 10x",
                findings="Normal tissue",
                has_abnormality=False,
            ),
            TileFinding(
                tile_location="(1000, 0) at 10x",
                findings="Normal tissue",
                has_abnormality=False,
            ),
        ]

        result = agent._synthesize_findings(
            tile_findings=tile_findings,
            tissue_type=TissueType.BREAST,
            stain_type=StainType.HE,
            magnification=10.0,
            clinical_context="Breast biopsy",
        )

        assert isinstance(result, PathologyAnalysis)
        assert result.tiles_analyzed == 2


class TestMemorySafety:
    """Tests for memory safety on T4 GPU."""

    def test_conservative_limits(self):
        """Verify conservative memory limits."""
        assert PathologyAgent.MAX_TILES == 4
        assert PathologyAgent.TILE_SIZE == 512
        # 4 tiles x 512x512x3 = ~3MB, well under T4 limits

    def test_supported_formats(self):
        """Verify supported WSI formats."""
        formats = PathologyAgent.SUPPORTED_FORMATS
        assert ".svs" in formats
        assert ".ndpi" in formats
        assert ".tif" in formats


class TestParsing:
    """Tests for response parsing."""

    def test_extract_sections(self, mock_model):
        """Test section extraction from response."""
        agent = PathologyAgent(model=mock_model)

        response = """
**IMPRESSION:**
Benign breast tissue

**DIFFERENTIAL DIAGNOSIS:**
1. Fibrocystic changes
2. Normal breast

**URGENT:**
NO
"""
        sections = agent._extract_sections(response)

        assert "IMPRESSION" in sections
        assert "DIFFERENTIAL DIAGNOSIS" in sections
        assert "URGENT" in sections

    def test_parse_single_analysis(self, mock_model):
        """Test parsing of single image analysis."""
        agent = PathologyAgent(model=mock_model)

        response = """
**MICROSCOPIC DESCRIPTION:**
Normal breast ductal epithelium.

**MORPHOLOGICAL FEATURES:**
- Regular ductal architecture
- No atypia

**IMPRESSION:**
Benign breast tissue.

**DIFFERENTIAL DIAGNOSIS:**
1. Normal breast
2. Fibrocystic changes

**URGENT:**
NO
"""

        result = agent._parse_single_analysis(response, TissueType.BREAST)

        assert result.tissue_type == TissueType.BREAST
        assert len(result.morphological_features) == 2
        assert len(result.differential_diagnosis) >= 1
        assert result.requires_urgent_review is False
