"""
Pathology Agent - Histopathology and Whole Slide Image Analysis

Analyzes gigapixel whole slide images (WSI) by extracting and analyzing
representative tiles using MedGemma.

Memory Strategy (T4 16GB - CRITICAL):
- NEVER load full WSI into memory (can be 50,000 x 50,000 pixels)
- Use OpenSlide for on-demand tile access
- Process maximum 4 tiles at 512x512
- Use 10x magnification by default (smaller tiles)
- Clear CUDA cache between each tile analysis
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from PIL import Image
from enum import Enum
import numpy as np


class TissueType(Enum):
    """Supported tissue types for pathology analysis."""
    BREAST = "breast"
    LUNG = "lung"
    COLON = "colon"
    PROSTATE = "prostate"
    SKIN = "skin"
    LIVER = "liver"
    KIDNEY = "kidney"
    THYROID = "thyroid"
    LYMPH_NODE = "lymph_node"
    GENERAL = "general"


class StainType(Enum):
    """Common histological stain types."""
    HE = "H&E"  # Hematoxylin and Eosin
    IHC = "IHC"  # Immunohistochemistry
    SPECIAL = "special"
    UNKNOWN = "unknown"


class MagnificationLevel(Enum):
    """Standard magnification levels."""
    LOW = 5  # 5x - overview
    MEDIUM = 10  # 10x - standard screening
    HIGH = 20  # 20x - detail
    VERY_HIGH = 40  # 40x - cellular detail


@dataclass
class TileSample:
    """A single tile extracted from a WSI."""
    image: Image.Image
    x: int  # X coordinate in full resolution
    y: int  # Y coordinate in full resolution
    width: int
    height: int
    level: int  # Pyramid level (0 = highest resolution)
    magnification: float
    has_tissue: bool = True  # Whether tile contains tissue

    @property
    def location_str(self) -> str:
        """Human-readable location string."""
        return f"({self.x}, {self.y}) at {self.magnification}x"


@dataclass
class TileFinding:
    """Finding from a single tile analysis."""
    tile_location: str
    findings: str
    morphology: str = ""  # Cell/tissue morphology observations
    has_abnormality: bool = False
    abnormality_type: Optional[str] = None  # e.g., "mitotic figure", "atypia"


@dataclass
class PathologyAnalysis:
    """Complete analysis of a pathology image or WSI."""
    tissue_type: TissueType
    stain_type: StainType
    magnification: float
    tiles_analyzed: int
    tile_findings: List[TileFinding] = field(default_factory=list)
    overall_impression: str = ""
    morphological_features: List[str] = field(default_factory=list)
    differential_diagnosis: List[str] = field(default_factory=list)
    recommended_stains: List[str] = field(default_factory=list)
    requires_urgent_review: bool = False
    grade: Optional[str] = None  # Tumor grade if applicable
    raw_synthesis: str = ""

    def to_prompt_context(self) -> str:
        """Format analysis for downstream prompts."""
        lines = [
            f"**Tissue Type:** {self.tissue_type.value}",
            f"**Stain:** {self.stain_type.value}",
            f"**Magnification:** {self.magnification}x",
            f"**Tiles Analyzed:** {self.tiles_analyzed}",
            "",
        ]

        if self.morphological_features:
            lines.append("**Morphological Features:**")
            for feature in self.morphological_features:
                lines.append(f"- {feature}")
            lines.append("")

        if self.overall_impression:
            lines.append(f"**Impression:** {self.overall_impression}")

        if self.differential_diagnosis:
            lines.append("")
            lines.append("**Differential Diagnosis:**")
            for i, dx in enumerate(self.differential_diagnosis, 1):
                lines.append(f"{i}. {dx}")

        if self.grade:
            lines.append(f"\n**Grade:** {self.grade}")

        if self.requires_urgent_review:
            lines.append("\n**URGENT PATHOLOGY REVIEW REQUIRED**")

        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate a formatted pathology report section."""
        return f"""
========================================
PATHOLOGY ANALYSIS
========================================
Tissue: {self.tissue_type.value}
Stain: {self.stain_type.value}
Magnification: {self.magnification}x
Tiles Examined: {self.tiles_analyzed}

MICROSCOPIC DESCRIPTION:
{self.overall_impression}

{f'GRADE: {self.grade}' if self.grade else ''}

DIFFERENTIAL DIAGNOSIS:
{chr(10).join([f'{i+1}. {dx}' for i, dx in enumerate(self.differential_diagnosis)])}

{f'RECOMMENDED ADDITIONAL STAINS: {", ".join(self.recommended_stains)}' if self.recommended_stains else ''}
{"URGENT REVIEW RECOMMENDED" if self.requires_urgent_review else ""}
========================================
"""


class PathologyAgent:
    """
    Agent for analyzing histopathology images and whole slide images.

    Uses OpenSlide for memory-efficient WSI access and MedGemma
    for tile analysis, with strict memory constraints for T4 GPU.

    Usage:
        agent = PathologyAgent()

        # Analyze a WSI file
        analysis = agent.analyze_wsi(
            wsi_path="slide.svs",
            tissue_type=TissueType.BREAST,
            clinical_context="Breast biopsy, suspicious mass"
        )

        # Analyze a single pathology image
        analysis = agent.analyze_image(
            image=pil_image,
            tissue_type=TissueType.LUNG
        )
    """

    # CRITICAL: Conservative limits for T4 16GB
    MAX_TILES = 4
    TILE_SIZE = 512
    DEFAULT_MAGNIFICATION = MagnificationLevel.MEDIUM  # 10x

    # Supported WSI formats
    SUPPORTED_FORMATS = [".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".vms", ".vmu", ".scn"]

    # Tissue-specific prompts
    TILE_ANALYSIS_PROMPT = """Analyze this histopathology tile image.

**Tissue Type:** {tissue_type}
**Stain:** {stain_type}
**Location in Slide:** {location}
**Magnification:** {magnification}x
**Clinical Context:** {context}

Provide a detailed microscopic description:

**MORPHOLOGY:**
Describe the tissue architecture and cellular features:
- Cell types present
- Nuclear features (size, shape, chromatin pattern)
- Cytoplasmic features
- Tissue organization

**FINDINGS:**
Identify any pathological features:
- Abnormal cells or growth patterns
- Inflammatory infiltrate
- Necrosis or hemorrhage
- Stromal changes

**ABNORMALITY:**
[YES/NO] - Are there features concerning for malignancy or significant pathology?
If YES, describe the specific concerning features.
"""

    SYNTHESIS_PROMPT = """Synthesize findings from multiple tiles of a histopathology slide.

**Tissue Type:** {tissue_type}
**Stain:** {stain_type}
**Clinical Context:** {context}
**Number of Tiles Analyzed:** {num_tiles}

**Individual Tile Findings:**
{tile_findings}

Provide a comprehensive pathology assessment:

**MICROSCOPIC DESCRIPTION:**
Synthesize all findings into a coherent microscopic description.

**MORPHOLOGICAL FEATURES:**
List key morphological features observed (bullet points)

**IMPRESSION:**
Overall interpretation of the findings

**DIFFERENTIAL DIAGNOSIS:**
List possible diagnoses in order of likelihood (numbered list)

**GRADE:**
If applicable, suggest tumor grade based on features

**RECOMMENDED STAINS:**
Suggest additional immunohistochemical or special stains if needed

**URGENT:**
[YES/NO] - Are there findings requiring urgent pathologist review?
"""

    SINGLE_IMAGE_PROMPT = """Analyze this histopathology image.

**Tissue Type:** {tissue_type}
**Clinical Context:** {context}

Provide a comprehensive pathology analysis:

**MICROSCOPIC DESCRIPTION:**
Describe the tissue architecture, cell types, and any pathological features.

**MORPHOLOGICAL FEATURES:**
Key features observed (bullet points)

**IMPRESSION:**
Overall interpretation

**DIFFERENTIAL DIAGNOSIS:**
Possible diagnoses (numbered list, most likely first)

**RECOMMENDED ADDITIONAL WORKUP:**
Suggest any additional stains or studies if needed

**URGENT:**
[YES/NO] - Requires urgent review?
"""

    def __init__(self, model=None):
        """
        Initialize the Pathology Agent.

        Args:
            model: Optional MedGemma model instance
        """
        self._model = model
        self._openslide_available = self._check_openslide()

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def _check_openslide(self) -> bool:
        """Check if OpenSlide is available."""
        try:
            import openslide
            return True
        except ImportError:
            return False

    def extract_tiles(
        self,
        wsi_path: Union[str, Path],
        num_tiles: int = 4,
        tile_size: int = 512,
        magnification: MagnificationLevel = MagnificationLevel.MEDIUM,
        tissue_threshold: float = 0.5,
    ) -> List[TileSample]:
        """
        Extract representative tiles from a whole slide image.

        Uses OpenSlide for memory-efficient access.

        Args:
            wsi_path: Path to WSI file (.svs, .ndpi, etc.)
            num_tiles: Number of tiles to extract (max 4 for T4)
            tile_size: Size of each tile in pixels
            magnification: Target magnification level
            tissue_threshold: Minimum tissue fraction for tile selection

        Returns:
            List of TileSample objects
        """
        if not self._openslide_available:
            raise ImportError(
                "OpenSlide is required for WSI analysis. "
                "Install with: pip install openslide-python\n"
                "Also requires system library: apt-get install openslide-tools"
            )

        import openslide

        # Enforce memory safety
        num_tiles = min(num_tiles, self.MAX_TILES)

        wsi = openslide.OpenSlide(str(wsi_path))

        try:
            # Get slide properties
            level_count = wsi.level_count
            dimensions = wsi.level_dimensions

            # Find appropriate level for target magnification
            base_mag = float(wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
            target_mag = magnification.value

            # Calculate level (higher level = lower resolution)
            level = 0
            for i in range(level_count):
                downsample = wsi.level_downsamples[i]
                level_mag = base_mag / downsample
                if level_mag <= target_mag:
                    level = i
                    break

            level_dims = dimensions[level]
            downsample = wsi.level_downsamples[level]
            actual_mag = base_mag / downsample

            # Get thumbnail for tissue detection
            thumb_size = (256, 256)
            thumbnail = wsi.get_thumbnail(thumb_size)
            thumbnail_np = np.array(thumbnail.convert("L"))

            # Simple tissue detection (non-white regions)
            tissue_mask = thumbnail_np < 220

            # Find tissue regions
            tissue_coords = np.argwhere(tissue_mask)

            if len(tissue_coords) == 0:
                # No tissue detected, sample uniformly
                return self._uniform_tile_sampling(wsi, level, tile_size, num_tiles, actual_mag)

            # Sample tiles from tissue regions
            samples = []
            scale_x = level_dims[0] / thumb_size[0]
            scale_y = level_dims[1] / thumb_size[1]

            # Grid-based sampling with tissue detection
            tile_coords = self._get_tile_grid(
                tissue_coords,
                tissue_mask.shape,
                num_tiles,
                scale_x,
                scale_y,
                tile_size,
                level_dims,
            )

            for i, (x, y) in enumerate(tile_coords):
                # Read tile at appropriate level
                tile_x = int(x * downsample)  # Convert to level 0 coordinates
                tile_y = int(y * downsample)

                try:
                    tile = wsi.read_region((tile_x, tile_y), level, (tile_size, tile_size))
                    tile = tile.convert("RGB")

                    samples.append(TileSample(
                        image=tile,
                        x=tile_x,
                        y=tile_y,
                        width=tile_size,
                        height=tile_size,
                        level=level,
                        magnification=actual_mag,
                    ))
                except Exception:
                    continue

            return samples

        finally:
            wsi.close()

    def _uniform_tile_sampling(
        self,
        wsi,
        level: int,
        tile_size: int,
        num_tiles: int,
        magnification: float,
    ) -> List[TileSample]:
        """Uniform grid sampling when tissue detection fails."""
        dims = wsi.level_dimensions[level]
        downsample = wsi.level_downsamples[level]

        # Calculate grid
        cols = int(np.sqrt(num_tiles))
        rows = (num_tiles + cols - 1) // cols

        step_x = dims[0] // (cols + 1)
        step_y = dims[1] // (rows + 1)

        samples = []
        for i in range(rows):
            for j in range(cols):
                if len(samples) >= num_tiles:
                    break

                x = (j + 1) * step_x
                y = (i + 1) * step_y

                tile_x = int(x * downsample)
                tile_y = int(y * downsample)

                try:
                    tile = wsi.read_region((tile_x, tile_y), level, (tile_size, tile_size))
                    tile = tile.convert("RGB")

                    samples.append(TileSample(
                        image=tile,
                        x=tile_x,
                        y=tile_y,
                        width=tile_size,
                        height=tile_size,
                        level=level,
                        magnification=magnification,
                    ))
                except Exception:
                    continue

        return samples

    def _get_tile_grid(
        self,
        tissue_coords: np.ndarray,
        thumb_shape: Tuple[int, int],
        num_tiles: int,
        scale_x: float,
        scale_y: float,
        tile_size: int,
        level_dims: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Get tile coordinates from tissue regions."""
        # Find bounding box of tissue
        y_min, x_min = tissue_coords.min(axis=0)
        y_max, x_max = tissue_coords.max(axis=0)

        # Scale to level coordinates
        x_min_scaled = int(x_min * scale_x)
        x_max_scaled = int(x_max * scale_x)
        y_min_scaled = int(y_min * scale_y)
        y_max_scaled = int(y_max * scale_y)

        # Ensure within bounds
        x_min_scaled = max(0, x_min_scaled)
        y_min_scaled = max(0, y_min_scaled)
        x_max_scaled = min(level_dims[0] - tile_size, x_max_scaled)
        y_max_scaled = min(level_dims[1] - tile_size, y_max_scaled)

        # Calculate grid
        x_range = x_max_scaled - x_min_scaled
        y_range = y_max_scaled - y_min_scaled

        if x_range <= 0 or y_range <= 0:
            # Fallback to center
            center_x = level_dims[0] // 2 - tile_size // 2
            center_y = level_dims[1] // 2 - tile_size // 2
            return [(center_x, center_y)]

        cols = int(np.sqrt(num_tiles))
        rows = (num_tiles + cols - 1) // cols

        coords = []
        for i in range(rows):
            for j in range(cols):
                if len(coords) >= num_tiles:
                    break

                x = x_min_scaled + (j * x_range) // max(1, cols - 1) if cols > 1 else x_min_scaled + x_range // 2
                y = y_min_scaled + (i * y_range) // max(1, rows - 1) if rows > 1 else y_min_scaled + y_range // 2

                coords.append((int(x), int(y)))

        return coords

    def analyze_wsi(
        self,
        wsi_path: Union[str, Path],
        tissue_type: TissueType = TissueType.GENERAL,
        stain_type: StainType = StainType.HE,
        clinical_context: Optional[str] = None,
        num_tiles: int = 4,
        magnification: MagnificationLevel = MagnificationLevel.MEDIUM,
    ) -> PathologyAnalysis:
        """
        Analyze a whole slide image.

        IMPORTANT: Uses memory-efficient tile extraction and sequential processing.

        Args:
            wsi_path: Path to WSI file
            tissue_type: Type of tissue
            stain_type: Staining method
            clinical_context: Clinical information
            num_tiles: Number of tiles to analyze (max 4)
            magnification: Target magnification

        Returns:
            PathologyAnalysis with findings
        """
        import torch

        # Extract tiles
        print(f"Extracting {num_tiles} tiles from WSI...")
        tiles = self.extract_tiles(
            wsi_path,
            num_tiles=num_tiles,
            magnification=magnification,
        )

        if not tiles:
            return PathologyAnalysis(
                tissue_type=tissue_type,
                stain_type=stain_type,
                magnification=magnification.value,
                tiles_analyzed=0,
                overall_impression="Failed to extract tiles from WSI.",
            )

        context = clinical_context or "No clinical context provided"

        # Analyze each tile sequentially
        tile_findings = []
        for i, tile in enumerate(tiles):
            print(f"Analyzing tile {i+1}/{len(tiles)} at {tile.location_str}...")

            prompt = self.TILE_ANALYSIS_PROMPT.format(
                tissue_type=tissue_type.value,
                stain_type=stain_type.value,
                location=tile.location_str,
                magnification=tile.magnification,
                context=context,
            )

            finding_text = self.model.analyze_image(
                tile.image,
                prompt,
                max_new_tokens=800,
            )

            # Check for abnormality
            has_abnormality = "YES" in finding_text.upper().split("ABNORMALITY")[1][:50] if "ABNORMALITY" in finding_text.upper() else False

            tile_findings.append(TileFinding(
                tile_location=tile.location_str,
                findings=finding_text,
                has_abnormality=has_abnormality,
            ))

            # Clear GPU cache between tiles
            torch.cuda.empty_cache()

        # Synthesize findings
        print("Synthesizing pathology findings...")
        torch.cuda.empty_cache()

        return self._synthesize_findings(
            tile_findings=tile_findings,
            tissue_type=tissue_type,
            stain_type=stain_type,
            magnification=tiles[0].magnification if tiles else magnification.value,
            clinical_context=context,
        )

    def analyze_image(
        self,
        image: Union[Image.Image, str, Path],
        tissue_type: TissueType = TissueType.GENERAL,
        clinical_context: Optional[str] = None,
    ) -> PathologyAnalysis:
        """
        Analyze a single pathology image (not WSI).

        Use this for:
        - Pre-extracted tiles
        - Low-resolution overview images
        - Non-WSI pathology images

        Args:
            image: PIL Image or path
            tissue_type: Type of tissue
            clinical_context: Clinical information

        Returns:
            PathologyAnalysis with findings
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        context = clinical_context or "No clinical context provided"

        prompt = self.SINGLE_IMAGE_PROMPT.format(
            tissue_type=tissue_type.value,
            context=context,
        )

        response = self.model.analyze_image(image, prompt, max_new_tokens=1500)

        # Parse response
        return self._parse_single_analysis(
            response=response,
            tissue_type=tissue_type,
        )

    def _synthesize_findings(
        self,
        tile_findings: List[TileFinding],
        tissue_type: TissueType,
        stain_type: StainType,
        magnification: float,
        clinical_context: str,
    ) -> PathologyAnalysis:
        """Synthesize tile findings into overall analysis."""
        # Format tile findings
        findings_text = "\n\n".join([
            f"**Tile at {tf.tile_location}:**\n{tf.findings}"
            for tf in tile_findings
        ])

        prompt = self.SYNTHESIS_PROMPT.format(
            tissue_type=tissue_type.value,
            stain_type=stain_type.value,
            context=clinical_context,
            num_tiles=len(tile_findings),
            tile_findings=findings_text,
        )

        response = self.model.ask(prompt, max_new_tokens=1500)

        return self._parse_synthesis(
            response=response,
            tissue_type=tissue_type,
            stain_type=stain_type,
            magnification=magnification,
            tile_findings=tile_findings,
        )

    def _parse_synthesis(
        self,
        response: str,
        tissue_type: TissueType,
        stain_type: StainType,
        magnification: float,
        tile_findings: List[TileFinding],
    ) -> PathologyAnalysis:
        """Parse synthesis response into structured result."""
        sections = self._extract_sections(response)

        # Extract morphological features
        morph_features = []
        morph_text = sections.get("MORPHOLOGICAL FEATURES", "")
        for line in morph_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•")):
                morph_features.append(line.lstrip("-•").strip())

        # Extract differential diagnosis
        differentials = []
        diff_text = sections.get("DIFFERENTIAL DIAGNOSIS", "")
        for line in diff_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                differentials.append(line.lstrip("0123456789.-) ").strip())

        # Extract recommended stains
        stains = []
        stains_text = sections.get("RECOMMENDED STAINS", "")
        for item in stains_text.replace(",", "\n").split("\n"):
            item = item.strip()
            if item:
                stains.append(item)

        # Check urgency
        urgent_text = sections.get("URGENT", "").upper()
        requires_urgent = "YES" in urgent_text

        return PathologyAnalysis(
            tissue_type=tissue_type,
            stain_type=stain_type,
            magnification=magnification,
            tiles_analyzed=len(tile_findings),
            tile_findings=tile_findings,
            overall_impression=sections.get("IMPRESSION", sections.get("MICROSCOPIC DESCRIPTION", "")),
            morphological_features=morph_features,
            differential_diagnosis=differentials,
            recommended_stains=stains,
            requires_urgent_review=requires_urgent,
            grade=sections.get("GRADE", "").strip() or None,
            raw_synthesis=response,
        )

    def _parse_single_analysis(
        self,
        response: str,
        tissue_type: TissueType,
    ) -> PathologyAnalysis:
        """Parse single image analysis into structured result."""
        sections = self._extract_sections(response)

        # Extract morphological features
        morph_features = []
        morph_text = sections.get("MORPHOLOGICAL FEATURES", "")
        for line in morph_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•")):
                morph_features.append(line.lstrip("-•").strip())

        # Extract differential diagnosis
        differentials = []
        diff_text = sections.get("DIFFERENTIAL DIAGNOSIS", "")
        for line in diff_text.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                differentials.append(line.lstrip("0123456789.-) ").strip())

        # Check urgency
        urgent_text = sections.get("URGENT", "").upper()
        requires_urgent = "YES" in urgent_text

        return PathologyAnalysis(
            tissue_type=tissue_type,
            stain_type=StainType.UNKNOWN,
            magnification=0,  # Unknown for single images
            tiles_analyzed=1,
            overall_impression=sections.get("IMPRESSION", sections.get("MICROSCOPIC DESCRIPTION", "")),
            morphological_features=morph_features,
            differential_diagnosis=differentials,
            requires_urgent_review=requires_urgent,
            raw_synthesis=response,
        )

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract labeled sections from response."""
        sections = {}
        current_section = None
        current_content = []

        for line in text.split("\n"):
            line_stripped = line.strip()

            # Check for section headers
            if line_stripped.startswith("**") and line_stripped.endswith("**"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line_stripped.strip("*: ").upper()
                current_content = []
            elif ":" in line_stripped and any(
                key in line_stripped.upper()
                for key in ["MORPHOLOG", "IMPRESSION", "DIFFERENTIAL", "GRADE", "RECOMMEND", "URGENT", "MICROSCOPIC"]
            ):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                parts = line_stripped.split(":", 1)
                current_section = parts[0].strip("* ").upper()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_section:
                current_content.append(line_stripped)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections
