"""
Volumetric Imaging Agent - CT/MRI 3D Analysis

Analyzes volumetric CT/MRI scans by sampling representative slices
and synthesizing findings using MedGemma.

Memory Strategy (T4 16GB):
- Never load full 3D volume into GPU
- Process maximum 6 slices, one at a time
- Clear CUDA cache between each slice analysis
- Final synthesis is text-only (no images)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from PIL import Image
from enum import Enum
import numpy as np


class VolumetricModality(Enum):
    """Supported volumetric imaging modalities."""
    CT = "ct"
    MRI = "mri"
    PET_CT = "pet_ct"


class SliceOrientation(Enum):
    """Anatomical orientation of slices."""
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"


class SamplingStrategy(Enum):
    """Strategy for selecting representative slices."""
    UNIFORM = "uniform"  # Evenly spaced slices
    ANATOMY_GUIDED = "anatomy_guided"  # Focus on specific regions
    PATHOLOGY_WEIGHTED = "pathology_weighted"  # More slices where abnormalities expected


@dataclass
class SliceSample:
    """A single slice sampled from a volumetric study."""
    image: Image.Image
    slice_index: int
    total_slices: int
    location: str  # Anatomical location (e.g., "upper lung", "mid abdomen")
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    level: int = 0
    magnification: float = 1.0
    orientation: SliceOrientation = SliceOrientation.AXIAL

    @property
    def position_percent(self) -> float:
        """Position as percentage through volume (0-100)."""
        return (self.slice_index / max(1, self.total_slices)) * 100

    @property
    def location_str(self) -> str:
        """Human-readable location string."""
        return f"({self.x}, {self.y}) at {self.magnification}x"


@dataclass
class SliceFinding:
    """Finding from a single slice analysis."""
    slice_index: int
    location: str
    findings: str
    has_abnormality: bool = False
    abnormality_description: Optional[str] = None


@dataclass
class VolumetricAnalysis:
    """Complete analysis of a volumetric CT/MRI study."""
    modality: VolumetricModality
    orientation: SliceOrientation
    total_slices: int
    slices_analyzed: int
    slice_findings: List[SliceFinding] = field(default_factory=list)
    synthesis: str = ""
    key_findings: List[str] = field(default_factory=list)
    impression: str = ""
    requires_urgent_review: bool = False
    recommendations: List[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Format analysis for downstream prompts."""
        lines = [
            f"**Modality:** {self.modality.value.upper()}",
            f"**Slices Analyzed:** {self.slices_analyzed} of {self.total_slices}",
            f"**Orientation:** {self.orientation.value}",
            "",
        ]

        if self.key_findings:
            lines.append("**Key Findings:**")
            for i, finding in enumerate(self.key_findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")

        if self.impression:
            lines.append(f"**Impression:** {self.impression}")

        if self.recommendations:
            lines.append("")
            lines.append("**Recommendations:**")
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        if self.requires_urgent_review:
            lines.append("")
            lines.append("**URGENT REVIEW REQUIRED**")

        return "\n".join(lines)

    def to_report_section(self) -> str:
        """Generate a formatted report section."""
        return f"""
========================================
VOLUMETRIC IMAGING ANALYSIS
========================================
Modality: {self.modality.value.upper()}
Slices: {self.slices_analyzed} analyzed of {self.total_slices} total
Orientation: {self.orientation.value}

SYNTHESIS:
{self.synthesis}

IMPRESSION:
{self.impression}
{"URGENT REVIEW REQUIRED" if self.requires_urgent_review else ""}
========================================
"""


class VolumetricImagingAgent:
    """
    Agent for analyzing volumetric CT/MRI scans.

    Uses slice sampling to efficiently analyze 3D volumes within
    T4 GPU memory constraints (16GB VRAM).

    Usage:
        agent = VolumetricImagingAgent()
        analysis = agent.analyze(
            volume=ct_slices,  # List of PIL images or numpy array
            modality=VolumetricModality.CT,
            clinical_context="65yo with abdominal pain"
        )
    """

    # Conservative slice limit for T4 16GB
    MAX_SLICES = 6

    # Default tile size for each slice
    TARGET_SIZE = (512, 512)

    # Anatomical location labels by position
    CHEST_LOCATIONS = ["apex", "upper lung", "mid lung", "lower lung", "base", "diaphragm"]
    ABDOMEN_LOCATIONS = ["upper abdomen", "liver level", "mid abdomen", "pelvis level", "lower pelvis", "pelvic floor"]
    HEAD_LOCATIONS = ["vertex", "high convexity", "centrum semiovale", "basal ganglia", "brainstem", "skull base"]

    # Analysis prompts
    CT_SLICE_PROMPT = """Analyze this CT slice from a volumetric study.

**Slice Location:** {location} (slice {slice_num} of {total_slices})
**Clinical Context:** {context}

Describe all visible findings:
1. Normal anatomical structures
2. Any abnormalities (masses, fluid, air, calcifications, etc.)
3. Quality of the image at this level

Keep the response concise but thorough.
"""

    MRI_SLICE_PROMPT = """Analyze this MRI slice from a volumetric study.

**Slice Location:** {location} (slice {slice_num} of {total_slices})
**Clinical Context:** {context}

Describe all visible findings:
1. Signal characteristics of visible structures
2. Any abnormal signal (masses, edema, enhancement, etc.)
3. Quality and artifacts at this level

Keep the response concise but thorough.
"""

    SYNTHESIS_PROMPT = """Synthesize the findings from a volumetric {modality} study.

**Number of slices analyzed:** {num_slices}

**Individual Slice Findings:**
{slice_findings}

**Clinical Context:** {context}

Provide a comprehensive synthesis:

**KEY FINDINGS:**
List the most important findings in order of clinical significance (bullet points)

**IMPRESSION:**
Concise overall impression of the study

**DIFFERENTIAL DIAGNOSIS:**
If abnormalities present, list differential diagnoses in order of likelihood

**RECOMMENDATIONS:**
Suggest follow-up imaging or clinical correlation if needed

**URGENT:**
[YES/NO] - Are there findings requiring immediate attention?
"""

    def __init__(self, model=None):
        """
        Initialize the Volumetric Imaging Agent.

        Args:
            model: Optional MedGemma model instance (lazy loaded if not provided)
        """
        self._model = model

    @property
    def model(self):
        """Lazy load MedGemma model."""
        if self._model is None:
            from ..model import MedGemma
            self._model = MedGemma()
        return self._model

    def sample_slices(
        self,
        volume: Union[List[Image.Image], np.ndarray, List[str], List[Path]],
        num_slices: int = 6,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        body_region: str = "chest",
    ) -> List[SliceSample]:
        """
        Sample representative slices from a volumetric study.

        Args:
            volume: List of PIL images, numpy array [D, H, W], or list of paths
            num_slices: Number of slices to sample (max 6 for T4)
            strategy: Sampling strategy
            body_region: Body region for anatomical labeling

        Returns:
            List of SliceSample objects
        """
        # Enforce max slices for memory safety
        num_slices = min(num_slices, self.MAX_SLICES)

        # Convert volume to list of PIL images
        images = self._volume_to_images(volume)
        total_slices = len(images)

        if total_slices == 0:
            return []

        # Calculate slice indices based on strategy
        if strategy == SamplingStrategy.UNIFORM:
            indices = self._uniform_sampling(total_slices, num_slices)
        else:
            # Default to uniform for now
            indices = self._uniform_sampling(total_slices, num_slices)

        # Get location labels
        if body_region.lower() == "chest":
            locations = self.CHEST_LOCATIONS
        elif body_region.lower() == "abdomen":
            locations = self.ABDOMEN_LOCATIONS
        elif body_region.lower() == "head":
            locations = self.HEAD_LOCATIONS
        else:
            locations = [f"slice {i+1}" for i in range(num_slices)]

        # Create slice samples
        samples = []
        for i, idx in enumerate(indices):
            # Map slice index to location label
            location_idx = min(i, len(locations) - 1)
            location = locations[location_idx]

            # Resize if needed
            img = images[idx]
            if img.size != self.TARGET_SIZE:
                img = img.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)

            samples.append(SliceSample(
                image=img,
                slice_index=idx,
                total_slices=total_slices,
                location=location,
            ))

        return samples

    def _uniform_sampling(self, total: int, num_samples: int) -> List[int]:
        """Sample evenly spaced slice indices."""
        if total <= num_samples:
            return list(range(total))
        indices = np.linspace(0, total - 1, num_samples, dtype=int)
        return indices.tolist()

    def _volume_to_images(
        self,
        volume: Union[List[Image.Image], np.ndarray, List[str], List[Path]],
    ) -> List[Image.Image]:
        """Convert various volume formats to list of PIL images."""
        if isinstance(volume, np.ndarray):
            # Assume [D, H, W] or [D, H, W, C] format
            images = []
            for i in range(volume.shape[0]):
                slice_data = volume[i]
                if slice_data.ndim == 2:
                    # Grayscale
                    img = Image.fromarray(slice_data.astype(np.uint8))
                else:
                    # RGB/RGBA
                    img = Image.fromarray(slice_data.astype(np.uint8))
                images.append(img)
            return images

        elif isinstance(volume, list):
            if len(volume) == 0:
                return []

            if isinstance(volume[0], Image.Image):
                return volume

            elif isinstance(volume[0], (str, Path)):
                return [Image.open(p) for p in volume]

        return []

    def analyze(
        self,
        volume: Union[List[Image.Image], np.ndarray, List[str], List[Path]],
        modality: VolumetricModality = VolumetricModality.CT,
        clinical_context: Optional[str] = None,
        body_region: str = "chest",
        num_slices: int = 6,
        orientation: SliceOrientation = SliceOrientation.AXIAL,
    ) -> VolumetricAnalysis:
        """
        Analyze a volumetric CT/MRI study.

        IMPORTANT: Processes slices sequentially to avoid GPU OOM.

        Args:
            volume: List of slice images, numpy array, or list of paths
            modality: CT or MRI
            clinical_context: Clinical information
            body_region: Body region for anatomical labeling
            num_slices: Number of slices to sample (max 6)
            orientation: Slice orientation

        Returns:
            VolumetricAnalysis with findings and synthesis
        """
        import torch

        # Sample representative slices
        samples = self.sample_slices(
            volume,
            num_slices=num_slices,
            body_region=body_region,
        )

        if not samples:
            return VolumetricAnalysis(
                modality=modality,
                orientation=orientation,
                total_slices=0,
                slices_analyzed=0,
                synthesis="No valid slices provided for analysis.",
            )

        total_slices = samples[0].total_slices if samples else 0
        context = clinical_context or "No clinical context provided"

        # Select prompt based on modality
        prompt_template = self.CT_SLICE_PROMPT if modality == VolumetricModality.CT else self.MRI_SLICE_PROMPT

        # Analyze each slice sequentially
        slice_findings = []
        for i, sample in enumerate(samples):
            print(f"Analyzing slice {i+1}/{len(samples)} ({sample.location})...")

            prompt = prompt_template.format(
                location=sample.location,
                slice_num=sample.slice_index + 1,
                total_slices=total_slices,
                context=context,
            )

            # Analyze slice
            finding_text = self.model.analyze_image(
                sample.image,
                prompt,
                max_new_tokens=800,
            )

            # Check for abnormalities
            has_abnormality = any(
                term in finding_text.lower()
                for term in ["abnormal", "mass", "nodule", "opacity", "effusion",
                            "lesion", "tumor", "consolidation", "edema", "hemorrhage"]
            )

            slice_findings.append(SliceFinding(
                slice_index=sample.slice_index,
                location=sample.location,
                findings=finding_text,
                has_abnormality=has_abnormality,
            ))

            # Clear GPU cache between slices
            torch.cuda.empty_cache()

        # Synthesize findings (text-only)
        print("Synthesizing findings...")
        torch.cuda.empty_cache()

        # Format slice findings for synthesis prompt
        findings_text = "\n\n".join([
            f"**{sf.location} (Slice {sf.slice_index + 1}):**\n{sf.findings}"
            for sf in slice_findings
        ])

        synthesis_prompt = self.SYNTHESIS_PROMPT.format(
            modality=modality.value.upper(),
            num_slices=len(slice_findings),
            slice_findings=findings_text,
            context=context,
        )

        synthesis_response = self.model.ask(synthesis_prompt, max_new_tokens=1500)

        # Parse synthesis response
        return self._parse_synthesis(
            synthesis_response=synthesis_response,
            modality=modality,
            orientation=orientation,
            total_slices=total_slices,
            slice_findings=slice_findings,
        )

    def _parse_synthesis(
        self,
        synthesis_response: str,
        modality: VolumetricModality,
        orientation: SliceOrientation,
        total_slices: int,
        slice_findings: List[SliceFinding],
    ) -> VolumetricAnalysis:
        """Parse synthesis response into structured result."""
        # Simple section extraction
        sections = {}
        current_section = None
        current_content = []

        for line in synthesis_response.split("\n"):
            line = line.strip()

            # Check for section headers
            if line.startswith("**") and line.endswith("**"):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = line.strip("*: ").upper()
                current_content = []
            elif ":" in line and any(
                key in line.upper() for key in ["KEY FINDINGS", "IMPRESSION", "RECOMMENDATIONS", "URGENT"]
            ):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                parts = line.split(":", 1)
                current_section = parts[0].strip("* ").upper()
                current_content = [parts[1].strip()] if len(parts) > 1 else []
            elif current_section:
                current_content.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        # Extract key findings
        key_findings = []
        key_findings_text = sections.get("KEY FINDINGS", "")
        for line in key_findings_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                content = line.lstrip("0123456789.-•) ").strip()
                if content:
                    key_findings.append(content)

        # Extract recommendations
        recommendations = []
        recs_text = sections.get("RECOMMENDATIONS", "")
        for line in recs_text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line[0].isdigit()):
                content = line.lstrip("0123456789.-•) ").strip()
                if content:
                    recommendations.append(content)

        # Check urgency
        urgent_text = sections.get("URGENT", "").upper()
        requires_urgent = "YES" in urgent_text

        return VolumetricAnalysis(
            modality=modality,
            orientation=orientation,
            total_slices=total_slices,
            slices_analyzed=len(slice_findings),
            slice_findings=slice_findings,
            synthesis=synthesis_response,
            key_findings=key_findings,
            impression=sections.get("IMPRESSION", ""),
            requires_urgent_review=requires_urgent,
            recommendations=recommendations,
        )

    def analyze_from_dicom_dir(
        self,
        dicom_dir: Union[str, Path],
        modality: VolumetricModality = VolumetricModality.CT,
        clinical_context: Optional[str] = None,
    ) -> VolumetricAnalysis:
        """
        Analyze CT/MRI from a directory of DICOM files.

        Note: Requires pydicom package (not in default requirements).

        Args:
            dicom_dir: Directory containing DICOM files
            modality: CT or MRI
            clinical_context: Clinical information

        Returns:
            VolumetricAnalysis with findings
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM analysis. "
                "Install with: pip install pydicom"
            )

        dicom_dir = Path(dicom_dir)
        dicom_files = sorted(dicom_dir.glob("*.dcm"))

        if not dicom_files:
            # Try without extension
            dicom_files = sorted([
                f for f in dicom_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ])

        # Load DICOM files and extract pixel data
        slices = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(dcm_file)
                pixel_array = ds.pixel_array

                # Normalize to 0-255
                pixel_array = pixel_array.astype(np.float32)
                pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
                pixel_array = (pixel_array * 255).astype(np.uint8)

                # Convert to PIL
                img = Image.fromarray(pixel_array)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                slices.append(img)
            except Exception:
                continue

        if not slices:
            return VolumetricAnalysis(
                modality=modality,
                orientation=SliceOrientation.AXIAL,
                total_slices=0,
                slices_analyzed=0,
                synthesis="Failed to load DICOM files.",
            )

        return self.analyze(
            volume=slices,
            modality=modality,
            clinical_context=clinical_context,
        )
