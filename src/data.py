"""
Data Loading Utilities for Medical Imaging Datasets

Provides easy loading of medical imaging datasets from Hugging Face.
Optimized for streaming to handle large datasets without full download.

Datasets:
- NIH ChestX-ray14: 112K chest X-rays with 14 pathology labels
- Chest X-ray Pneumonia: Pediatric pneumonia classification
- CT-RATE: 50K CT volumes with reports
- MedQA: Medical question answering
"""

from datasets import load_dataset, Dataset
from PIL import Image
from typing import Iterator, Dict, Any, List, Optional, Tuple
from pathlib import Path
import random


# =============================================================================
# Dataset IDs
# =============================================================================

class DatasetIDs:
    """Hugging Face dataset IDs for medical data."""

    # Chest X-ray
    NIH_CHEST_XRAY = "alkzar90/NIH-Chest-X-ray-dataset"
    CHEST_XRAY_PNEUMONIA = "hf-vision/chest-xray-pneumonia"

    # CT
    CT_RATE = "ibrahimhamamci/CT-RATE"

    # Medical QA
    MEDQA = "bigbio/med_qa"
    MEDMCQA = "openlifescienceai/medmcqa"


# =============================================================================
# NIH Chest X-ray Dataset
# =============================================================================

class NIHChestXray:
    """
    NIH Chest X-ray14 Dataset loader.

    112,120 frontal-view chest X-rays from 30,805 patients.
    14 pathology labels: Atelectasis, Cardiomegaly, Effusion, Infiltration,
    Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema,
    Fibrosis, Pleural_Thickening, Hernia.

    Usage:
        dataset = NIHChestXray()
        for sample in dataset.stream(limit=10):
            image = sample['image']
            labels = sample['labels']
    """

    PATHOLOGIES = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
        "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    def __init__(self, streaming: bool = True):
        """
        Initialize dataset loader.

        Args:
            streaming: If True, stream data without full download (recommended)
        """
        self.streaming = streaming
        self._dataset = None

    def load(self, split: str = "train") -> Dataset:
        """Load the full dataset (may take time and space)."""
        self._dataset = load_dataset(
            DatasetIDs.NIH_CHEST_XRAY,
            split=split,
            streaming=False
        )
        return self._dataset

    def stream(self, split: str = "train", limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream samples from the dataset.

        Args:
            split: Dataset split ("train")
            limit: Maximum number of samples to yield

        Yields:
            Dictionary with 'image' (PIL Image) and 'labels' (list of strings)
        """
        dataset = load_dataset(
            DatasetIDs.NIH_CHEST_XRAY,
            split=split,
            streaming=True
        )

        count = 0
        for sample in dataset:
            if limit and count >= limit:
                break

            yield {
                "image": sample["image"],
                "labels": sample.get("labels", []),
                "patient_id": sample.get("Patient ID", ""),
                "age": sample.get("Patient Age", ""),
                "gender": sample.get("Patient Gender", ""),
            }
            count += 1

    def get_samples_by_pathology(
        self,
        pathology: str,
        limit: int = 10,
        split: str = "train"
    ) -> List[Dict[str, Any]]:
        """
        Get samples containing a specific pathology.

        Args:
            pathology: One of the 14 pathologies (e.g., "Pneumonia")
            limit: Maximum samples to return
            split: Dataset split

        Returns:
            List of samples with the specified pathology
        """
        if pathology not in self.PATHOLOGIES:
            raise ValueError(f"Unknown pathology: {pathology}. Must be one of {self.PATHOLOGIES}")

        samples = []
        for sample in self.stream(split=split):
            if pathology in sample.get("labels", []):
                samples.append(sample)
                if len(samples) >= limit:
                    break

        return samples

    def get_normal_samples(self, limit: int = 10, split: str = "train") -> List[Dict[str, Any]]:
        """Get samples with no pathology findings."""
        samples = []
        for sample in self.stream(split=split):
            if not sample.get("labels") or sample["labels"] == ["No Finding"]:
                samples.append(sample)
                if len(samples) >= limit:
                    break
        return samples


# =============================================================================
# Chest X-ray Pneumonia Dataset
# =============================================================================

class ChestXrayPneumonia:
    """
    Chest X-ray Pneumonia Dataset (pediatric).

    Binary classification: Normal vs Pneumonia.

    Usage:
        dataset = ChestXrayPneumonia()
        for sample in dataset.stream(split="train", limit=10):
            image = sample['image']
            label = sample['label']  # 0=Normal, 1=Pneumonia
    """

    def __init__(self, streaming: bool = True):
        self.streaming = streaming

    def stream(
        self,
        split: str = "train",
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Stream samples from the dataset."""
        dataset = load_dataset(
            DatasetIDs.CHEST_XRAY_PNEUMONIA,
            split=split,
            streaming=True
        )

        count = 0
        for sample in dataset:
            if limit and count >= limit:
                break

            yield {
                "image": sample["image"],
                "label": sample["label"],
                "label_name": "Pneumonia" if sample["label"] == 1 else "Normal"
            }
            count += 1


# =============================================================================
# Medical QA Dataset
# =============================================================================

class MedQA:
    """
    MedQA Dataset - USMLE-style medical questions.

    12,723 English questions from medical board exams.

    Usage:
        dataset = MedQA()
        for sample in dataset.stream(limit=10):
            question = sample['question']
            options = sample['options']
            answer = sample['answer']
    """

    def __init__(self, streaming: bool = True):
        self.streaming = streaming

    def stream(
        self,
        split: str = "train",
        limit: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """Stream questions from MedQA."""
        dataset = load_dataset(
            DatasetIDs.MEDQA,
            "med_qa_en_source",
            split=split,
            streaming=True
        )

        count = 0
        for sample in dataset:
            if limit and count >= limit:
                break

            yield {
                "question": sample.get("question", ""),
                "options": sample.get("options", {}),
                "answer": sample.get("answer", ""),
                "answer_idx": sample.get("answer_idx", ""),
            }
            count += 1


# =============================================================================
# Sample Data Generator
# =============================================================================

def get_sample_chest_xrays(n: int = 5) -> List[Dict[str, Any]]:
    """
    Get a diverse sample of chest X-rays for testing.

    Returns samples with different pathologies and normal cases.
    """
    dataset = NIHChestXray()
    samples = []

    # Get some normal samples
    normals = dataset.get_normal_samples(limit=max(1, n // 3))
    samples.extend(normals)

    # Get samples with common pathologies
    common_pathologies = ["Pneumonia", "Effusion", "Cardiomegaly", "Infiltration"]
    remaining = n - len(samples)

    for pathology in common_pathologies[:remaining]:
        pathology_samples = dataset.get_samples_by_pathology(pathology, limit=1)
        samples.extend(pathology_samples)
        if len(samples) >= n:
            break

    return samples[:n]


def get_sample_medical_questions(n: int = 5) -> List[Dict[str, Any]]:
    """Get sample medical questions for testing."""
    dataset = MedQA()
    return list(dataset.stream(limit=n))


# =============================================================================
# Convenience Functions
# =============================================================================

def load_nih_chest_xray(streaming: bool = True) -> NIHChestXray:
    """Get NIH Chest X-ray dataset loader."""
    return NIHChestXray(streaming=streaming)


def load_pneumonia_dataset(streaming: bool = True) -> ChestXrayPneumonia:
    """Get Pneumonia classification dataset loader."""
    return ChestXrayPneumonia(streaming=streaming)


def load_medqa(streaming: bool = True) -> MedQA:
    """Get MedQA dataset loader."""
    return MedQA(streaming=streaming)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Testing data loaders...")

    # Test NIH Chest X-ray
    print("\n1. NIH Chest X-ray Dataset:")
    nih = NIHChestXray()
    for i, sample in enumerate(nih.stream(limit=3)):
        print(f"   Sample {i+1}: Labels = {sample['labels']}")

    # Test Pneumonia dataset
    print("\n2. Pneumonia Dataset:")
    pneumonia = ChestXrayPneumonia()
    for i, sample in enumerate(pneumonia.stream(limit=3)):
        print(f"   Sample {i+1}: {sample['label_name']}")

    # Test MedQA
    print("\n3. MedQA Dataset:")
    medqa = MedQA()
    for i, sample in enumerate(medqa.stream(limit=2)):
        print(f"   Q{i+1}: {sample['question'][:100]}...")

    print("\nAll data loaders working!")
