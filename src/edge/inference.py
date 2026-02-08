"""
Edge Classifier - CPU-Only Inference with ONNX

Provides fast chest X-ray classification using a quantized ONNX
model running on CPU. Designed for resource-limited clinical settings.

API mirrors MedSigLIP.classify() for drop-in replacement.

Usage:
    classifier = EdgeClassifier("models/edge/medsiglip_int8.onnx")
    result = classifier.classify_pneumonia(image)
    # {"normal": 0.82, "pneumonia": 0.18}
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class EdgeClassifier:
    """
    CPU-only chest X-ray classifier using ONNX-quantized MedSigLIP.

    Runs the vision encoder via ONNX Runtime on CPU, then computes
    cosine similarity against pre-computed text embeddings for
    binary pneumonia classification.
    """

    # Labels corresponding to pre-computed text embeddings
    LABELS = ["normal", "pneumonia"]

    def __init__(
        self,
        model_path: str,
        text_embeddings_path: Optional[str] = None,
    ):
        """
        Initialize the edge classifier.

        Args:
            model_path: Path to the ONNX model file
            text_embeddings_path: Path to pre-computed text embeddings (.npy).
                                  Defaults to text_embeddings.npy in the same directory.
        """
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load ONNX model with CPU provider only
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        # Load pre-computed text embeddings
        if text_embeddings_path is None:
            text_embeddings_path = str(self.model_path.parent / "text_embeddings.npy")

        embeddings_path = Path(text_embeddings_path)
        if embeddings_path.exists():
            self.text_embeddings = np.load(str(embeddings_path))
        else:
            raise FileNotFoundError(
                f"Text embeddings not found: {embeddings_path}. "
                "Run scripts/export_edge_model.py to generate them."
            )

        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def _preprocess_image(self, image) -> np.ndarray:
        """
        Preprocess image for the vision encoder.

        Applies the same transforms as the MedSigLIP processor:
        resize to 448x448, normalize, convert to float32.

        Args:
            image: PIL Image

        Returns:
            numpy array of shape (1, 3, H, W)
        """
        from PIL import Image

        # Resize to model input size
        image = image.convert("RGB").resize((448, 448))

        # Convert to numpy and normalize to [0, 1]
        pixel_values = np.array(image, dtype=np.float32) / 255.0

        # Apply ImageNet normalization (same as SigLIP)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        pixel_values = (pixel_values - mean) / std

        # Transpose to (C, H, W) and add batch dimension
        pixel_values = pixel_values.transpose(2, 0, 1)
        pixel_values = np.expand_dims(pixel_values, axis=0)

        return pixel_values

    def classify_pneumonia(self, image) -> Dict[str, float]:
        """
        Classify a chest X-ray as normal or pneumonia.

        Args:
            image: PIL Image

        Returns:
            Dictionary with "normal" and "pneumonia" probabilities
        """
        # Preprocess
        pixel_values = self._preprocess_image(image)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: pixel_values})
        image_features = outputs[0]

        # Pool if needed (take [CLS] token or mean pool)
        if image_features.ndim == 3:
            # Shape: (batch, seq_len, embed_dim) -> mean pool
            image_features = image_features.mean(axis=1)

        # Normalize image features
        image_features = image_features / np.linalg.norm(
            image_features, axis=-1, keepdims=True
        )

        # Compute cosine similarity with pre-computed text embeddings
        similarities = np.dot(image_features, self.text_embeddings.T).squeeze()

        # Apply softmax to get probabilities
        exp_sim = np.exp(similarities - np.max(similarities))
        probs = exp_sim / exp_sim.sum()

        return {
            "normal": float(probs[0]),
            "pneumonia": float(probs[1]),
        }

    @property
    def model_size_mb(self) -> float:
        """Get model file size in MB."""
        return self.model_path.stat().st_size / (1024 * 1024)
