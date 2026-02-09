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

        # Load preprocessor config (saved during export), fall back to defaults
        self._image_mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._image_std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._image_size = 448
        self._logit_scale = 10.0  # fallback; overwritten by config
        self._logit_bias = 0.0    # fallback; overwritten by config
        try:
            import json
            config_path = self.model_path.parent / "preprocess_config.json"
            with open(str(config_path)) as f:
                cfg = json.load(f)
            self._image_mean = np.array(cfg["image_mean"], dtype=np.float32)
            self._image_std = np.array(cfg["image_std"], dtype=np.float32)
            self._image_size = cfg["size"]
            self._logit_scale = float(cfg.get("logit_scale", 10.0))
            self._logit_bias = float(cfg.get("logit_bias", 0.0))
        except (FileNotFoundError, KeyError):
            pass  # use defaults

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

        # Resize to model input size (BICUBIC matches SigLIP processor default)
        size = self._image_size
        image = image.convert("RGB").resize(
            (size, size), Image.Resampling.BICUBIC
        )

        # Convert to numpy and rescale to [0, 1]
        pixel_values = np.array(image, dtype=np.float32) / 255.0

        # Apply normalization (values from saved preprocessor config)
        pixel_values = (pixel_values - self._image_mean) / self._image_std

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

        # Run ONNX inference (output is already projected + L2 normalized)
        outputs = self.session.run(None, {self.input_name: pixel_values})
        image_features = outputs[0]

        # Compute cosine similarity with pre-computed text embeddings
        # Both are L2-normalized, so dot product = cosine similarity
        similarities = np.dot(image_features, self.text_embeddings.T).squeeze()

        # Apply SigLIP's learned logit_scale and logit_bias (saved during export).
        # GPU path: logits = logit_scale * (image @ text.T) + logit_bias
        logits = similarities * self._logit_scale + self._logit_bias

        # Apply softmax to get probabilities
        exp_sim = np.exp(logits - np.max(logits))
        probs = exp_sim / exp_sim.sum()

        return {
            "normal": float(probs[0]),
            "pneumonia": float(probs[1]),
        }

    @property
    def model_size_mb(self) -> float:
        """Get model file size in MB."""
        return self.model_path.stat().st_size / (1024 * 1024)
