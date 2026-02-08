"""
Tests for Edge AI module.

All tests use mocks - no ONNX runtime or GPU required.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import numpy as np
from PIL import Image

from src.edge.benchmark import EdgeBenchmarkResult, compare_models


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8))


class TestEdgeBenchmarkResult:
    """Tests for EdgeBenchmarkResult dataclass."""

    def test_to_dict_returns_correct_keys(self):
        result = EdgeBenchmarkResult(
            model_type="edge_int8",
            model_size_mb=50.0,
            avg_latency_ms=100.0,
            p95_latency_ms=150.0,
            memory_peak_mb=200.0,
            accuracy=0.76,
            recall=0.98,
            f1=0.803,
            n_samples=100,
        )
        d = result.to_dict()
        assert d["model_type"] == "edge_int8"
        assert d["model_size_mb"] == 50.0
        assert d["avg_latency_ms"] == 100.0
        assert d["f1"] == 0.803
        assert d["n_samples"] == 100

    def test_to_dict_rounds_values(self):
        result = EdgeBenchmarkResult(
            model_type="test",
            accuracy=0.76543,
            f1=0.80312,
        )
        d = result.to_dict()
        assert d["accuracy"] == 0.7654
        assert d["f1"] == 0.8031


class TestCompareModels:
    """Tests for the compare_models formatting function."""

    def test_compare_models_returns_formatted_table(self):
        gpu = EdgeBenchmarkResult(
            model_type="gpu_full",
            model_size_mb=3500.0,
            avg_latency_ms=50.0,
            p95_latency_ms=80.0,
            memory_peak_mb=4000.0,
            accuracy=0.76,
            recall=0.98,
            f1=0.803,
            n_samples=100,
        )
        edge = EdgeBenchmarkResult(
            model_type="edge_int8",
            model_size_mb=500.0,
            avg_latency_ms=200.0,
            p95_latency_ms=300.0,
            memory_peak_mb=600.0,
            accuracy=0.72,
            recall=0.94,
            f1=0.78,
            n_samples=100,
        )

        table = compare_models(gpu, edge)
        assert "GPU" in table
        assert "Edge" in table
        assert "Latency" in table
        assert "Size reduction" in table


class TestEdgeClassifier:
    """Tests for EdgeClassifier using mocked ONNX runtime."""

    @patch("src.edge.inference.np.load")
    @patch("onnxruntime.InferenceSession")
    def test_classify_pneumonia_returns_probs(self, mock_session_cls, mock_np_load, sample_image):
        """Test that classify_pneumonia returns normal and pneumonia probs."""
        # Mock text embeddings
        mock_np_load.return_value = np.random.randn(2, 768).astype(np.float32)

        # Mock ONNX session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="pixel_values", shape=[1, 3, 448, 448])
        ]
        # Return mock image features
        mock_session.run.return_value = [np.random.randn(1, 768).astype(np.float32)]
        mock_session_cls.return_value = mock_session

        # Mock file existence
        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=50 * 1024 * 1024)

            from src.edge.inference import EdgeClassifier
            classifier = EdgeClassifier("fake_model.onnx", "fake_embeddings.npy")
            result = classifier.classify_pneumonia(sample_image)

        assert "normal" in result
        assert "pneumonia" in result
        assert abs(result["normal"] + result["pneumonia"] - 1.0) < 0.01

    @patch("src.edge.inference.np.load")
    @patch("onnxruntime.InferenceSession")
    def test_edge_classifier_model_size(self, mock_session_cls, mock_np_load):
        """Test model_size_mb property."""
        mock_np_load.return_value = np.random.randn(2, 768).astype(np.float32)
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="pixel_values", shape=[1, 3, 448, 448])
        ]
        mock_session_cls.return_value = mock_session

        with patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=52_428_800)  # 50 MB

            from src.edge.inference import EdgeClassifier
            classifier = EdgeClassifier("fake.onnx", "fake_emb.npy")
            assert abs(classifier.model_size_mb - 50.0) < 0.1


class TestQuantize:
    """Tests for ONNX export and quantization."""

    def test_compute_text_embeddings_has_correct_signature(self):
        """Test that _compute_text_embeddings has the expected parameters."""
        import inspect
        from src.edge.quantize import _compute_text_embeddings

        sig = inspect.signature(_compute_text_embeddings)
        assert "model_id" in sig.parameters
        assert "labels" in sig.parameters

    def test_quantize_onnx_int8_has_correct_signature(self):
        """Test that quantize_onnx_int8 has the expected parameters."""
        import inspect
        from src.edge.quantize import quantize_onnx_int8

        sig = inspect.signature(quantize_onnx_int8)
        assert "onnx_path" in sig.parameters
        assert "output_path" in sig.parameters

    def test_export_medsiglip_onnx_has_correct_signature(self):
        """Test that export_medsiglip_onnx has the expected parameters."""
        import inspect
        from src.edge.quantize import export_medsiglip_onnx

        sig = inspect.signature(export_medsiglip_onnx)
        assert "output_path" in sig.parameters
        assert "model_id" in sig.parameters
