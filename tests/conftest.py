"""
Pytest Configuration and Shared Fixtures

Provides common test fixtures and configuration for PrimaCare AI tests.
"""

import pytest
from unittest.mock import Mock
from PIL import Image
import numpy as np


@pytest.fixture
def mock_medgemma():
    """Create a mock MedGemma model for testing without GPU."""
    model = Mock()
    model.analyze_image = Mock(return_value="Mock analysis result.")
    model.ask = Mock(return_value="Mock text response.")
    model.chat = Mock(return_value="Mock chat response.")
    return model


@pytest.fixture
def mock_medsiglip():
    """Create a mock MedSigLIP classifier for testing without GPU."""
    classifier = Mock()
    classifier.classify = Mock(return_value={
        "normal chest x-ray": 0.8,
        "pneumonia": 0.1,
        "pleural effusion": 0.05,
        "other": 0.05,
    })
    return classifier


@pytest.fixture
def sample_cxr_image():
    """Create a sample chest X-ray sized image."""
    # Typical CXR dimensions
    return Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))


# Skip tests that require GPU
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    # Check GPU availability
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="GPU not available or torch not installed")
        for item in items:
            if "requires_gpu" in item.keywords:
                item.add_marker(skip_gpu)
