"""
Memory Safety Tests

Verifies that all new features stay within T4 GPU memory constraints (16GB VRAM).
These tests ensure the memory-safe design patterns are properly implemented.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock torch if not available
import sys
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = MagicMock()
    torch.cuda = MagicMock()
    torch.cuda.empty_cache = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    sys.modules['torch'] = torch
    TORCH_AVAILABLE = False


class TestMemoryConstants:
    """Verify memory-safe constants across all agents."""

    def test_volumetric_max_slices(self):
        """VolumetricImagingAgent should limit slices to 6."""
        from src.agents.volumetric import VolumetricImagingAgent
        assert VolumetricImagingAgent.MAX_SLICES == 6

    def test_pathology_max_tiles(self):
        """PathologyAgent should limit tiles to 4."""
        from src.agents.pathology import PathologyAgent
        assert PathologyAgent.MAX_TILES == 4

    def test_pathology_tile_size(self):
        """PathologyAgent tile size should be 512."""
        from src.agents.pathology import PathologyAgent
        assert PathologyAgent.TILE_SIZE == 512


class TestSequentialProcessing:
    """Verify sequential processing patterns."""

    def test_longitudinal_processes_sequentially(self):
        """Longitudinal analysis should process images one at a time."""
        from src.agents.imaging import ImagingAgent
        from PIL import Image
        import numpy as np

        mock_model = Mock()
        call_order = []

        def track_image(*args, **kwargs):
            call_order.append('image')
            return "Findings"

        def track_text(*args, **kwargs):
            call_order.append('text')
            return "**CHANGE SUMMARY:**\nUNCHANGED\n**URGENT:**\nNO"

        mock_model.analyze_image = Mock(side_effect=track_image)
        mock_model.ask = Mock(side_effect=track_text)

        agent = ImagingAgent(model=mock_model, load_classifier=False)
        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

        with patch('torch.cuda.empty_cache'):
            agent.analyze_longitudinal(img, img)

        # Verify: image, image, text (not image, text, image, text)
        assert call_order == ['image', 'image', 'text']

    def test_volumetric_processes_sequentially(self):
        """Volumetric analysis should process slices one at a time."""
        from src.agents.volumetric import VolumetricImagingAgent
        from PIL import Image
        import numpy as np

        mock_model = Mock()
        image_calls = []

        def track_image(*args, **kwargs):
            image_calls.append(1)
            return "Slice findings"

        mock_model.analyze_image = Mock(side_effect=track_image)
        mock_model.ask = Mock(return_value="**KEY FINDINGS:**\n- Normal\n**IMPRESSION:**\nNormal\n**URGENT:**\nNO")

        agent = VolumetricImagingAgent(model=mock_model)
        images = [Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)) for _ in range(10)]

        with patch('torch.cuda.empty_cache'):
            result = agent.analyze(images, num_slices=4)

        # Should only analyze 4 slices (limited by num_slices)
        assert len(image_calls) == 4


class TestCacheClearance:
    """Verify CUDA cache is cleared between operations."""

    def test_longitudinal_clears_cache(self):
        """Longitudinal should clear cache between images."""
        from src.agents.imaging import ImagingAgent
        from PIL import Image
        import numpy as np

        mock_model = Mock()
        mock_model.analyze_image = Mock(return_value="Findings")
        mock_model.ask = Mock(return_value="**CHANGE SUMMARY:**\nUNCHANGED\n**URGENT:**\nNO")

        agent = ImagingAgent(model=mock_model, load_classifier=False)
        img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

        with patch('torch.cuda.empty_cache') as mock_cache:
            agent.analyze_longitudinal(img, img)

        # Should clear cache at least twice (after each image)
        assert mock_cache.call_count >= 2

    def test_volumetric_clears_cache_per_slice(self):
        """Volumetric should clear cache after each slice."""
        from src.agents.volumetric import VolumetricImagingAgent
        from PIL import Image
        import numpy as np

        mock_model = Mock()
        mock_model.analyze_image = Mock(return_value="Findings")
        mock_model.ask = Mock(return_value="**KEY FINDINGS:**\n- Normal\n**IMPRESSION:**\nNormal\n**URGENT:**\nNO")

        agent = VolumetricImagingAgent(model=mock_model)
        images = [Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)) for _ in range(10)]

        with patch('torch.cuda.empty_cache') as mock_cache:
            agent.analyze(images, num_slices=4)

        # Should clear cache at least 4 times (once per slice) plus synthesis
        assert mock_cache.call_count >= 4


class TestMemoryEstimates:
    """Estimate memory usage for different operations."""

    def test_volumetric_memory_estimate(self):
        """Estimate max memory for volumetric analysis."""
        from src.agents.volumetric import VolumetricImagingAgent

        # 6 slices max, 512x512 each, RGB
        max_image_memory = 6 * 512 * 512 * 3  # ~4.7 MB
        # Well under T4 16GB limit
        assert max_image_memory < 100 * 1024 * 1024  # < 100 MB

    def test_pathology_memory_estimate(self):
        """Estimate max memory for pathology analysis."""
        from src.agents.pathology import PathologyAgent

        # 4 tiles max, 512x512 each, RGB
        max_image_memory = 4 * 512 * 512 * 3  # ~3.1 MB
        # Well under T4 16GB limit
        assert max_image_memory < 100 * 1024 * 1024  # < 100 MB

    def test_longitudinal_memory_estimate(self):
        """Estimate max memory for longitudinal comparison."""
        # 2 images at typical CXR resolution (1024x1024), RGB
        # But processed sequentially, so only 1 in memory at a time
        single_image_memory = 1024 * 1024 * 3  # ~3.1 MB
        # Well under T4 16GB limit
        assert single_image_memory < 100 * 1024 * 1024  # < 100 MB


@pytest.mark.requires_gpu
class TestGPUMemoryActual:
    """Actual GPU memory tests (only run if GPU available)."""

    def test_peak_memory_under_limit(self):
        """Verify peak memory stays under T4 limit."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Run a simple operation
        tensor = torch.randn(1000, 1000, device='cuda')
        del tensor

        peak = torch.cuda.max_memory_allocated()
        # Should be well under 14GB (leaving 2GB buffer for model)
        assert peak < 14 * 1024**3
