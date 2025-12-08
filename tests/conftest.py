"""
Pytest configuration and fixtures
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


@pytest.fixture
def dummy_image():
    """Create a dummy 3D medical image"""
    return np.random.randn(64, 64, 64).astype(np.float32)


@pytest.fixture
def dummy_mask():
    """Create a dummy 3D mask"""
    return np.random.randint(0, 2, (64, 64, 64)).astype(np.float32)


@pytest.fixture
def dummy_tensor():
    """Create a dummy tensor for testing"""
    return torch.randn(2, 1, 32, 32, 32)


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing"""
    return {
        "image": torch.randn(2, 1, 32, 32, 32),
        "struct": torch.randn(2, 1, 32, 32, 32),
        "mask": torch.randint(0, 2, (2, 1, 32, 32, 32)).float(),
        "subject_id": ["10001", "10002"],
        "has_aneurysm": [True, False]
    }


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create dummy subject directories
        for i in range(3):
            subject_dir = temp_path / f"1000{i}"
            subject_dir.mkdir()
            
            # Create dummy NIfTI files
            (subject_dir / "TOF.nii").touch()
            (subject_dir / "struct.nii").touch()
            (subject_dir / "aneurysms.nii").touch()
        
        yield str(temp_path)


@pytest.fixture
def gpu_available():
    """Check if GPU is available"""
    return torch.cuda.is_available()


@pytest.fixture
def device(gpu_available):
    """Get appropriate device for testing"""
    return torch.device("cuda" if gpu_available else "cpu")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to tests that don't have any marker
        if not any(marker.name in ["slow", "gpu", "integration", "unit"] 
                  for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
