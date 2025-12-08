"""
Tests for data loading and preprocessing
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os


class TestDataUtils:
    """Test data utility functions"""
    
    def test_get_image_stats(self):
        """Test image statistics calculation"""
        from data.utils import get_image_stats
        
        # Create dummy image
        image = np.random.randn(64, 64, 64)
        
        stats = get_image_stats(image)
        
        assert "shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        
        assert stats["shape"] == (64, 64, 64)
        assert stats["mean"] == pytest.approx(np.mean(image), rel=1e-5)
        assert stats["std"] == pytest.approx(np.std(image), rel=1e-5)
    
    def test_validate_dataset_structure(self):
        """Test dataset validation"""
        from data.utils import validate_dataset
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create valid subject directory
            subject_dir = temp_path / "10001"
            subject_dir.mkdir()
            
            # Create dummy files
            (subject_dir / "TOF.nii").touch()
            (subject_dir / "struct.nii").touch()
            (subject_dir / "aneurysms.nii").touch()
            
            # Validate dataset
            results = validate_dataset(str(temp_path))
            
            assert "valid_subjects" in results
            assert "invalid_subjects" in results
            assert "10001" in results["valid_subjects"]
    
    def test_validate_dataset_missing_files(self):
        """Test dataset validation with missing files"""
        from data.utils import validate_dataset
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create invalid subject directory (missing files)
            subject_dir = temp_path / "10002"
            subject_dir.mkdir()
            
            # Only create one file
            (subject_dir / "TOF.nii").touch()
            
            # Validate dataset
            results = validate_dataset(str(temp_path))
            
            assert "10002" in results["invalid_subjects"]
            assert len(results["missing_files"]) > 0


class TestPreprocessing:
    """Test preprocessing functionality"""
    
    def test_medical_image_preprocessor(self):
        """Test medical image preprocessor"""
        from data.preprocessing import MedicalImagePreprocessor
        
        preprocessor = MedicalImagePreprocessor(
            normalize_method="z_score",
            bias_correction=False,
            noise_reduction=False
        )
        
        # Create dummy image
        image = np.random.randn(64, 64, 64) * 100 + 50
        
        processed = preprocessor(image)
        
        assert processed.shape == image.shape
        assert not np.isnan(processed).any()
        assert not np.isinf(processed).any()
        
        # Check normalization (should be approximately zero mean, unit variance)
        assert abs(np.mean(processed)) < 0.1
        assert abs(np.std(processed) - 1.0) < 0.1
    
    def test_preprocessor_mask_handling(self):
        """Test preprocessor with mask"""
        from data.preprocessing import MedicalImagePreprocessor
        
        preprocessor = MedicalImagePreprocessor()
        
        # Create dummy mask
        mask = np.random.randint(0, 2, (32, 32, 32)).astype(float)
        
        processed_mask = preprocessor(mask, is_mask=True)
        
        assert processed_mask.shape == mask.shape
        assert not np.isnan(processed_mask).any()
        assert not np.isinf(processed_mask).any()
    
    def test_preprocessor_different_methods(self):
        """Test different normalization methods"""
        from data.preprocessing import MedicalImagePreprocessor
        
        image = np.random.randn(32, 32, 32) * 100 + 50
        
        # Test z-score normalization
        preprocessor_z = MedicalImagePreprocessor(normalize_method="z_score")
        processed_z = preprocessor_z(image)
        
        # Test min-max normalization
        preprocessor_mm = MedicalImagePreprocessor(normalize_method="min_max")
        processed_mm = preprocessor_mm(image)
        
        # Test robust normalization
        preprocessor_robust = MedicalImagePreprocessor(normalize_method="robust")
        processed_robust = preprocessor_robust(image)
        
        # All should have same shape
        assert processed_z.shape == image.shape
        assert processed_mm.shape == image.shape
        assert processed_robust.shape == image.shape
        
        # All should be finite
        assert not np.isnan(processed_z).any()
        assert not np.isnan(processed_mm).any()
        assert not np.isnan(processed_robust).any()


class TestAugmentation:
    """Test data augmentation"""
    
    def test_medical_image_augmentation(self):
        """Test medical image augmentation"""
        from data.augmentation import MedicalImageAugmentation
        
        augmentation = MedicalImageAugmentation(
            rotation_range=10,
            translation_range=0.1,
            scale_range=(0.9, 1.1),
            apply_probability=1.0  # Always apply for testing
        )
        
        # Create dummy tensors
        tof_tensor = torch.randn(1, 1, 32, 32, 32)
        struct_tensor = torch.randn(1, 1, 32, 32, 32)
        mask_tensor = torch.randint(0, 2, (1, 1, 32, 32, 32)).float()
        
        # Apply augmentation
        aug_tof, aug_struct, aug_mask = augmentation(
            tof_tensor, struct_tensor, mask_tensor
        )
        
        # Check shapes are preserved
        assert aug_tof.shape == tof_tensor.shape
        assert aug_struct.shape == struct_tensor.shape
        assert aug_mask.shape == mask_tensor.shape
        
        # Check that tensors are finite
        assert not torch.isnan(aug_tof).any()
        assert not torch.isnan(aug_struct).any()
        assert not torch.isnan(aug_mask).any()
    
    def test_augmentation_probability(self):
        """Test augmentation probability"""
        from data.augmentation import MedicalImageAugmentation
        
        # Set probability to 0 (no augmentation)
        augmentation = MedicalImageAugmentation(apply_probability=0.0)
        
        tof_tensor = torch.randn(1, 1, 16, 16, 16)
        struct_tensor = torch.randn(1, 1, 16, 16, 16)
        mask_tensor = torch.randint(0, 2, (1, 1, 16, 16, 16)).float()
        
        # Apply augmentation (should not change anything)
        aug_tof, aug_struct, aug_mask = augmentation(
            tof_tensor, struct_tensor, mask_tensor
        )
        
        # Should be identical (within floating point precision)
        assert torch.allclose(aug_tof, tof_tensor, atol=1e-6)
        assert torch.allclose(aug_struct, struct_tensor, atol=1e-6)
        assert torch.allclose(aug_mask, mask_tensor, atol=1e-6)


class TestDataModule:
    """Test data module functionality"""
    
    def test_data_module_creation(self):
        """Test data module creation"""
        from data import ADAMDataModule
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create dummy subject directories
            for i in range(5):
                subject_dir = temp_path / f"1000{i}"
                subject_dir.mkdir()
                
                # Create dummy files
                (subject_dir / "TOF.nii").touch()
                (subject_dir / "struct.nii").touch()
                (subject_dir / "aneurysms.nii").touch()
            
            # Create data module
            data_module = ADAMDataModule(
                data_path=str(temp_path),
                batch_size=1,
                num_workers=0,  # Use 0 for testing
                patch_size=(32, 32, 32),
                load_patches=False  # Don't load patches for testing
            )
            
            # Test that data module was created successfully
            assert data_module is not None
            assert len(data_module.subjects) == 5


if __name__ == "__main__":
    pytest.main([__file__])
