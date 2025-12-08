"""
Optimized ADAM Dataset Implementation for Server Environments

This module implements memory-efficient data loading to prevent timeouts
and server disconnections in resource-constrained environments.
"""

import os
import gc
import psutil
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split

from .preprocessing import MedicalImagePreprocessor
from .augmentation import MedicalImageAugmentation
from .utils import load_nifti, get_image_stats

logger = logging.getLogger(__name__)


class OptimizedADAMDataset(Dataset):
    """
    Memory-optimized ADAM Dataset for server environments.
    
    Features:
    - Lazy loading to prevent memory overflow
    - Memory monitoring and cleanup
    - Reduced patch generation
    - Efficient caching strategy
    """
    
    def __init__(
        self,
        data_path: str,
        subjects: List[str],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.25,
        preprocessing: Optional[MedicalImagePreprocessor] = None,
        augmentation: Optional[MedicalImageAugmentation] = None,
        is_training: bool = True,
        load_patches: bool = True,
        max_patches_per_subject: int = 50,
        memory_limit: float = 0.8
    ):
        """
        Initialize optimized ADAM dataset.
        
        Args:
            data_path: Path to the ADAM dataset directory
            subjects: List of subject IDs to include
            patch_size: Size of patches to extract (reduced for memory)
            overlap: Overlap between patches (reduced for fewer patches)
            preprocessing: Preprocessing pipeline
            augmentation: Data augmentation pipeline
            is_training: Whether this is training data
            load_patches: Whether to load patches or full volumes
            max_patches_per_subject: Maximum patches per subject
            memory_limit: Maximum memory usage ratio
        """
        self.data_path = Path(data_path)
        self.subjects = subjects
        self.patch_size = patch_size
        self.overlap = overlap
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.is_training = is_training
        self.load_patches = load_patches
        self.max_patches_per_subject = max_patches_per_subject
        self.memory_limit = memory_limit
        
        # Memory monitoring
        self.total_memory = psutil.virtual_memory().total
        self.memory_threshold = self.total_memory * self.memory_limit
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")
            
        # Load subject data
        self.subject_data = self._load_subject_data()
        
        # Generate patches if needed
        if self.load_patches:
            self.patches = self._generate_patches_optimized()
        else:
            self.patches = None
            
        logger.info(f"Loaded {len(self.subjects)} subjects with {len(self.subject_data)} valid subjects")
        logger.info(f"Memory limit: {self.memory_threshold / (1024**3):.2f} GB")
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        memory_usage = psutil.virtual_memory().used
        return memory_usage < self.memory_threshold
    
    def _cleanup_memory(self):
        """Clean up memory to prevent overflow."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_subject_data(self) -> Dict[str, Dict[str, str]]:
        """Load subject data paths."""
        subject_data = {}
        
        for subject_id in self.subjects:
            subject_path = self.data_path / subject_id
            
            if not subject_path.exists():
                logger.warning(f"Subject {subject_id} not found, skipping")
                continue
                
            # Look for NIfTI files
            tof_file = None
            struct_file = None
            mask_file = None
            
            # Find TOF-MRA file (prefer pre/ directory, fallback to orig/)
            tof_file = None
            for subdir in ["pre", "orig"]:
                subdir_path = subject_path / subdir
                if subdir_path.exists():
                    tof_patterns = ["*TOF*.nii*", "*tof*.nii*", "*TOF.nii*"]
                    for pattern in tof_patterns:
                        tof_files = list(subdir_path.glob(pattern))
                        if tof_files:
                            tof_file = str(tof_files[0])
                            break
                    if tof_file:
                        break
            
            # Find structural file (prefer pre/ directory, fallback to orig/)
            struct_file = None
            for subdir in ["pre", "orig"]:
                subdir_path = subject_path / subdir
                if subdir_path.exists():
                    struct_patterns = ["*struct*.nii*", "*T1*.nii*", "*T2*.nii*", "*FLAIR*.nii*"]
                    for pattern in struct_patterns:
                        struct_files = list(subdir_path.glob(pattern))
                        if struct_files:
                            struct_file = str(struct_files[0])
                            break
                    if struct_file:
                        break
            
            # Find mask file (aneurysm annotation) - at subject level
            mask_file = None
            mask_patterns = ["aneurysms.nii*", "*aneurysm*.nii*", "*mask*.nii*", "*label*.nii*"]
            for pattern in mask_patterns:
                mask_files = list(subject_path.glob(pattern))
                if mask_files:
                    mask_file = str(mask_files[0])
                    break
            
            if tof_file and struct_file:
                subject_data[subject_id] = {
                    "tof": tof_file,
                    "struct": struct_file,
                    "mask": mask_file  # May be None if no aneurysm
                }
            else:
                logger.warning(f"Missing required files for subject {subject_id}")
                
        return subject_data
    
    def _generate_patches_optimized(self) -> List[Dict[str, Any]]:
        """Generate patches with memory optimization."""
        patches = []
        
        for subject_id, files in self.subject_data.items():
            try:
                # Check memory before processing
                if not self._check_memory_usage():
                    logger.warning(f"Memory usage high, cleaning up before processing {subject_id}")
                    self._cleanup_memory()
                
                # Load images
                tof_img = load_nifti(files["tof"])
                struct_img = load_nifti(files["struct"])
                mask_img = load_nifti(files["mask"]) if files["mask"] else None
                
                # Get image dimensions
                img_shape = tof_img.shape
                
                # Calculate patch coordinates with reduced overlap
                patch_coords = self._calculate_patch_coordinates_optimized(img_shape)
                
                # Limit number of patches per subject
                if len(patch_coords) > self.max_patches_per_subject:
                    # Select patches with higher priority for aneurysm regions
                    if mask_img is not None:
                        patch_coords = self._prioritize_aneurysm_patches(
                            patch_coords, mask_img, self.max_patches_per_subject
                        )
                    else:
                        patch_coords = patch_coords[:self.max_patches_per_subject]
                
                for coord in patch_coords:
                    patch_info = {
                        "subject_id": subject_id,
                        "tof_file": files["tof"],
                        "struct_file": files["struct"],
                        "mask_file": files["mask"],
                        "patch_coord": coord,
                        "has_aneurysm": mask_img is not None
                    }
                    patches.append(patch_info)
                
                # Clean up after each subject
                del tof_img, struct_img, mask_img
                self._cleanup_memory()
                    
            except Exception as e:
                logger.error(f"Error processing subject {subject_id}: {e}")
                continue
                
        return patches
    
    def _calculate_patch_coordinates_optimized(self, img_shape: Tuple[int, ...]) -> List[Tuple[int, int, int, int, int, int]]:
        """Calculate patch coordinates with reduced overlap."""
        coords = []
        
        # Increased step size to reduce number of patches
        step_size = [int(s * (1 - self.overlap)) for s in self.patch_size]
        
        for z in range(0, img_shape[0] - self.patch_size[0] + 1, step_size[0]):
            for y in range(0, img_shape[1] - self.patch_size[1] + 1, step_size[1]):
                for x in range(0, img_shape[2] - self.patch_size[2] + 1, step_size[2]):
                    z_end = min(z + self.patch_size[0], img_shape[0])
                    y_end = min(y + self.patch_size[1], img_shape[1])
                    x_end = min(x + self.patch_size[2], img_shape[2])
                    
                    coords.append((z, z_end, y, y_end, x, x_end))
                    
        return coords
    
    def _prioritize_aneurysm_patches(self, patch_coords: List[Tuple], mask_img: np.ndarray, max_patches: int) -> List[Tuple]:
        """Prioritize patches that contain aneurysm regions."""
        patch_scores = []
        
        for coord in patch_coords:
            z_start, z_end, y_start, y_end, x_start, x_end = coord
            patch_mask = mask_img[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Score based on aneurysm content
            aneurysm_ratio = np.sum(patch_mask > 0) / patch_mask.size
            patch_scores.append((coord, aneurysm_ratio))
        
        # Sort by aneurysm content (descending)
        patch_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top patches
        return [coord for coord, _ in patch_scores[:max_patches]]
    
    def __len__(self) -> int:
        """Return dataset length."""
        if self.load_patches:
            return len(self.patches)
        else:
            return len(self.subject_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from dataset with memory optimization."""
        if self.load_patches:
            return self._get_patch_item_optimized(idx)
        else:
            return self._get_volume_item_optimized(idx)
    
    def _get_patch_item_optimized(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get patch item with memory optimization."""
        patch_info = self.patches[idx]
        
        # Check memory before loading
        if not self._check_memory_usage():
            self._cleanup_memory()
        
        # Load images
        tof_img = load_nifti(patch_info["tof_file"])
        struct_img = load_nifti(patch_info["struct_file"])
        mask_img = load_nifti(patch_info["mask_file"]) if patch_info["mask_file"] else None
        
        # Extract patch
        z_start, z_end, y_start, y_end, x_start, x_end = patch_info["patch_coord"]
        
        tof_patch = tof_img[z_start:z_end, y_start:y_end, x_start:x_end]
        struct_patch = struct_img[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_patch = mask_img[z_start:z_end, y_start:y_end, x_start:x_end] if mask_img is not None else None
        
        # Clean up original images
        del tof_img, struct_img, mask_img
        
        # Preprocessing
        if self.preprocessing:
            tof_patch = self.preprocessing(tof_patch)
            struct_patch = self.preprocessing(struct_patch)
            if mask_patch is not None:
                mask_patch = self.preprocessing(mask_patch, is_mask=True)
        
        # Convert to tensors
        tof_tensor = torch.from_numpy(tof_patch).float().unsqueeze(0)
        struct_tensor = torch.from_numpy(struct_patch).float().unsqueeze(0)
        
        if mask_patch is not None:
            mask_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros_like(tof_tensor)
        
        # Clean up numpy arrays
        del tof_patch, struct_patch, mask_patch
        
        # Data augmentation
        if self.augmentation and self.is_training and hasattr(self.augmentation, 'enabled') and self.augmentation.enabled:
            tof_tensor, struct_tensor, mask_tensor = self.augmentation(
                tof_tensor, struct_tensor, mask_tensor
            )
        
        return {
            "image": tof_tensor,  # Primary input (TOF-MRA)
            "struct": struct_tensor,  # Structural image
            "mask": mask_tensor,  # Ground truth mask
            "subject_id": patch_info["subject_id"],
            "has_aneurysm": patch_info["has_aneurysm"]
        }
    
    def _get_volume_item_optimized(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get full volume item with memory optimization."""
        subject_id = list(self.subject_data.keys())[idx]
        files = self.subject_data[subject_id]
        
        # Check memory before loading
        if not self._check_memory_usage():
            self._cleanup_memory()
        
        # Load images
        tof_img = load_nifti(files["tof"])
        struct_img = load_nifti(files["struct"])
        mask_img = load_nifti(files["mask"]) if files["mask"] else None
        
        # Preprocessing
        if self.preprocessing:
            tof_img = self.preprocessing(tof_img)
            struct_img = self.preprocessing(struct_img)
            if mask_img is not None:
                mask_img = self.preprocessing(mask_img, is_mask=True)
        
        # Convert to tensors
        tof_tensor = torch.from_numpy(tof_img).float().unsqueeze(0)
        struct_tensor = torch.from_numpy(struct_img).float().unsqueeze(0)
        
        if mask_img is not None:
            mask_tensor = torch.from_numpy(mask_img).float().unsqueeze(0)
        else:
            mask_tensor = torch.zeros_like(tof_tensor)
        
        # Clean up numpy arrays
        del tof_img, struct_img, mask_img
        
        # Data augmentation
        if self.augmentation and self.is_training and hasattr(self.augmentation, 'enabled') and self.augmentation.enabled:
            tof_tensor, struct_tensor, mask_tensor = self.augmentation(
                tof_tensor, struct_tensor, mask_tensor
            )
        
        return {
            "image": tof_tensor,
            "struct": struct_tensor,
            "mask": mask_tensor,
            "subject_id": subject_id,
            "has_aneurysm": mask_img is not None
        }


class OptimizedADAMDataModule:
    """
    Memory-optimized PyTorch Lightning DataModule for ADAM dataset.
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,  # Reduced default batch size
        num_workers: int = 2,  # Reduced default workers
        patch_size: Tuple[int, int, int] = (64, 64, 64),  # Reduced patch size
        overlap: float = 0.25,  # Reduced overlap
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        preprocessing: Optional[MedicalImagePreprocessor] = None,
        augmentation: Optional[MedicalImageAugmentation] = None,
        load_patches: bool = True,
        max_patches_per_subject: int = 50,
        memory_limit: float = 0.8
    ):
        """
        Initialize optimized ADAM DataModule.
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.overlap = overlap
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.load_patches = load_patches
        self.max_patches_per_subject = max_patches_per_subject
        self.memory_limit = memory_limit
        
        # Get all subjects
        self.subjects = self._get_all_subjects()
        
        # Split subjects
        self.train_subjects, self.val_subjects, self.test_subjects = self._split_subjects()
        
        logger.info(f"Optimized dataset splits - Train: {len(self.train_subjects)}, "
                   f"Val: {len(self.val_subjects)}, Test: {len(self.test_subjects)}")
        logger.info(f"Memory limit: {memory_limit * 100:.1f}%")
    
    def _get_all_subjects(self) -> List[str]:
        """Get all subject IDs from dataset."""
        data_path = Path(self.data_path)
        subjects = []
        
        for subject_dir in data_path.iterdir():
            if subject_dir.is_dir():
                subjects.append(subject_dir.name)
        
        return sorted(subjects)
    
    def _split_subjects(self) -> Tuple[List[str], List[str], List[str]]:
        """Split subjects into train/val/test."""
        train_subjects, temp_subjects = train_test_split(
            self.subjects, 
            test_size=(1 - self.train_split),
            random_state=42
        )
        
        val_size = self.val_split / (self.val_split + self.test_split)
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=(1 - val_size),
            random_state=42
        )
        
        return train_subjects, val_subjects, test_subjects
    
    def train_dataloader(self) -> DataLoader:
        """Create optimized training dataloader."""
        dataset = OptimizedADAMDataset(
            data_path=self.data_path,
            subjects=self.train_subjects,
            patch_size=self.patch_size,
            overlap=self.overlap,
            preprocessing=self.preprocessing,
            augmentation=self.augmentation,
            is_training=True,
            load_patches=self.load_patches,
            max_patches_per_subject=self.max_patches_per_subject,
            memory_limit=self.memory_limit
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,  # Disabled to save memory
            drop_last=True,
            persistent_workers=False,  # Disabled to save memory
            prefetch_factor=2  # Reduced prefetch factor
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create optimized validation dataloader."""
        dataset = OptimizedADAMDataset(
            data_path=self.data_path,
            subjects=self.val_subjects,
            patch_size=self.patch_size,
            overlap=self.overlap,
            preprocessing=self.preprocessing,
            augmentation=None,  # No augmentation for validation
            is_training=False,
            load_patches=self.load_patches,
            max_patches_per_subject=self.max_patches_per_subject,
            memory_limit=self.memory_limit
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,  # Disabled to save memory
            drop_last=False,
            persistent_workers=False,  # Disabled to save memory
            prefetch_factor=2  # Reduced prefetch factor
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create optimized test dataloader."""
        dataset = OptimizedADAMDataset(
            data_path=self.data_path,
            subjects=self.test_subjects,
            patch_size=self.patch_size,
            overlap=self.overlap,
            preprocessing=self.preprocessing,
            augmentation=None,  # No augmentation for testing
            is_training=False,
            load_patches=self.load_patches,
            max_patches_per_subject=self.max_patches_per_subject,
            memory_limit=self.memory_limit
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,  # Disabled to save memory
            drop_last=False,
            persistent_workers=False,  # Disabled to save memory
            prefetch_factor=2  # Reduced prefetch factor
        )
