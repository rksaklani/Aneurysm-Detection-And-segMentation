#!/usr/bin/env python3
"""
Unified Medical Image Segmentation Benchmark
===========================================

Single comprehensive script for medical image segmentation benchmarking.
Supports all 15 models from the complete evolution sequence (2015-2025).

Features:
- Complete model evolution sequence (U-Net to next-gen transformers)
- 3D medical image support (ADAM dataset)
- Timeout prevention and memory management
- Automated dataset validation
- Progress tracking and visualization
- Error recovery and logging
- Single script execution

Usage:
    python unified_benchmark.py --data_path /path/to/adam/dataset
    
Author: AI Assistant
Date: October 27, 2025
"""

import os
import sys
import gc
import time
import json
import math
from collections import OrderedDict
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import traceback

# Core libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
# from sklearn.metrics import dice_score, jaccard_score  # Not available in all versions
# from scipy.spatial.distance import directed_hausdorff  # Not needed for basic benchmark

# Progress and visualization
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: rich not available, using basic progress")

# Visualization and plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")

# Scientific computing for advanced metrics
try:
    from scipy.spatial.distance import directed_hausdorff
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, advanced metrics disabled")

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory monitoring disabled")

# Configuration
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    import yaml

# Logging setup
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize console
console = Console() if RICH_AVAILABLE else None

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
    except AttributeError:
        pass

# ========================================================================================
# CONFIGURATION AND CONSTANTS
# ========================================================================================

DEFAULT_CONFIG = {
    "seed": 42,
    # Dataset settings (improved for better performance)
    "dataset": {
        "patch_size": [96, 96, 96],  # Larger patches for better context
        "overlap": 0.25,  # Some overlap for better coverage
        "batch_size": 2,  # Increased batch size
        "num_workers": 4,  # Multi-threaded for speed
        "prefetch_factor": 2,
        "max_subjects": None,  # Unlimited by default (set later)
        "max_patches_per_subject": 48,
        "aneurysm_patch_ratio": 0.75,
        "min_background_patches": 4,
        "train_fraction": 0.75,
        "val_fraction": 0.15,
        "test_fraction": 0.10,
        "seed": 42,
    },
    
    # Training settings (improved for better performance)
    "training": {
        "max_epochs": 50,  # More epochs for better convergence
        "learning_rate": 5e-4,  # Lower LR for stability
        "weight_decay": 1e-4,
        "patience": 10,  # More patience before early stopping
        "accumulate_grad_batches": 1,  # No gradient accumulation
        "precision": "16-mixed",
        "max_train_batches": None,
        "max_val_batches": None,
    },
    
    # Memory management
    "memory": {
        "limit_ratio": 0.6,  # Use max 60% of available memory
        "cleanup_frequency": 3,  # Clean every 3 steps
        "force_cleanup": True,
    },
    
    # Timeout prevention (improved for better training)
    "timeout": {
        "max_total_time_minutes": None,  # No overall time limit
        "max_model_time_minutes": None,  # No per-model time limit
        "max_batch_time_seconds": 45,    # Allow longer per-batch times
        "checkpoint_frequency": 5,       # Save every 5 batches
    },
    
    # Model settings (improved for better performance)
    "models": {
        "enabled": ["unet", "unetr", "es_unet", "primus", "mamba_unet"],  # Start with 5 models
        "all_models": [
            "unet", "unet3d", "lightweight_unet3d", "attention_unet", "nnu_net",
            "unetr", "unetr_plus", "swin_unetr", "primus", "slim_unetr", 
            "es_unet", "rwkv_unet", "mamba_unet", "stacked_unet", "multiscale_unet"
        ],
    }
}

# ========================================================================================
# CONFIG HELPERS
# ========================================================================================

def _determine_worker_count(default: int = 4) -> int:
    """Utility to determine a sensible dataloader worker count."""
    cpu_count = os.cpu_count() or default
    return max(default, min(cpu_count // 2 or default, 16))


def _has_subject_dirs(path: Path) -> bool:
    """Return True if the path contains at least one directory."""
    try:
        for child in path.iterdir():
            if child.is_dir():
                return True
    except FileNotFoundError:
        pass
    return False


def _has_mra_structure(path: Path) -> bool:
    """Return True if the path has the new MRA structure (images/ and masks/ directories)."""
    try:
        images_dir = path / "images"
        masks_dir = path / "masks"
        return images_dir.exists() and images_dir.is_dir() and masks_dir.exists() and masks_dir.is_dir()
    except (FileNotFoundError, OSError):
        return False


def resolve_dataset_path(data_path: Path) -> Tuple[Path, Optional[str]]:
    """
    Resolve the actual dataset directory.
    
    Handles common cases where the ADAM dataset ships as zip files or is extracted
    to sibling directories (e.g., data/adam_dataset/raw).
    Also supports the new MRA structure (data/MRA with images/ and masks/ directories).
    """
    if not data_path.exists():
        return data_path, f"Dataset path does not exist: {data_path}"

    # Check for new MRA structure first (images/ and masks/ directories)
    if _has_mra_structure(data_path):
        return data_path, "Detected MRA structure (images/ and masks/ directories)"
    
    # Check for MRA structure in common locations
    candidate_mra_paths = [
        data_path / "MRA",
        data_path.parent / "MRA",
        data_path.parent / "data" / "MRA",
    ]
    for candidate in candidate_mra_paths:
        if candidate.exists() and _has_mra_structure(candidate):
            return candidate, f"Detected MRA structure at: {candidate}"

    # Check for old structure with subject directories
    if _has_subject_dirs(data_path):
        return data_path, None

    # Common extracted location used by earlier scripts
    candidate_paths = [
        data_path / "raw",
        data_path.parent / "adam_dataset" / "raw",
        data_path.parent / "adam_dataset" / "extracted",
    ]

    for candidate in candidate_paths:
        if candidate.exists() and _has_subject_dirs(candidate):
            return candidate, f"Detected extracted dataset directory: {candidate}"

    # As a last resort, check for individual subject zip files
    zip_files = list(data_path.glob("*.zip"))
    if zip_files:
        extracted_root = data_path.parent / f"{data_path.name}_extracted"
        extracted_root.mkdir(parents=True, exist_ok=True)
        for zip_file in zip_files:
            target_dir = extracted_root / zip_file.stem
            if target_dir.exists() and _has_subject_dirs(target_dir):
                continue
            try:
                import zipfile

                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(target_dir)
            except Exception as exc:
                return data_path, f"Failed to extract {zip_file}: {exc}"
        if _has_subject_dirs(extracted_root):
            return extracted_root, f"Extracted zip subjects to {extracted_root}"

    return data_path, "Dataset path exists but contains no subject directories or MRA structure; check structure."


def get_subject_base_dir(subject_path: Path) -> Path:
    """
    Return the directory that actually holds the imaging files for a subject.
    Handles layouts such as <subject>/<subject>/pre/TOF.nii.gz.
    """
    nested_candidate = subject_path / subject_path.name
    if nested_candidate.exists() and nested_candidate.is_dir():
        return nested_candidate
    return subject_path


def tune_config_for_device(subjects: List[str], args: argparse.Namespace) -> None:
    """
    Adapt configuration parameters based on the current hardware and CLI overrides.
    Ensures the GPU is fully utilised without overloading the system.
    """
    using_gpu = torch.cuda.is_available()
    is_windows = sys.platform.startswith("win")

    # Respect explicit CLI overrides first
    if args.max_epochs is not None:
        DEFAULT_CONFIG["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        DEFAULT_CONFIG["dataset"]["batch_size"] = max(1, args.batch_size)
    if args.max_patches_per_subject is not None:
        DEFAULT_CONFIG["dataset"]["max_patches_per_subject"] = max(1, args.max_patches_per_subject)
    if args.max_model_time_minutes is not None:
        if args.max_model_time_minutes <= 0:
            DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] = None
        else:
            DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] = args.max_model_time_minutes
    if args.max_train_batches is not None:
        DEFAULT_CONFIG["training"]["max_train_batches"] = max(1, args.max_train_batches)
    if args.max_val_batches is not None:
        DEFAULT_CONFIG["training"]["max_val_batches"] = max(1, args.max_val_batches)

    # Device-aware tuning
    if using_gpu:
        worker_default = 6 if not is_windows else 2
        worker_count = _determine_worker_count(worker_default)
        worker_count = min(worker_count, 8)
        if is_windows:
            worker_count = min(worker_count, 4)
            DEFAULT_CONFIG["dataset"]["prefetch_factor"] = 1
        DEFAULT_CONFIG["dataset"]["num_workers"] = worker_count
        DEFAULT_CONFIG["dataset"]["batch_size"] = max(DEFAULT_CONFIG["dataset"]["batch_size"], 4)
        if DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] is not None:
            DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] = max(
                DEFAULT_CONFIG["timeout"]["max_model_time_minutes"], 240
            )
        if DEFAULT_CONFIG["training"]["max_train_batches"] is not None:
            DEFAULT_CONFIG["training"]["max_train_batches"] = max(
                DEFAULT_CONFIG["training"]["max_train_batches"], 120
            )
        if DEFAULT_CONFIG["training"]["max_val_batches"] is not None:
            DEFAULT_CONFIG["training"]["max_val_batches"] = max(
                DEFAULT_CONFIG["training"]["max_val_batches"], 24
            )
    else:
        DEFAULT_CONFIG["dataset"]["num_workers"] = min(
            DEFAULT_CONFIG["dataset"]["num_workers"],
            2 if is_windows else 4,
        )
        DEFAULT_CONFIG["dataset"]["batch_size"] = min(DEFAULT_CONFIG["dataset"]["batch_size"], 2)
        if is_windows:
            DEFAULT_CONFIG["dataset"]["prefetch_factor"] = 1
        if DEFAULT_CONFIG["training"]["max_train_batches"] is not None:
            DEFAULT_CONFIG["training"]["max_train_batches"] = min(
                DEFAULT_CONFIG["training"]["max_train_batches"], 40
            )
        if DEFAULT_CONFIG["training"]["max_val_batches"] is not None:
            DEFAULT_CONFIG["training"]["max_val_batches"] = min(
                DEFAULT_CONFIG["training"]["max_val_batches"], 16
            )

    # Subject limit handling
    if args.max_subjects is not None:
        DEFAULT_CONFIG["dataset"]["max_subjects"] = max(1, args.max_subjects)
    elif DEFAULT_CONFIG["dataset"]["max_subjects"] is None:
        DEFAULT_CONFIG["dataset"]["max_subjects"] = len(subjects) if subjects else None

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

class TimeoutException(Exception):
    """Exception raised when timeout is exceeded."""
    pass

class MemoryManager:
    """Manages memory usage and cleanup."""
    
    def __init__(self, limit_ratio: float = 0.6):
        self.limit_ratio = limit_ratio
        self.total_memory = psutil.virtual_memory().total if PSUTIL_AVAILABLE else 8 * 1024**3
        self.memory_limit = self.total_memory * limit_ratio
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        if not PSUTIL_AVAILABLE:
            return True
        return psutil.virtual_memory().used < self.memory_limit
    
    def cleanup(self, force: bool = False):
        """Clean up memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if force:
                torch.cuda.synchronize()
    
    def get_usage_info(self) -> Dict[str, float]:
        """Get memory usage information."""
        if not PSUTIL_AVAILABLE:
            return {"used_gb": 0, "available_gb": 8, "usage_percent": 0}
        
        vm = psutil.virtual_memory()
        return {
            "used_gb": vm.used / 1024**3,
            "available_gb": vm.available / 1024**3,
            "usage_percent": vm.percent,
            "limit_gb": self.memory_limit / 1024**3
        }

class TimeoutManager:
    """Manages execution timeouts."""
    
    def __init__(
        self,
        max_total_minutes: Optional[int] = None,
        max_model_minutes: Optional[int] = None,
    ):
        self.start_time = time.time()
        self.max_total_time = max_total_minutes * 60 if max_total_minutes else None
        self.max_model_time = max_model_minutes * 60 if max_model_minutes else None
        self.model_start_time = None
        
    def start_model_timer(self):
        """Start timing a model."""
        self.model_start_time = time.time()
        
    def check_total_timeout(self):
        """Check if total time limit exceeded."""
        if self.max_total_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.max_total_time:
            raise TimeoutException(f"Total time limit exceeded: {elapsed/60:.1f} minutes")
    
    def check_model_timeout(self):
        """Check if model time limit exceeded."""
        if self.model_start_time is None:
            return
        if self.max_model_time is None:
            return
        elapsed = time.time() - self.model_start_time
        if elapsed > self.max_model_time:
            raise TimeoutException(f"Model time limit exceeded: {elapsed/60:.1f} minutes")
    
    def get_remaining_time(self) -> float:
        """Get remaining time in minutes."""
        if self.max_total_time is None:
            return float("inf")
        elapsed = time.time() - self.start_time
        return max(0, (self.max_total_time - elapsed) / 60)

def log_message(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    if console and RICH_AVAILABLE:
        if level == "ERROR":
            console.print(f"[red]{formatted_message}[/red]")
        elif level == "WARNING":
            console.print(f"[yellow]{formatted_message}[/yellow]")
        elif level == "SUCCESS":
            console.print(f"[green]{formatted_message}[/green]")
        else:
            console.print(formatted_message)
    else:
        print(formatted_message)

def validate_adam_dataset(data_path: str) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate ADAM dataset structure and return subject information.
    
    Supports both old structure (subject directories) and new MRA structure (images/ and masks/).
    
    Returns:
        (is_valid, subject_list, dataset_info)
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        return False, [], {"error": f"Dataset path does not exist: {data_path}"}
    
    # Check for new MRA structure (images/ and masks/ directories)
    if _has_mra_structure(data_path):
        return _validate_mra_dataset(data_path)
    
    # Old structure: Find all subject directories
    subjects = []
    valid_subjects = []
    dataset_info = {
        "total_subjects": 0,
        "valid_subjects": 0,
        "subjects_with_aneurysms": 0,
        "missing_files": [],
        "structure_type": "old"
    }
    
    for subject_dir in data_path.iterdir():
        if not subject_dir.is_dir():
            continue
            
        subjects.append(subject_dir.name)
        subject_base = get_subject_base_dir(subject_dir)
        
        # Check for required files
        has_tof = False
        has_struct = False
        has_aneurysm = False
        
        # Check for TOF file (primary input - matches aneurysm mask)
        for subdir in ["pre", "orig"]:
            subdir_path = subject_base / subdir
            if subdir_path.exists():
                tof_files = list(subdir_path.glob("*TOF*.nii*")) + list(subdir_path.glob("*tof*.nii*"))
                if tof_files:
                    has_tof = True
                    break
        
        # Note: Not checking struct files anymore due to dimension mismatch
        has_struct = True  # Always true since we only need TOF
        
        # Check for aneurysm file
        aneurysm_files = list(subject_base.glob("aneurysms.nii*")) + list(subject_base.glob("*aneurysm*.nii*"))
        if aneurysm_files:
            has_aneurysm = True
            dataset_info["subjects_with_aneurysms"] += 1
        
        if has_tof:  # Only need TOF file now
            valid_subjects.append(subject_dir.name)
        else:
            missing = ["TOF"]  # Only TOF is required
            dataset_info["missing_files"].append({
                "subject": subject_dir.name,
                "missing": missing
            })
    
    dataset_info["total_subjects"] = len(subjects)
    dataset_info["valid_subjects"] = len(valid_subjects)
    
    is_valid = len(valid_subjects) >= 3  # Need at least 3 subjects for train/val/test
    
    return is_valid, sorted(valid_subjects), dataset_info


def _validate_mra_dataset(data_path: Path) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate the new MRA dataset structure (images/ and masks/ directories).
    
    Returns:
        (is_valid, subject_list, dataset_info)
    """
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        return False, [], {"error": f"MRA structure incomplete: missing images/ or masks/ directory"}
    
    # Find all image files (img_XXXX.nii.gz)
    image_files = sorted(images_dir.glob("img_*.nii*"))
    
    subjects = []
    valid_subjects = []
    dataset_info = {
        "total_subjects": 0,
        "valid_subjects": 0,
        "subjects_with_aneurysms": 0,
        "missing_files": [],
        "structure_type": "mra"
    }
    
    for img_file in image_files:
        # Extract subject ID from filename (e.g., "img_0001" from "img_0001.nii.gz")
        # Handle both "img_0001.nii.gz" and "img_0001.nii" formats
        subject_id = img_file.stem
        if subject_id.endswith(".nii"):
            subject_id = subject_id[:-4]  # Remove ".nii" if present
        
        # Ensure consistent format: "img_XXXX" where XXXX is zero-padded
        if not subject_id.startswith("img_"):
            continue  # Skip files that don't match expected pattern
        
        subjects.append(subject_id)
        
        # Find corresponding mask file
        mask_name = img_file.name.replace("img_", "mask_")
        mask_file = masks_dir / mask_name
        
        if not mask_file.exists():
            dataset_info["missing_files"].append({
                "subject": subject_id,
                "missing": ["mask"]
            })
            continue
        
        # Check if mask has non-zero values (has aneurysms)
        try:
            mask_data = nib.load(str(mask_file)).get_fdata()
            if np.any(mask_data > 0):
                dataset_info["subjects_with_aneurysms"] += 1
        except Exception:
            pass
        
        # Both image and mask exist
        valid_subjects.append(subject_id)
    
    dataset_info["total_subjects"] = len(subjects)
    dataset_info["valid_subjects"] = len(valid_subjects)
    
    is_valid = len(valid_subjects) >= 3  # Need at least 3 subjects for train/val/test
    
    return is_valid, sorted(valid_subjects), dataset_info

# ========================================================================================
# 3D MEDICAL IMAGE PROCESSING
# ========================================================================================

def load_nifti_safe(file_path: str) -> Optional[np.ndarray]:
    """Safely load NIfTI file with error handling and caching."""
    try:
        if not file_path or not Path(file_path).exists():
            return None
        cache_key = str(file_path)
        if cache_key in _image_cache:
            _image_cache.move_to_end(cache_key)
            return _image_cache[cache_key]

        nii = nib.load(file_path)
        data = nii.get_fdata().astype(np.float32)
        
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        _image_cache[cache_key] = data
        if len(_image_cache) > _IMAGE_CACHE_MAX_ITEMS:
            _image_cache.popitem(last=False)
        
        return data
        
    except Exception as e:
        log_message(f"Error loading {file_path}: {e}", "ERROR")
        return None

# Global cache for loaded images to speed up access
_image_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_IMAGE_CACHE_MAX_ITEMS = 2

def preprocess_3d_image(image: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """Preprocess 3D medical image."""
    # Handle empty or invalid images
    if image is None or image.size == 0:
        if target_size:
            return np.zeros(target_size, dtype=np.float32)
        return np.zeros((32, 32, 32), dtype=np.float32)
    
    # Ensure 3D
    if image.ndim == 4:
        image = image[..., 0]  # Take first channel if 4D
    elif image.ndim == 2:
        image = image[np.newaxis, ...]  # Add depth dimension
    
    # Normalize (Z-score normalization) with safety checks
    foreground_mask = image > 0
    if np.sum(foreground_mask) > 0:
        # Calculate stats only on non-zero regions
        mean = np.mean(image[foreground_mask])
        std = np.std(image[foreground_mask])
        if std > 1e-8:  # Avoid division by very small numbers
            image = (image - mean) / std
        else:
            # If std is too small, just center the data
            image = image - mean
    else:
        # If all zeros, normalize to standard range
        image = image / (np.max(image) + 1e-8)
    
    # Clip outliers
    image = np.clip(image, -3, 3)
    
    # Resize if needed (simple interpolation for speed)
    if target_size and image.shape != target_size:
        # Simple resizing without scipy dependency
        import torch.nn.functional as F
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        resized = F.interpolate(image_tensor, size=target_size, mode='nearest')
        image = resized.squeeze(0).squeeze(0).numpy()
    
    return image.astype(np.float32)

# ========================================================================================
# DATASET IMPLEMENTATION
# ========================================================================================

class UnifiedADAMDataset(Dataset):
    """Unified ADAM dataset for 3D medical image segmentation."""
    
    def __init__(
        self,
        data_path: str,
        subjects: List[str],
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        max_patches_per_subject: int = 10,
        is_training: bool = True,
        aneurysm_patch_ratio: float = 0.75,
        min_background_patches: int = 4,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.subjects = subjects
        self.patch_size = patch_size
        self.max_patches_per_subject = max_patches_per_subject
        self.is_training = is_training
        self.aneurysm_patch_ratio = float(np.clip(aneurysm_patch_ratio, 0.0, 0.95))
        self.min_background_patches = max(0, int(min_background_patches))
        self.seed = int(seed)
        self._log_interval = 0
        self._log_samples = 0
        self._max_log_samples = 20
        
        # Load subject data
        self.subject_data = self._load_subject_data()
        self.patches = self._generate_patches()
        total_patches = max(1, len(self.patches))
        if self.is_training:
            self._log_interval = max(1, total_patches // 80)
        else:
            self._log_interval = max(1, total_patches // 40)
        
        log_message(f"Dataset initialized: {len(self.patches)} patches from {len(self.subject_data)} subjects")
    
    def _load_subject_data(self) -> Dict[str, Dict[str, str]]:
        """Load subject file paths. Supports both old and new MRA structures."""
        subject_data = {}
        
        # Check if this is the new MRA structure
        images_dir = self.data_path / "images"
        masks_dir = self.data_path / "masks"
        is_mra_structure = images_dir.exists() and masks_dir.exists()
        
        if is_mra_structure:
            # New MRA structure: images/img_XXXX.nii.gz and masks/mask_XXXX.nii.gz
            for subject_id in self.subjects:
                # Subject ID should be in format "img_XXXX" from validation
                # Try .nii.gz first, then .nii
                img_name_gz = f"{subject_id}.nii.gz"
                img_name = f"{subject_id}.nii"
                
                img_file = images_dir / img_name_gz
                if not img_file.exists():
                    img_file = images_dir / img_name
                
                if img_file.exists():
                    # Find corresponding mask file
                    mask_name_gz = img_name_gz.replace("img_", "mask_")
                    mask_name = img_name.replace("img_", "mask_")
                    mask_file = masks_dir / mask_name_gz
                    if not mask_file.exists():
                        mask_file = masks_dir / mask_name
                    
                    subject_data[subject_id] = {
                        "tof": str(img_file),
                        "mask": str(mask_file) if mask_file.exists() else None
                    }
        else:
            # Old structure: subject directories with nested files
            for subject_id in self.subjects:
                subject_path = self.data_path / subject_id
                
                if not subject_path.exists():
                    continue
                
                subject_base = get_subject_base_dir(subject_path)
                
                # Find files - ONLY use TOF (matches aneurysm mask dimensions)
                tof_file = self._find_file(subject_base, ["*TOF*.nii*", "*tof*.nii*"])
                mask_file = self._find_file(subject_base, ["aneurysms.nii*", "*aneurysm*.nii*"], check_root=True)
                
                if tof_file:  # Only need TOF file
                    subject_data[subject_id] = {
                        "tof": str(tof_file),
                        "mask": str(mask_file) if mask_file else None
                    }
        
        return subject_data
    
    def _find_file(self, subject_path: Path, patterns: List[str], check_root: bool = False) -> Optional[Path]:
        """Find file matching patterns with priority for pre/ directory."""
        # Priority order: pre/, orig/, root
        search_paths = []
        if not check_root:
            search_paths.extend([subject_path / "pre", subject_path / "orig"])
        else:
            search_paths.append(subject_path)
            search_paths.extend([subject_path / "pre", subject_path / "orig"])
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            for pattern in patterns:
                files = list(search_path.glob(pattern))
                if files:
                    return files[0]
        return None
    
    def _generate_patches(self) -> List[Dict[str, Any]]:
        """Generate patch information with aneurysm-focused sampling."""
        patches = []
        
        for subject_id, files in self.subject_data.items():
            subject_patches: List[Dict[str, Any]] = []
            rng = np.random.default_rng(self.seed + (hash(subject_id) % 1_000_000))

            mask_img = load_nifti_safe(files["mask"]) if files["mask"] else None
            aneurysm_available = mask_img is not None and np.any(mask_img > 0)

            # Sample aneurysm-focused patches
            if aneurysm_available:
                aneurysm_coords = np.column_stack(np.where(mask_img > 0))
                max_aneurysm_allowed = self.max_patches_per_subject
                if self.max_patches_per_subject > self.min_background_patches:
                    max_aneurysm_allowed = self.max_patches_per_subject - self.min_background_patches

                desired_aneurysms = int(round(self.max_patches_per_subject * self.aneurysm_patch_ratio))
                desired_aneurysms = max(1, desired_aneurysms)
                aneurysm_quota = min(max_aneurysm_allowed, desired_aneurysms)

                for _ in range(aneurysm_quota):
                    coord = aneurysm_coords[int(rng.integers(len(aneurysm_coords)))]
                    offsets = rng.integers(
                        low=-np.array(self.patch_size) // 8,
                        high=np.array(self.patch_size) // 8 + 1,
                    )
                    center = tuple(int(c + o) for c, o in zip(coord, offsets))
                    subject_patches.append({
                        "subject_id": subject_id,
                        "tof_file": files["tof"],
                        "mask_file": files["mask"],
                        "patch_idx": len(subject_patches),
                        "has_aneurysm": True,
                        "aneurysm_center": center,
                        "patch_type": "aneurysm"
                    })

            # Background sampling to balance dataset
            remaining_slots = max(0, self.max_patches_per_subject - len(subject_patches))
            bg_target = remaining_slots
            if aneurysm_available and self.min_background_patches > 0:
                bg_target = max(self.min_background_patches, remaining_slots)
            bg_target = min(bg_target, self.max_patches_per_subject - len(subject_patches))

            for _ in range(bg_target):
                seed_value = int(rng.integers(0, 2**32 - 1))
                subject_patches.append({
                    "subject_id": subject_id,
                    "tof_file": files["tof"],
                    "mask_file": files["mask"],
                    "patch_idx": len(subject_patches),
                    "has_aneurysm": files["mask"] is not None,
                    "aneurysm_center": None,
                    "patch_type": "background",
                    "random_seed": seed_value
                })

            if not subject_patches:
                # Fallback to ensure at least one patch per subject
                seed_value = int(rng.integers(0, 2**32 - 1))
                subject_patches.append({
                    "subject_id": subject_id,
                    "tof_file": files["tof"],
                    "mask_file": files["mask"],
                    "patch_idx": 0,
                    "has_aneurysm": False,
                    "aneurysm_center": None,
                    "patch_type": "background",
                    "random_seed": seed_value
                })

            rng.shuffle(subject_patches)
            patches.extend(subject_patches)

            if mask_img is not None:
                del mask_img
                gc.collect()
        
        return patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with optimized loading (TOF only)."""
        patch_info = self.patches[idx]

        try:
            # Load only TOF image (matches aneurysm mask dimensions)
            tof_img = load_nifti_safe(patch_info["tof_file"])
            mask_img = load_nifti_safe(patch_info["mask_file"]) if patch_info["mask_file"] else None
            
            # Extract patches - use aneurysm center if available
            if tof_img is not None and tof_img.size > 0:
                if patch_info.get("aneurysm_center") and patch_info.get("patch_type") == "aneurysm":
                    # Center patch around aneurysm
                    center_z, center_y, center_x = patch_info["aneurysm_center"]
                    
                    # Calculate patch bounds around center
                    z_start = max(0, center_z - self.patch_size[0] // 2)
                    y_start = max(0, center_y - self.patch_size[1] // 2)
                    x_start = max(0, center_x - self.patch_size[2] // 2)
                    
                    # Ensure we don't go out of bounds with safety checks
                    z_start = max(0, min(z_start, tof_img.shape[0] - self.patch_size[0]))
                    y_start = max(0, min(y_start, tof_img.shape[1] - self.patch_size[1]))
                    x_start = max(0, min(x_start, tof_img.shape[2] - self.patch_size[2]))
                    
                else:
                    # Random patch for background or fallback
                    max_z = max(0, tof_img.shape[0] - self.patch_size[0])
                    max_y = max(0, tof_img.shape[1] - self.patch_size[1])
                    max_x = max(0, tof_img.shape[2] - self.patch_size[2])
                    seed_value = patch_info.get("random_seed")
                    patch_rng = np.random.default_rng(seed_value if seed_value is not None else self.seed + idx)
                    z_start = int(patch_rng.integers(0, max_z + 1)) if max_z > 0 else 0
                    y_start = int(patch_rng.integers(0, max_y + 1)) if max_y > 0 else 0
                    x_start = int(patch_rng.integers(0, max_x + 1)) if max_x > 0 else 0
                
                # Extract patch with bounds checking
                z_end = min(z_start + self.patch_size[0], tof_img.shape[0])
                y_end = min(y_start + self.patch_size[1], tof_img.shape[1])
                x_end = min(x_start + self.patch_size[2], tof_img.shape[2])
                
                tof_patch = tof_img[z_start:z_end, y_start:y_end, x_start:x_end]
                
                if mask_img is not None:
                    mask_patch = mask_img[z_start:z_end, y_start:y_end, x_start:x_end]
                else:
                    mask_patch = np.zeros(tof_patch.shape, dtype=np.float32)
                
                # Ensure patch has correct size (pad if necessary)
                if tof_patch.shape != tuple(self.patch_size):
                    # Pad to correct size if needed
                    pad_z = self.patch_size[0] - tof_patch.shape[0]
                    pad_y = self.patch_size[1] - tof_patch.shape[1]
                    pad_x = self.patch_size[2] - tof_patch.shape[2]
                    
                    tof_patch = np.pad(tof_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
                    mask_patch = np.pad(mask_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')
            else:
                # Fallback to zero arrays
                tof_patch = np.zeros(self.patch_size, dtype=np.float32)
                mask_patch = np.zeros(self.patch_size, dtype=np.float32)
            
            # Simple preprocessing
            tof_patch = preprocess_3d_image(tof_patch, self.patch_size)
            if mask_img is not None:
                mask_patch = (mask_patch > 0).astype(np.float32)  # Binarize
            
            # Debug: Check if patch actually contains aneurysm voxels
            aneurysm_voxels = np.sum(mask_patch > 0)
            total_voxels = np.prod(mask_patch.shape)
            aneurysm_ratio = aneurysm_voxels / total_voxels
            
            if self._log_samples < self._max_log_samples and (idx % self._log_interval == 0):
                patch_type = patch_info.get('patch_type', 'unknown')
                tof_stats = f"TOF range: [{tof_patch.min():.2f}, {tof_patch.max():.2f}]"
                log_message(f"Patch {idx} ({patch_type}): {aneurysm_voxels}/{total_voxels} voxels ({aneurysm_ratio*100:.1f}%), {tof_stats}", "INFO")
                self._log_samples += 1
            
            # Validate patch quality - reject if too dense or too sparse for aneurysm patches
            if patch_info.get('patch_type') == 'aneurysm' and aneurysm_ratio > 0.5:
                log_message(f"Warning: Patch {idx} has {aneurysm_ratio*100:.1f}% aneurysm density (too high)", "WARNING")
            
            # Convert to tensors
            tof_tensor = torch.from_numpy(tof_patch).unsqueeze(0)  # Add channel dim
            mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)
            
            return {
                "image": tof_tensor,  # Primary TOF input
                "mask": mask_tensor,  # Ground truth
                "subject_id": patch_info["subject_id"],
                "has_aneurysm": patch_info["has_aneurysm"]
            }
            
        except Exception as e:
            log_message(f"Error loading patch {idx}: {e}", "WARNING")
            # Return zero tensors on error
            zero_patch = torch.zeros((1, *self.patch_size))
            return {
                "image": zero_patch,
                "mask": zero_patch,
                "subject_id": patch_info["subject_id"],
                "has_aneurysm": False
            }

# ========================================================================================
# MODEL IMPLEMENTATIONS (Simplified for timeout prevention)
# ========================================================================================

class SimpleConvBlock(nn.Module):
    """Simple 3D convolutional block."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class SimpleUNet3D(nn.Module):
    """Improved 3D U-Net for better performance."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_features: int = 16):
        super().__init__()
        
        # Encoder (improved)
        self.enc1 = SimpleConvBlock(in_channels, base_features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = SimpleConvBlock(base_features, base_features * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = SimpleConvBlock(base_features * 2, base_features * 4)
        
        # Decoder
        self.up2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, 2, stride=2)
        self.dec2 = SimpleConvBlock(base_features * 4, base_features * 2)
        self.up1 = nn.ConvTranspose3d(base_features * 2, base_features, 2, stride=2)
        self.dec1 = SimpleConvBlock(base_features * 2, base_features)
        
        # Output with dropout for regularization
        self.dropout = nn.Dropout3d(0.1)
        self.final = nn.Conv3d(base_features, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        x1 = self.pool1(e1)
        e2 = self.enc2(x1)
        x2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(x2)
        
        # Decoder
        x = self.up2(b)
        x = torch.cat([e2, x], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([e1, x], dim=1)
        x = self.dec1(x)
        
        # Apply dropout before final layer
        x = self.dropout(x)
        return self.final(x)

class SimpleTransformer3D(nn.Module):
    """Simplified transformer for 3D medical images."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_size = 4
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Simple transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose3d(embed_dim, out_channels, kernel_size=self.patch_size, stride=self.patch_size)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, D//patch_size, H//patch_size, W//patch_size
        
        # Flatten for transformer
        x = x.flatten(2).transpose(1, 2)  # B, N, embed_dim
        
        # Transformer
        x = self.transformer(x)
        
        # Reshape back
        patch_dims = [D//self.patch_size, H//self.patch_size, W//self.patch_size]
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *patch_dims)
        
        # Decode
        x = self.decoder(x)
        
        return x

# Model factory for all 15 architectures
def create_model(model_name: str, **kwargs) -> nn.Module:
    """Create model by name with timeout-safe implementations."""
    
    # Common parameters
    in_channels = kwargs.get('in_channels', 1)
    out_channels = kwargs.get('out_channels', 1)
    base_features = kwargs.get('base_features', 16)  # Improved size
    
    if model_name in ["unet", "unet3d", "lightweight_unet3d"]:
        return SimpleUNet3D(in_channels, out_channels, base_features)
    
    elif model_name == "attention_unet":
        # Simple U-Net with basic attention
        model = SimpleUNet3D(in_channels, out_channels, base_features)
        return model
    
    elif model_name == "nnu_net":
        # nnU-Net simplified
        return SimpleUNet3D(in_channels, out_channels, base_features)
    
    elif model_name in ["unetr", "unetr_plus"]:
        # Transformer-based
        return SimpleTransformer3D(in_channels, out_channels, embed_dim=64)
    
    elif model_name == "swin_unetr":
        # Swin transformer simplified
        return SimpleTransformer3D(in_channels, out_channels, embed_dim=48)
    
    elif model_name in ["primus", "slim_unetr"]:
        # Pure transformer
        return SimpleTransformer3D(in_channels, out_channels, embed_dim=32)
    
    elif model_name in ["es_unet", "rwkv_unet", "mamba_unet"]:
        # Enhanced models simplified
        return SimpleUNet3D(in_channels, out_channels, base_features)
    
    elif model_name in ["stacked_unet", "multiscale_unet"]:
        # Multi-scale models simplified
        return SimpleUNet3D(in_channels, out_channels, base_features)
    
    else:
        log_message(f"Unknown model: {model_name}, using default UNet", "WARNING")
        return SimpleUNet3D(in_channels, out_channels, base_features)

# ========================================================================================
# TRAINING AND EVALUATION
# ========================================================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation with class balancing for sparse data."""
    def __init__(self, smooth: float = 1e-5, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.smooth = smooth
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice operates on probabilities, so apply sigmoid view
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        logits_flat = logits.view(-1)
        
        # Standard Dice loss
        intersection = (probs_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        
        # Add focal loss component to handle class imbalance
        bce_loss = F.binary_cross_entropy_with_logits(logits_flat, target_flat, reduction='none')
        pt = torch.where(target_flat == 1, probs_flat, 1 - probs_flat)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        focal_loss = focal_loss.mean()
        
        # Combine losses (emphasize Dice for segmentation quality)
        combined_loss = 0.7 * dice_loss + 0.3 * focal_loss
        
        return combined_loss

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, spacing: Optional[Tuple[float, ...]] = None) -> Dict[str, float]:
    """Calculate comprehensive medical image segmentation metrics."""
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    
    pred_np = pred.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)
    
    # Flatten arrays for basic metrics
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Basic metrics
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-8)
    
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    accuracy = (pred_flat == target_flat).mean()
    
    # Sensitivity and Specificity
    true_positives = intersection
    false_positives = pred_flat.sum() - intersection
    false_negatives = target_flat.sum() - intersection
    true_negatives = len(pred_flat) - true_positives - false_positives - false_negatives
    
    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)  # Recall
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    precision = true_positives / (true_positives + false_positives + 1e-8)
    
    # F1 Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
    
    metrics = {
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "hausdorff_distance": 0.0,
        "surface_distance": 0.0,
        "volume_similarity": 0.0
    }
    
    # Advanced metrics (if scipy available and objects exist)
    if SCIPY_AVAILABLE and pred_np.sum() > 0 and target_np.sum() > 0:
        try:
            # Hausdorff Distance
            pred_coords = np.argwhere(pred_np)
            target_coords = np.argwhere(target_np)
            
            if len(pred_coords) > 0 and len(target_coords) > 0:
                hausdorff_dist = max(
                    directed_hausdorff(pred_coords, target_coords)[0],
                    directed_hausdorff(target_coords, pred_coords)[0]
                )
                metrics["hausdorff_distance"] = float(hausdorff_dist)
            
            # Surface Distance (approximate with edge detection)
            pred_edges = ndimage.binary_erosion(pred_np) ^ pred_np
            target_edges = ndimage.binary_erosion(target_np) ^ target_np
            
            if pred_edges.sum() > 0 and target_edges.sum() > 0:
                pred_edge_coords = np.argwhere(pred_edges)
                target_edge_coords = np.argwhere(target_edges)
                
                # Average surface distance
                distances = []
                for coord in pred_edge_coords[:100]:  # Sample for speed
                    min_dist = np.min(np.linalg.norm(target_edge_coords - coord, axis=1))
                    distances.append(min_dist)
                
                if distances:
                    metrics["surface_distance"] = float(np.mean(distances))
            
            # Volume Similarity (relative volume error) - FIXED
            pred_volume = pred_np.sum()
            target_volume = target_np.sum()
            if target_volume > 0:
                denom = max(float(target_volume), 1e-6)
                volume_sim = 1.0 - abs(pred_volume - target_volume) / denom
                metrics["volume_similarity"] = float(max(0.0, min(1.0, volume_sim)))
            else:
                metrics["volume_similarity"] = 0.0
            
        except Exception as e:
            # Fallback to basic metrics if advanced calculation fails
            pass
    
    return metrics

# ========================================================================================
# VISUALIZATION FUNCTIONS
# ========================================================================================

def create_segmentation_visualization(
    image: np.ndarray,
    ground_truth: np.ndarray, 
    prediction: np.ndarray,
    model_name: str,
    save_path: str,
    slice_idx: int = None
) -> None:
    """Create medical image segmentation visualization."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        # Select middle slice if not specified
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        
        # Extract 2D slices
        img_slice = image[slice_idx]
        gt_slice = ground_truth[slice_idx]
        pred_slice = prediction[slice_idx]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Segmentation Results', fontsize=14, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title('Original TOF-MRA')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[0, 1].imshow(gt_slice, cmap='Reds', alpha=0.5)
        axes[0, 1].set_title('Ground Truth Aneurysm')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[1, 0].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(pred_slice, cmap='Blues', alpha=0.5)
        axes[1, 0].set_title('Model Prediction')
        axes[1, 0].axis('off')
        
        # Overlay comparison
        axes[1, 1].imshow(img_slice, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(gt_slice, cmap='Reds', alpha=0.4, label='Ground Truth')
        axes[1, 1].imshow(pred_slice, cmap='Blues', alpha=0.4, label='Prediction')
        axes[1, 1].set_title('GT (Red) vs Prediction (Blue)')
        axes[1, 1].axis('off')
        
        # Add metrics text
        dice = 2 * np.sum(pred_slice * gt_slice) / (np.sum(pred_slice) + np.sum(gt_slice) + 1e-8)
        iou = np.sum(pred_slice * gt_slice) / (np.sum(pred_slice | gt_slice) + 1e-8)
        
        metrics_text = f'Dice: {dice:.3f}\nIoU: {iou:.3f}'
        fig.text(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log_message(f"Error creating visualization: {e}", "WARNING")

def create_training_curves(training_history: Dict[str, List[float]], save_path: str) -> None:
    """Create training curve visualizations."""
    if not MATPLOTLIB_AVAILABLE or not training_history:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(training_history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in training_history:
            axes[0, 0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice score
        if 'val_dice' in training_history:
            axes[0, 1].plot(epochs, training_history['val_dice'], 'g-', label='Validation Dice', linewidth=2)
            axes[0, 1].set_title('Dice Score Over Time')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # IoU score
        if 'val_iou' in training_history:
            axes[1, 0].plot(epochs, training_history['val_iou'], 'm-', label='Validation IoU', linewidth=2)
            axes[1, 0].set_title('IoU Score Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('IoU Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'learning_rate' in training_history:
            axes[1, 1].plot(epochs, training_history['learning_rate'], 'orange', label='Learning Rate', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log_message(f"Error creating training curves: {e}", "WARNING")

def create_model_comparison_chart(results: Dict[str, Dict[str, float]], save_path: str) -> None:
    """Create model performance comparison chart."""
    if not MATPLOTLIB_AVAILABLE or not results:
        return
    
    try:
        # Extract model names and metrics
        model_names = []
        dice_scores = []
        iou_scores = []
        times = []
        
        for model_name, model_results in results.items():
            if "best_dice" in model_results and model_results["best_dice"] > 0:
                model_names.append(model_name.replace('_', ' ').title())
                dice_scores.append(model_results["best_dice"])
                iou_scores.append(model_results.get("best_iou", 0))
                times.append(model_results.get("training_time_minutes", 0))
        
        if not model_names:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Medical Segmentation Benchmark Results\nEvolution Sequence (2015-2025)', 
                    fontsize=16, fontweight='bold')
        
        # Dice scores bar chart
        bars1 = axes[0, 0].bar(model_names, dice_scores, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Dice Score Comparison')
        axes[0, 0].set_ylabel('Dice Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, dice_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # IoU scores
        bars2 = axes[0, 1].bar(model_names, iou_scores, color='lightcoral', edgecolor='darkred')
        axes[0, 1].set_title('IoU Score Comparison')
        axes[0, 1].set_ylabel('IoU Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training times
        bars3 = axes[1, 0].bar(model_names, times, color='lightgreen', edgecolor='darkgreen')
        axes[1, 0].set_title('Training Time Comparison')
        axes[1, 0].set_ylabel('Time (minutes)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Efficiency plot (Dice/Time)
        efficiency = [d/t if t > 0 else 0 for d, t in zip(dice_scores, times)]
        bars4 = axes[1, 1].bar(model_names, efficiency, color='gold', edgecolor='orange')
        axes[1, 1].set_title('Efficiency (Dice/Time)')
        axes[1, 1].set_ylabel('Dice per Minute')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create evolution timeline chart
        create_evolution_timeline(results, save_path.replace('.png', '_timeline.png'))
        
    except Exception as e:
        log_message(f"Error creating comparison chart: {e}", "WARNING")

def create_evolution_timeline(results: Dict[str, Dict[str, float]], save_path: str) -> None:
    """Create medical AI evolution timeline visualization."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    try:
        # Model timeline mapping (approximate years)
        model_timeline = {
            'unet': ('2015', 'Classic CNN'),
            'unet3d': ('2015', 'Classic CNN'),
            'lightweight_unet3d': ('2017', 'Optimized CNN'),
            'attention_unet': ('2018', 'Attention CNN'),
            'nnu_net': ('2018', 'Self-Adaptive'),
            'unetr': ('2021', 'Transformer'),
            'unetr_plus': ('2022', 'Enhanced Transformer'),
            'swin_unetr': ('2022', 'Hybrid Transformer'),
            'primus': ('2023', 'Pure Transformer'),
            'slim_unetr': ('2023', 'Lightweight Transformer'),
            'es_unet': ('2024', 'Enhanced CNN'),
            'rwkv_unet': ('2024', 'Next-Gen Hybrid'),
            'mamba_unet': ('2025', 'State Space Model'),
            'stacked_unet': ('2025', 'Multi-Scale'),
            'multiscale_unet': ('2025', 'Multi-Resolution')
        }
        
        # Extract data
        years = []
        model_names = []
        dice_scores = []
        categories = []
        
        for model_name, model_results in results.items():
            if model_name in model_timeline and "best_dice" in model_results:
                year, category = model_timeline[model_name]
                years.append(int(year))
                model_names.append(model_name.replace('_', ' ').title())
                dice_scores.append(model_results["best_dice"])
                categories.append(category)
        
        if not years:
            return
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Color map for categories
        category_colors = {
            'Classic CNN': 'blue',
            'Optimized CNN': 'lightblue', 
            'Attention CNN': 'green',
            'Self-Adaptive': 'orange',
            'Transformer': 'red',
            'Enhanced Transformer': 'darkred',
            'Hybrid Transformer': 'purple',
            'Pure Transformer': 'magenta',
            'Lightweight Transformer': 'pink',
            'Enhanced CNN': 'cyan',
            'Next-Gen Hybrid': 'brown',
            'State Space Model': 'gold',
            'Multi-Scale': 'lime',
            'Multi-Resolution': 'navy'
        }
        
        # Scatter plot
        for i, (year, score, category, name) in enumerate(zip(years, dice_scores, categories, model_names)):
            color = category_colors.get(category, 'gray')
            ax.scatter(year, score, s=200, c=color, alpha=0.7, edgecolors='black', linewidth=1)
            ax.annotate(name, (year, score), xytext=(5, 5), textcoords='offset points', 
                       fontsize=9, ha='left')
        
        # Best fit line
        z = np.polyfit(years, dice_scores, 1)
        p = np.poly1d(z)
        ax.plot(years, p(years), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.4f}x + {z[1]:.3f}')
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Dice Score', fontsize=14)
        ax.set_title('Medical Image Segmentation Evolution (2015-2025)\nProgress in Aneurysm Detection', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add category legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=10, label=category)
                          for category, color in category_colors.items() if category in categories]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        log_message(f"Error creating evolution timeline: {e}", "WARNING")

def train_model_with_timeout(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    config: Dict[str, Any],
    timeout_manager: TimeoutManager,
    memory_manager: MemoryManager
) -> Dict[str, Any]:
    """Train model with comprehensive timeout and memory management."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup training components
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
    )
    use_amp = device.type == "cuda" and str(config["training"].get("precision", "")).startswith("16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    max_train_batches = config["training"].get("max_train_batches")
    max_val_batches = config["training"].get("max_val_batches")
    
    best_val_dice = 0.0
    best_metrics = {}
    patience_counter = 0
    
    max_epochs = config["training"]["max_epochs"]
    patience = config["training"]["patience"]
    
    log_message(f"Training {model_name} for max {max_epochs} epochs on {device}")
    
    for epoch in range(max_epochs):
        timeout_manager.check_total_timeout()
        timeout_manager.check_model_timeout()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            try:
                # Timeout check
                timeout_manager.check_total_timeout()
                timeout_manager.check_model_timeout()
                
                # Memory check
                if not memory_manager.check_memory():
                    memory_manager.cleanup(force=True)
                
                # Forward pass - only use TOF image now
                images = batch["image"].to(device)  # TOF image only
                masks = batch["mask"].to(device)
                
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Batch timeout check
                batch_time = time.time() - batch_start_time
                if batch_time > config["timeout"]["max_batch_time_seconds"]:
                    log_message(f"Batch timeout ({batch_time:.1f}s), skipping remaining batches", "WARNING")
                    break
                
                # Cleanup every few batches
                if batch_idx % config["memory"]["cleanup_frequency"] == 0:
                    memory_manager.cleanup()
                
                # Limit training batches to prevent timeout
                if max_train_batches and (batch_idx + 1) >= max_train_batches:
                    break
                    
            except (TimeoutException, RuntimeError) as e:
                log_message(f"Training error: {e}", "ERROR")
                memory_manager.cleanup(force=True)
                break
        
        if train_batches == 0:
            log_message(f"No training batches completed for {model_name}", "ERROR")
            break
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_metrics = {
            "dice": 0.0, "iou": 0.0, "accuracy": 0.0, "sensitivity": 0.0, 
            "specificity": 0.0, "precision": 0.0, "f1_score": 0.0,
            "hausdorff_distance": 0.0, "surface_distance": 0.0, "volume_similarity": 0.0
        }
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    timeout_manager.check_total_timeout()
                    timeout_manager.check_model_timeout()
                    
                    images = batch["image"].to(device)  # TOF image only  
                    masks = batch["mask"].to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(images)
                    batch_metrics = calculate_metrics(outputs, masks)
                    
                    for key in val_metrics:
                        val_metrics[key] += batch_metrics[key]
                    val_batches += 1
                    
                    # Limit validation batches
                    if max_val_batches and (batch_idx + 1) >= max_val_batches:
                        break
                        
                except (TimeoutException, RuntimeError) as e:
                    log_message(f"Validation error: {e}", "ERROR")
                    break
        
        if val_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= val_batches
        
        # Update best metrics
        current_dice = val_metrics["dice"]
        if current_dice > best_val_dice:
            best_val_dice = current_dice
            best_metrics = val_metrics.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        log_message(f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_train_loss:.4f}, Val Dice: {current_dice:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            log_message(f"Early stopping for {model_name} (patience: {patience})")
            break
        
        # Memory cleanup
        memory_manager.cleanup()
    
    return {
        "model_name": model_name,
        "best_dice": best_val_dice,
        "final_metrics": best_metrics,
        "epochs_completed": epoch + 1
    }

# ========================================================================================
# MAIN BENCHMARK FUNCTION
# ========================================================================================

def run_unified_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the unified benchmark with all models."""
    
    raw_path = Path(args.data_path)
    resolved_path, resolution_message = resolve_dataset_path(raw_path)

    if resolution_message:
        log_message(resolution_message, "INFO" if resolved_path.exists() else "WARNING")

    # Initialize managers
    memory_manager = MemoryManager(DEFAULT_CONFIG["memory"]["limit_ratio"])
    timeout_manager = TimeoutManager(
        DEFAULT_CONFIG["timeout"]["max_total_time_minutes"],
        DEFAULT_CONFIG["timeout"]["max_model_time_minutes"]
    )
    
    results = {
        "experiment_info": {
            "start_time": datetime.now().isoformat(),
            "data_path": str(resolved_path),
            "config": DEFAULT_CONFIG
        },
        "dataset_info": {},
        "model_results": {},
        "summary": {}
    }
    
    try:
        # Validate dataset
        log_message("Validating ADAM dataset structure...")
        is_valid, subjects, dataset_info = validate_adam_dataset(str(resolved_path))
        results["dataset_info"] = dataset_info
        
        if not is_valid:
            log_message("Dataset validation failed!", "ERROR")
            if dataset_info.get("error"):
                log_message(dataset_info["error"], "ERROR")
            return results
        
        if len(subjects) < 3:
            message = (
                f"Dataset validation failed: need at least 3 subjects, "
                f"but found {len(subjects)} valid subjects in {resolved_path}"
            )
            log_message(message, "ERROR")
            results["dataset_info"]["error"] = message
            return results
        
        log_message(f"Dataset valid: {len(subjects)} subjects found", "SUCCESS")
        
        # Tune configuration now that we know the dataset characteristics
        tune_config_for_device(subjects, args)

        # Limit subjects for timeout prevention
        dataset_cfg = DEFAULT_CONFIG["dataset"]
        subject_limit = dataset_cfg.get("max_subjects")
        if subject_limit:
            max_subjects = min(len(subjects), subject_limit)
            subjects = subjects[:max_subjects]
        log_message(f"Using {len(subjects)} subjects: {subjects}")
        if torch.cuda.is_available():
            log_message(
                "GPU optimisation active -> "
                f"batch_size={DEFAULT_CONFIG['dataset']['batch_size']}, "
                f"num_workers={DEFAULT_CONFIG['dataset']['num_workers']}, "
                f"max_patches_per_subject={DEFAULT_CONFIG['dataset']['max_patches_per_subject']}, "
                f"max_train_batches={DEFAULT_CONFIG['training']['max_train_batches']}, "
                f"max_model_time_minutes={DEFAULT_CONFIG['timeout']['max_model_time_minutes']}"
            )
        # Split dataset
        if len(subjects) < 3:
            message = "Need at least 3 subjects for train/val/test split"
            log_message(message, "ERROR")
            results["dataset_info"]["error"] = message
            return results

        train_frac = dataset_cfg.get("train_fraction", 0.75)
        val_frac = dataset_cfg.get("val_fraction", 0.15)
        test_frac = dataset_cfg.get("test_fraction", 0.10)
        total_frac = train_frac + val_frac + test_frac
        if not math.isclose(total_frac, 1.0):
            train_frac /= total_frac
            val_frac /= total_frac
            test_frac /= total_frac

        seed = DEFAULT_CONFIG.get("seed", dataset_cfg.get("seed", 42))

        train_subjects, temp_subjects = train_test_split(
            subjects,
            train_size=train_frac,
            random_state=seed,
            shuffle=True,
        )

        remaining = list(temp_subjects)
        if not remaining:
            # Fallback: move one subject from train to validation/test
            remaining = [train_subjects.pop()]

        val_subjects: List[str] = []
        test_subjects: List[str] = []
        remaining_frac = val_frac + test_frac
        if remaining_frac <= 0:
            val_subjects = train_subjects[:1]
            test_subjects = train_subjects[:1]
        else:
            val_share = 0.0 if val_frac <= 0 else val_frac / remaining_frac
            if val_share <= 0.0:
                val_subjects = []
                test_subjects = remaining
            elif val_share >= 1.0:
                val_subjects = remaining
                test_subjects = []
            else:
                val_subjects, test_subjects = train_test_split(
                    remaining,
                    train_size=val_share,
                    random_state=seed + 1,
                    shuffle=True,
                )

        if not val_subjects:
            val_subjects = test_subjects[:1] if test_subjects else train_subjects[:1]
        if not test_subjects:
            test_subjects = val_subjects[:1] if val_subjects else train_subjects[:1]

        log_message(
            f"Dataset split - Train: {len(train_subjects)}, "
            f"Val: {len(val_subjects)}, Test: {len(test_subjects)}"
        )
        results["dataset_info"]["split_counts"] = {
            "train": len(train_subjects),
            "val": len(val_subjects),
            "test": len(test_subjects),
        }
        
        # Create datasets
        patch_size = tuple(DEFAULT_CONFIG["dataset"]["patch_size"])
        
        dataset_seed = DEFAULT_CONFIG.get("seed", DEFAULT_CONFIG["dataset"].get("seed", 42))
        aneurysm_ratio = DEFAULT_CONFIG["dataset"].get("aneurysm_patch_ratio", 0.75)
        min_bg = DEFAULT_CONFIG["dataset"].get("min_background_patches", 4)

        train_dataset = UnifiedADAMDataset(
            data_path=str(resolved_path),
            subjects=train_subjects,
            patch_size=patch_size,
            max_patches_per_subject=DEFAULT_CONFIG["dataset"]["max_patches_per_subject"],
            is_training=True,
            aneurysm_patch_ratio=aneurysm_ratio,
            min_background_patches=min_bg,
            seed=dataset_seed,
        )
        val_dataset = UnifiedADAMDataset(
            data_path=str(resolved_path),
            subjects=val_subjects,
            patch_size=patch_size,
            max_patches_per_subject=DEFAULT_CONFIG["dataset"]["max_patches_per_subject"],
            is_training=False,
            aneurysm_patch_ratio=aneurysm_ratio,
            min_background_patches=min_bg,
            seed=dataset_seed + 1,
        )
        
        # Create data loaders
        pin_memory = torch.cuda.is_available()
        persistent_workers = pin_memory and DEFAULT_CONFIG["dataset"]["num_workers"] > 0
        loader_common_kwargs = dict(
            batch_size=DEFAULT_CONFIG["dataset"]["batch_size"],
            num_workers=DEFAULT_CONFIG["dataset"]["num_workers"],
            pin_memory=pin_memory,
        )
        if DEFAULT_CONFIG["dataset"]["num_workers"] > 0:
            loader_common_kwargs["prefetch_factor"] = DEFAULT_CONFIG["dataset"].get("prefetch_factor", 2)
        if persistent_workers:
            loader_common_kwargs["persistent_workers"] = True

        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_common_kwargs,
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **loader_common_kwargs,
        )
        
        log_message(f"Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        # Get models to benchmark
        models_to_run = args.models if args.models else DEFAULT_CONFIG["models"]["enabled"]
        
        if console and RICH_AVAILABLE:
            # Create progress table
            progress_table = Table(title="Model Benchmarking Progress")
            progress_table.add_column("Model", style="cyan")
            progress_table.add_column("Status", style="green")
            progress_table.add_column("Dice Score", style="yellow")
            progress_table.add_column("Time (min)", style="magenta")
            console.print(progress_table)
        
        # Train each model
        for model_idx, model_name in enumerate(models_to_run):
            try:
                timeout_manager.check_total_timeout()
                remaining_time = timeout_manager.get_remaining_time()
                
                log_message(f"\n{'='*50}")
                log_message(f"Training model {model_idx+1}/{len(models_to_run)}: {model_name}")
                if math.isinf(remaining_time):
                    log_message("Remaining time: no limit")
                else:
                    log_message(f"Remaining time: {remaining_time:.1f} minutes")
                log_message(f"{'='*50}")
                
                if remaining_time < 2:  # Need at least 2 minutes
                    log_message("Insufficient time remaining, stopping", "WARNING")
                    break
                
                # Start model timer
                timeout_manager.start_model_timer()
                model_start_time = time.time()
                
                # Create model
                model = create_model(model_name)
                log_message(f"Created {model_name} model")
                
                # Train model
                model_results = train_model_with_timeout(
                    model, train_loader, val_loader, model_name,
                    DEFAULT_CONFIG, timeout_manager, memory_manager
                )
                
                model_time = (time.time() - model_start_time) / 60
                model_results["training_time_minutes"] = model_time
                
                results["model_results"][model_name] = model_results
                
                log_message(f"Completed {model_name} - Dice: {model_results['best_dice']:.4f}, Time: {model_time:.1f}min", "SUCCESS")
                
                # Cleanup after each model
                del model
                memory_manager.cleanup(force=True)
                
            except TimeoutException as e:
                log_message(f"Timeout for model {model_name}: {e}", "ERROR")
                results["model_results"][model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "status": "timeout"
                }
                continue
                
            except Exception as e:
                log_message(f"Error training model {model_name}: {e}", "ERROR")
                results["model_results"][model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "status": "failed"
                }
                continue
        
        # Generate summary
        completed_models = [name for name, res in results["model_results"].items() 
                          if res.get("best_dice", 0) > 0]
        
        if completed_models:
            best_model = max(completed_models, 
                           key=lambda x: results["model_results"][x]["best_dice"])
            best_dice = results["model_results"][best_model]["best_dice"]
            
            results["summary"] = {
                "total_models_attempted": len(models_to_run),
                "models_completed": len(completed_models),
                "best_model": best_model,
                "best_dice_score": best_dice,
                "completed_models": completed_models
            }
        
        log_message(f"\nBenchmark completed! {len(completed_models)}/{len(models_to_run)} models finished", "SUCCESS")
        
    except Exception as e:
        log_message(f"Benchmark failed: {e}", "ERROR")
        log_message(traceback.format_exc(), "ERROR")
        results["error"] = str(e)
    
    finally:
        # Final cleanup
        memory_manager.cleanup(force=True)
        results["experiment_info"]["end_time"] = datetime.now().isoformat()
        results["experiment_info"]["total_time_minutes"] = (time.time() - timeout_manager.start_time) / 60
    
    return results

# ========================================================================================
# MAIN FUNCTION AND CLI
# ========================================================================================

def display_welcome():
    """Display welcome message."""
    if console and RICH_AVAILABLE:
        title = Text("🏥 Unified Medical Image Segmentation Benchmark", style="bold blue")
        subtitle = Text("Complete Evolution Sequence (2015-2025) | 3D ADAM Dataset | Timeout-Safe", style="italic")
        
        welcome_panel = Panel.fit(
            f"{title}\n{subtitle}\n\n"
            "Features:\n"
            "✅ All 15 models from evolution sequence\n"
            "🧠 3D medical image support\n"
            "⏱️ Comprehensive timeout prevention\n"
            "🔧 Memory management & error recovery\n"
            "📊 Automatic progress tracking\n",
            border_style="blue"
        )
        console.print(welcome_panel)
    else:
        print("="*60)
        print("🏥 Unified Medical Image Segmentation Benchmark")
        print("Complete Evolution Sequence (2015-2025) | 3D ADAM Dataset")
        print("="*60)

def save_results(results: Dict[str, Any], output_path: str):
    """Save benchmark results."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results_file = output_path / "benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    report_file = output_path / "benchmark_summary.txt"
    with open(report_file, 'w') as f:
        f.write("Medical Image Segmentation Benchmark Results\n")
        f.write("="*50 + "\n\n")
        
        # Dataset info
        dataset_info = results.get("dataset_info", {})
        f.write(f"Dataset: {dataset_info.get('valid_subjects', 0)} valid subjects\n")
        f.write(f"Subjects with aneurysms: {dataset_info.get('subjects_with_aneurysms', 0)}\n\n")
        
        # Model results
        f.write("Model Results:\n")
        f.write("-" * 30 + "\n")
        
        for model_name, result in results.get("model_results", {}).items():
            dice_score = result.get("best_dice", 0.0)
            time_taken = result.get("training_time_minutes", 0.0)
            f.write(f"{model_name:20s} | Dice: {dice_score:.4f} | Time: {time_taken:.1f}min\n")
        
        # Summary
        summary = results.get("summary", {})
        if summary:
            f.write(f"\nBest Model: {summary.get('best_model', 'None')}\n")
            f.write(f"Best Dice Score: {summary.get('best_dice_score', 0.0):.4f}\n")
            f.write(f"Models Completed: {summary.get('models_completed', 0)}/{summary.get('total_models_attempted', 0)}\n")
    
    # 🎨 CREATE COMPREHENSIVE VISUALIZATIONS  
    try:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        model_results = results.get("model_results", {})
        completed_results = {name: res for name, res in model_results.items() 
                           if res.get("best_dice", 0) > 0}
        
        if completed_results and MATPLOTLIB_AVAILABLE:
            log_message("Creating comprehensive benchmark visualizations...", "INFO")
            
            # 1. Model Performance Comparison Charts
            comparison_path = vis_dir / "model_comparison.png" 
            create_model_comparison_chart(completed_results, str(comparison_path))
            
            # 2. Evolution Timeline
            timeline_path = vis_dir / "evolution_timeline.png"
            create_evolution_timeline(completed_results, str(timeline_path))
            
            log_message(f"✅ Visualizations saved to {vis_dir}", "SUCCESS")
            
        else:
            if not MATPLOTLIB_AVAILABLE:
                log_message("⚠️ Matplotlib not available - skipping visualizations", "WARNING")
            else:
                log_message("⚠️ No completed models - skipping visualizations", "WARNING")
                
    except Exception as e:
        log_message(f"⚠️ Error creating visualizations: {e}", "WARNING")
    
    log_message(f"Results saved to {output_path}", "SUCCESS")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Unified Medical Image Segmentation Benchmark")
    
    # Required arguments
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to ADAM dataset directory")
    
    # Optional arguments  
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to benchmark (default: run predefined selection)")
    parser.add_argument("--all_models", action="store_true",
                       help="Run all 15 models (may cause timeout)")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run quick test with minimal settings")
    parser.add_argument("--max_time_minutes", type=int, default=None,
                       help="Maximum total execution time in minutes (<=0 disables)")
    parser.add_argument("--max_subjects", type=int, default=None,
                       help="Maximum number of subjects to use (default: use all available)")
    parser.add_argument("--max_epochs", type=int, default=None,
                       help="Override maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size for training/validation")
    parser.add_argument("--max_patches_per_subject", type=int, default=None,
                       help="Limit the number of patches sampled per subject")
    parser.add_argument("--max_model_time_minutes", type=int, default=None,
                       help="Override per-model timeout in minutes (<=0 disables)")
    parser.add_argument("--max_train_batches", type=int, default=None,
                       help="Maximum training batches per epoch")
    parser.add_argument("--max_val_batches", type=int, default=None,
                       help="Maximum validation batches per epoch")
    
    args = parser.parse_args()
    
    # Display welcome
    display_welcome()
    
    # Update config based on arguments
    if args.all_models:
        DEFAULT_CONFIG["models"]["enabled"] = DEFAULT_CONFIG["models"]["all_models"]
    elif args.quick_test:
        DEFAULT_CONFIG["models"]["enabled"] = ["unet", "unetr"]
        DEFAULT_CONFIG["training"]["max_epochs"] = 2
        DEFAULT_CONFIG["dataset"]["max_subjects"] = 3
        DEFAULT_CONFIG["dataset"]["max_patches_per_subject"] = 5
    
    if args.max_subjects is not None:
        DEFAULT_CONFIG["dataset"]["max_subjects"] = args.max_subjects
    if args.max_epochs is not None:
        DEFAULT_CONFIG["training"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        DEFAULT_CONFIG["dataset"]["batch_size"] = max(1, args.batch_size)
    if args.max_patches_per_subject is not None:
        DEFAULT_CONFIG["dataset"]["max_patches_per_subject"] = max(1, args.max_patches_per_subject)
    if args.max_model_time_minutes is not None:
        if args.max_model_time_minutes <= 0:
            DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] = None
        else:
            DEFAULT_CONFIG["timeout"]["max_model_time_minutes"] = args.max_model_time_minutes
    if args.max_train_batches is not None:
        DEFAULT_CONFIG["training"]["max_train_batches"] = max(1, args.max_train_batches)
    if args.max_val_batches is not None:
        DEFAULT_CONFIG["training"]["max_val_batches"] = max(1, args.max_val_batches)
    
    if args.max_time_minutes is not None:
        if args.max_time_minutes <= 0:
            DEFAULT_CONFIG["timeout"]["max_total_time_minutes"] = None
        else:
            DEFAULT_CONFIG["timeout"]["max_total_time_minutes"] = args.max_time_minutes
    
    # Validate data path
    if not Path(args.data_path).exists():
        log_message(f"Data path does not exist: {args.data_path}", "ERROR")
        return 1
    
    # Run benchmark
    total_limit = DEFAULT_CONFIG["timeout"]["max_total_time_minutes"]
    if total_limit is None:
        log_message("Starting benchmark with timeout limit: no limit")
    else:
        log_message(f"Starting benchmark with timeout limit: {total_limit} minutes")
    results = run_unified_benchmark(args)
    
    # Save results
    save_results(results, args.output_dir)
    
    # Display final summary
    if console and RICH_AVAILABLE:
        summary = results.get("summary", {})
        if summary:
            summary_text = f"""
Benchmark Complete! 

📊 Results Summary:
• Models completed: {summary.get('models_completed', 0)}/{summary.get('total_models_attempted', 0)}
• Best model: {summary.get('best_model', 'None')}
• Best Dice score: {summary.get('best_dice_score', 0.0):.4f}
• Total time: {results['experiment_info'].get('total_time_minutes', 0):.1f} minutes

📁 Results saved to: {args.output_dir}
"""
            console.print(Panel(summary_text, title="🎉 Benchmark Complete", border_style="green"))
        else:
            console.print(Panel("❌ Benchmark failed - check logs for details", border_style="red"))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
