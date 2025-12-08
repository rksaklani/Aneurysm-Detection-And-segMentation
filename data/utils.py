"""
Medical Image Utilities

This module provides utility functions for medical image processing,
including I/O operations, statistics, and visualization.
"""

import logging
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

logger = logging.getLogger(__name__)


def load_nifti(file_path: str) -> np.ndarray:
    """
    Load NIfTI file and return image data.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Image data as numpy array
    """
    try:
        nii_img = nib.load(file_path)
        return nii_img.get_fdata()
    except Exception as e:
        logger.error(f"Failed to load NIfTI file {file_path}: {e}")
        raise


def save_nifti(image: np.ndarray, file_path: str, reference: Optional[nib.Nifti1Image] = None):
    """
    Save image data as NIfTI file.
    
    Args:
        image: Image data to save
        file_path: Output file path
        reference: Reference NIfTI image for header information
    """
    try:
        if reference is not None:
            nii_img = nib.Nifti1Image(image, reference.affine, reference.header)
        else:
            nii_img = nib.Nifti1Image(image, np.eye(4))
        
        nib.save(nii_img, file_path)
        logger.info(f"Saved NIfTI file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save NIfTI file {file_path}: {e}")
        raise


def get_image_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Get comprehensive statistics for medical image.
    
    Args:
        image: Input image array
        
    Returns:
        Dictionary containing image statistics
    """
    stats = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "median": float(np.median(image)),
        "q25": float(np.percentile(image, 25)),
        "q75": float(np.percentile(image, 75)),
        "non_zero_voxels": int(np.count_nonzero(image)),
        "total_voxels": int(image.size),
        "sparsity": float(1 - np.count_nonzero(image) / image.size)
    }
    
    return stats


def visualize_medical_image(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Medical Image",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Visualize medical image with optional mask overlay.
    
    Args:
        image: Input image
        mask: Optional mask for overlay
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get middle slices
    mid_z = image.shape[0] // 2
    mid_y = image.shape[1] // 2
    mid_x = image.shape[2] // 2
    
    # Axial view
    axes[0].imshow(image[mid_z, :, :], cmap='gray')
    if mask is not None:
        axes[0].imshow(mask[mid_z, :, :], alpha=0.5, cmap='Reds')
    axes[0].set_title('Axial View')
    axes[0].axis('off')
    
    # Coronal view
    axes[1].imshow(image[:, mid_y, :], cmap='gray')
    if mask is not None:
        axes[1].imshow(mask[:, mid_y, :], alpha=0.5, cmap='Reds')
    axes[1].set_title('Coronal View')
    axes[1].axis('off')
    
    # Sagittal view
    axes[2].imshow(image[:, :, mid_x], cmap='gray')
    if mask is not None:
        axes[2].imshow(mask[:, :, mid_x], alpha=0.5, cmap='Reds')
    axes[2].set_title('Sagittal View')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization: {save_path}")
    
    plt.show()


def create_histogram(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "Intensity Histogram",
    bins: int = 100,
    save_path: Optional[str] = None
) -> None:
    """
    Create intensity histogram for medical image.
    
    Args:
        image: Input image
        mask: Optional mask to filter histogram
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    if mask is not None:
        # Histogram for masked region
        masked_image = image[mask > 0]
        plt.hist(masked_image, bins=bins, alpha=0.7, label='Masked Region', color='red')
        
        # Histogram for background
        background_mask = mask == 0
        background_image = image[background_mask]
        plt.hist(background_image, bins=bins, alpha=0.7, label='Background', color='blue')
        
        plt.legend()
    else:
        plt.hist(image.flatten(), bins=bins, alpha=0.7)
    
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved histogram: {save_path}")
    
    plt.show()


def compare_images(
    image1: np.ndarray,
    image2: np.ndarray,
    title1: str = "Image 1",
    title2: str = "Image 2",
    save_path: Optional[str] = None
) -> None:
    """
    Compare two medical images side by side.
    
    Args:
        image1: First image
        image2: Second image
        title1: Title for first image
        title2: Title for second image
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get middle slices
    mid_z = image1.shape[0] // 2
    mid_y = image1.shape[1] // 2
    mid_x = image1.shape[2] // 2
    
    # First image
    axes[0, 0].imshow(image1[mid_z, :, :], cmap='gray')
    axes[0, 0].set_title(f'{title1} - Axial')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image1[:, mid_y, :], cmap='gray')
    axes[0, 1].set_title(f'{title1} - Coronal')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(image1[:, :, mid_x], cmap='gray')
    axes[0, 2].set_title(f'{title1} - Sagittal')
    axes[0, 2].axis('off')
    
    # Second image
    axes[1, 0].imshow(image2[mid_z, :, :], cmap='gray')
    axes[1, 0].set_title(f'{title2} - Axial')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image2[:, mid_y, :], cmap='gray')
    axes[1, 1].set_title(f'{title2} - Coronal')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(image2[:, :, mid_x], cmap='gray')
    axes[1, 2].set_title(f'{title2} - Sagittal')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison: {save_path}")
    
    plt.show()


def create_dataset_summary(
    data_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive summary of dataset.
    
    Args:
        data_path: Path to dataset
        output_path: Optional path to save summary
        
    Returns:
        Dataset summary dictionary
    """
    data_path = Path(data_path)
    summary = {
        "total_subjects": 0,
        "subjects_with_aneurysms": 0,
        "subjects_without_aneurysms": 0,
        "total_aneurysms": 0,
        "image_statistics": {},
        "file_structure": {}
    }
    
    subjects = []
    for subject_dir in data_path.iterdir():
        if subject_dir.is_dir():
            subjects.append(subject_dir.name)
    
    summary["total_subjects"] = len(subjects)
    
    # Analyze each subject
    for subject_id in subjects:
        subject_path = data_path / subject_id
        
        # Check for files
        tof_files = list(subject_path.glob("*TOF*.nii*"))
        struct_files = list(subject_path.glob("*struct*.nii*"))
        mask_files = list(subject_path.glob("*aneurysm*.nii*"))
        
        if mask_files:
            summary["subjects_with_aneurysms"] += 1
            
            # Count aneurysms in mask
            try:
                mask = load_nifti(str(mask_files[0]))
                num_aneurysms = len(np.unique(mask)) - 1  # Subtract background
                summary["total_aneurysms"] += num_aneurysms
            except Exception as e:
                logger.warning(f"Failed to analyze mask for subject {subject_id}: {e}")
        else:
            summary["subjects_without_aneurysms"] += 1
    
    # Save summary
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved dataset summary: {output_path}")
    
    return summary


def validate_dataset(data_path: str) -> Dict[str, Any]:
    """
    Validate dataset structure and integrity.
    
    Args:
        data_path: Path to dataset
        
    Returns:
        Validation results
    """
    data_path = Path(data_path)
    validation_results = {
        "valid_subjects": [],
        "invalid_subjects": [],
        "missing_files": [],
        "corrupted_files": []
    }
    
    for subject_dir in data_path.iterdir():
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        is_valid = True
        missing_files = []
        
        # Check for required files
        tof_files = list(subject_dir.glob("*TOF*.nii*"))
        struct_files = list(subject_dir.glob("*struct*.nii*"))
        
        if not tof_files:
            missing_files.append("TOF-MRA")
            is_valid = False
        
        if not struct_files:
            missing_files.append("Structural")
            is_valid = False
        
        # Check file integrity
        for file_path in tof_files + struct_files:
            try:
                load_nifti(str(file_path))
            except Exception as e:
                validation_results["corrupted_files"].append({
                    "subject": subject_id,
                    "file": str(file_path),
                    "error": str(e)
                })
                is_valid = False
        
        if is_valid:
            validation_results["valid_subjects"].append(subject_id)
        else:
            validation_results["invalid_subjects"].append(subject_id)
            validation_results["missing_files"].append({
                "subject": subject_id,
                "missing": missing_files
            })
    
    return validation_results
