"""
Medical Image Preprocessing Pipeline

This module provides comprehensive preprocessing utilities for medical images,
including normalization, resampling, and intensity correction.
"""

import logging
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom
from skimage import exposure, filters
from skimage.transform import resize

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """
    Comprehensive preprocessing pipeline for medical images.
    
    This class handles:
    - Intensity normalization
    - Spatial resampling
    - Bias field correction
    - Noise reduction
    - Contrast enhancement
    """
    
    def __init__(
        self,
        target_spacing: Optional[Tuple[float, float, float]] = None,
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize_method: str = "z_score",
        clip_percentiles: Tuple[float, float] = (0.5, 99.5),
        bias_correction: bool = True,
        noise_reduction: bool = True,
        contrast_enhancement: bool = False
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_spacing: Target voxel spacing for resampling
            target_size: Target image size (if None, calculated from spacing)
            normalize_method: Normalization method ('z_score', 'min_max', 'robust')
            clip_percentiles: Percentiles for clipping outliers
            bias_correction: Whether to apply bias field correction
            noise_reduction: Whether to apply noise reduction
            contrast_enhancement: Whether to apply contrast enhancement
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.clip_percentiles = clip_percentiles
        self.bias_correction = bias_correction
        self.noise_reduction = noise_reduction
        self.contrast_enhancement = contrast_enhancement
        
        # Initialize preprocessing steps
        self._setup_preprocessing_steps()
    
    def _setup_preprocessing_steps(self):
        """Setup preprocessing steps based on configuration."""
        self.steps = []
        
        if self.bias_correction:
            self.steps.append(self._bias_field_correction)
        
        if self.noise_reduction:
            self.steps.append(self._noise_reduction)
        
        if self.contrast_enhancement:
            self.steps.append(self._contrast_enhancement)
        
        self.steps.append(self._intensity_normalization)
    
    def __call__(self, image: np.ndarray, is_mask: bool = False) -> np.ndarray:
        """
        Apply preprocessing pipeline to image.
        
        Args:
            image: Input image array
            is_mask: Whether this is a mask (skip intensity processing)
            
        Returns:
            Preprocessed image
        """
        if is_mask:
            # For masks, only apply spatial preprocessing
            return self._spatial_preprocessing(image)
        else:
            # For images, apply full preprocessing pipeline
            processed_image = image.copy()
            
            for step in self.steps:
                processed_image = step(processed_image)
            
            return self._spatial_preprocessing(processed_image)
    
    def _bias_field_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply bias field correction using N4ITK-like approach."""
        try:
            # Simple bias field correction using Gaussian filtering
            # In practice, you might want to use N4ITK or similar
            bias_field = ndimage.gaussian_filter(image, sigma=50)
            bias_field = bias_field / np.mean(bias_field)
            corrected_image = image / (bias_field + 1e-8)
            return corrected_image
        except Exception as e:
            logger.warning(f"Bias field correction failed: {e}")
            return image
    
    def _noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using Gaussian filtering."""
        try:
            # Apply Gaussian filter for noise reduction
            sigma = 0.5  # Adjust based on noise level
            filtered_image = ndimage.gaussian_filter(image, sigma=sigma)
            return filtered_image
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return image
    
    def _contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement using CLAHE."""
        try:
            # Apply Contrast Limited Adaptive Histogram Equalization
            enhanced_image = exposure.equalize_adapthist(
                image, 
                clip_limit=0.03,
                nbins=256
            )
            return enhanced_image
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def _intensity_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply intensity normalization."""
        try:
            # Clip outliers
            p_low, p_high = np.percentile(image, self.clip_percentiles)
            clipped_image = np.clip(image, p_low, p_high)
            
            if self.normalize_method == "z_score":
                # Z-score normalization
                mean = np.mean(clipped_image)
                std = np.std(clipped_image)
                normalized_image = (clipped_image - mean) / (std + 1e-8)
                
            elif self.normalize_method == "min_max":
                # Min-max normalization
                min_val = np.min(clipped_image)
                max_val = np.max(clipped_image)
                normalized_image = (clipped_image - min_val) / (max_val - min_val + 1e-8)
                
            elif self.normalize_method == "robust":
                # Robust normalization using median and IQR
                median = np.median(clipped_image)
                q75, q25 = np.percentile(clipped_image, [75, 25])
                iqr = q75 - q25
                normalized_image = (clipped_image - median) / (iqr + 1e-8)
                
            else:
                raise ValueError(f"Unknown normalization method: {self.normalize_method}")
            
            return normalized_image
            
        except Exception as e:
            logger.warning(f"Intensity normalization failed: {e}")
            return image
    
    def _spatial_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply spatial preprocessing (resampling, resizing)."""
        try:
            if self.target_spacing is not None:
                # Resample to target spacing
                current_spacing = np.array([1.0, 1.0, 1.0])  # Assume isotropic
                zoom_factors = current_spacing / np.array(self.target_spacing)
                resampled_image = zoom(image, zoom_factors, order=1)
                return resampled_image
            
            elif self.target_size is not None:
                # Resize to target size
                resized_image = resize(
                    image, 
                    self.target_size, 
                    order=1, 
                    preserve_range=True
                )
                return resized_image
            
            else:
                return image
                
        except Exception as e:
            logger.warning(f"Spatial preprocessing failed: {e}")
            return image


class MultiModalPreprocessor:
    """
    Preprocessor for multi-modal medical images (TOF-MRA + Structural).
    """
    
    def __init__(
        self,
        tof_preprocessor: MedicalImagePreprocessor,
        struct_preprocessor: MedicalImagePreprocessor
    ):
        """
        Initialize multi-modal preprocessor.
        
        Args:
            tof_preprocessor: Preprocessor for TOF-MRA images
            struct_preprocessor: Preprocessor for structural images
        """
        self.tof_preprocessor = tof_preprocessor
        self.struct_preprocessor = struct_preprocessor
    
    def __call__(
        self, 
        tof_image: np.ndarray, 
        struct_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess multi-modal images.
        
        Args:
            tof_image: TOF-MRA image
            struct_image: Structural image
            mask: Optional mask
            
        Returns:
            Tuple of (preprocessed_tof, preprocessed_struct, preprocessed_mask)
        """
        # Preprocess TOF-MRA
        processed_tof = self.tof_preprocessor(tof_image)
        
        # Preprocess structural image
        processed_struct = self.struct_preprocessor(struct_image)
        
        # Preprocess mask if provided
        processed_mask = None
        if mask is not None:
            processed_mask = self.tof_preprocessor(mask, is_mask=True)
        
        return processed_tof, processed_struct, processed_mask


def create_preprocessor(config: Dict[str, Any]) -> MedicalImagePreprocessor:
    """
    Create preprocessor from configuration.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        Configured preprocessor
    """
    return MedicalImagePreprocessor(
        target_spacing=config.get("target_spacing"),
        target_size=config.get("target_size"),
        normalize_method=config.get("normalize_method", "z_score"),
        clip_percentiles=config.get("clip_percentiles", (0.5, 99.5)),
        bias_correction=config.get("bias_correction", True),
        noise_reduction=config.get("noise_reduction", True),
        contrast_enhancement=config.get("contrast_enhancement", False)
    )
