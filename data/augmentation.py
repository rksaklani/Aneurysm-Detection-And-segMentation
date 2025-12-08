"""
Medical Image Augmentation Pipeline

This module provides comprehensive data augmentation utilities for medical images,
including geometric transformations, intensity augmentations, and elastic deformations.
"""

import logging
from typing import Tuple, Optional, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import map_coordinates, gaussian_filter, rotate
import random

logger = logging.getLogger(__name__)


class MedicalImageAugmentation:
    """
    Comprehensive augmentation pipeline for medical images.
    
    This class handles:
    - Geometric transformations (rotation, translation, scaling)
    - Intensity augmentations (brightness, contrast, noise)
    - Elastic deformations
    - Multi-modal consistency
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        translation_range: float = 0.1,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        noise_std: float = 0.01,
        elastic_alpha: float = 1000.0,
        elastic_sigma: float = 30.0,
        elastic_probability: float = 0.5,
        flip_probability: float = 0.5,
        apply_probability: float = 0.8,
        enabled: bool = True
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            translation_range: Maximum translation as fraction of image size
            scale_range: Range for scaling (min, max)
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            elastic_probability: Probability of applying elastic deformation
            flip_probability: Probability of applying random flips
            apply_probability: Overall probability of applying augmentations
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_probability = elastic_probability
        self.flip_probability = flip_probability
        self.apply_probability = apply_probability
        self.enabled = enabled
    
    def __call__(
        self, 
        tof_image: torch.Tensor, 
        struct_image: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to multi-modal images.
        
        Args:
            tof_image: TOF-MRA image tensor
            struct_image: Structural image tensor
            mask: Ground truth mask tensor
            
        Returns:
            Tuple of augmented (tof_image, struct_image, mask)
        """
        if random.random() > self.apply_probability:
            return tof_image, struct_image, mask
        
        # Convert to numpy for augmentation
        tof_np = tof_image.squeeze().numpy()
        struct_np = struct_image.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        
        # Apply geometric augmentations
        tof_np, struct_np, mask_np = self._geometric_augmentation(
            tof_np, struct_np, mask_np
        )
        
        # Apply intensity augmentations
        tof_np = self._intensity_augmentation(tof_np)
        struct_np = self._intensity_augmentation(struct_np)
        
        # Convert back to tensors
        tof_tensor = torch.from_numpy(tof_np.copy()).float().unsqueeze(0)
        struct_tensor = torch.from_numpy(struct_np.copy()).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np.copy()).float().unsqueeze(0)
        
        return tof_tensor, struct_tensor, mask_tensor
    
    def _geometric_augmentation(
        self, 
        tof_image: np.ndarray, 
        struct_image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply geometric augmentations."""
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            tof_image = self._rotate_image(tof_image, angle)
            struct_image = self._rotate_image(struct_image, angle)
            mask = self._rotate_image(mask, angle)
        
        # Random translation
        if random.random() < 0.5:
            translation = self._get_random_translation(tof_image.shape)
            tof_image = self._translate_image(tof_image, translation)
            struct_image = self._translate_image(struct_image, translation)
            mask = self._translate_image(mask, translation)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            tof_image = self._scale_image(tof_image, scale)
            struct_image = self._scale_image(struct_image, scale)
            mask = self._scale_image(mask, scale)
        
        # Random flips
        if random.random() < self.flip_probability:
            axes = random.choice([0, 1, 2])
            tof_image = np.flip(tof_image, axis=axes)
            struct_image = np.flip(struct_image, axis=axes)
            mask = np.flip(mask, axis=axes)
        
        # Elastic deformation
        if random.random() < self.elastic_probability:
            tof_image, struct_image, mask = self._elastic_deformation(
                tof_image, struct_image, mask
            )
        
        return tof_image, struct_image, mask
    
    def _intensity_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply intensity augmentations."""
        # Brightness adjustment
        if random.random() < 0.5:
            brightness_factor = random.uniform(
                1 - self.brightness_range, 
                1 + self.brightness_range
            )
            image = image * brightness_factor
        
        # Contrast adjustment
        if random.random() < 0.5:
            contrast_factor = random.uniform(
                1 - self.contrast_range, 
                1 + self.contrast_range
            )
            mean = np.mean(image)
            image = (image - mean) * contrast_factor + mean
        
        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, image.shape)
            image = image + noise
        
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        try:
            # Rotate around z-axis (most common for medical images)
            rotated = rotate(image, angle, axes=(1, 2), reshape=False, order=1)
            return rotated
        except Exception as e:
            logger.warning(f"Rotation failed: {e}")
            return image
    
    def _get_random_translation(self, shape: Tuple[int, ...]) -> Tuple[float, float, float]:
        """Get random translation values."""
        max_translation = [
            shape[i] * self.translation_range for i in range(len(shape))
        ]
        translation = [
            random.uniform(-max_translation[i], max_translation[i])
            for i in range(len(shape))
        ]
        return tuple(translation)
    
    def _translate_image(self, image: np.ndarray, translation: Tuple[float, ...]) -> np.ndarray:
        """Translate image by given values."""
        try:
            # Create translation matrix
            translation_matrix = np.eye(len(translation) + 1)
            for i, t in enumerate(translation):
                translation_matrix[i, -1] = t
            
            # Apply translation
            translated = ndimage.affine_transform(
                image, 
                translation_matrix[:-1, :-1], 
                offset=translation_matrix[:-1, -1],
                order=1
            )
            return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return image
    
    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor."""
        try:
            # Calculate zoom factors
            zoom_factors = [scale] * len(image.shape)
            scaled = ndimage.zoom(image, zoom_factors, order=1)
            
            # Crop or pad to original size
            if scale > 1:
                # Crop from center
                start = [(scaled.shape[i] - image.shape[i]) // 2 
                        for i in range(len(image.shape))]
                end = [start[i] + image.shape[i] for i in range(len(image.shape))]
                scaled = scaled[tuple(slice(start[i], end[i]) for i in range(len(image.shape)))]
            else:
                # Pad with zeros
                pad_width = [(image.shape[i] - scaled.shape[i]) // 2 
                           for i in range(len(image.shape))]
                pad_width = [(pad_width[i], image.shape[i] - scaled.shape[i] - pad_width[i])
                           for i in range(len(image.shape))]
                scaled = np.pad(scaled, pad_width, mode='constant')
            
            return scaled
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
            return image
    
    def _elastic_deformation(
        self, 
        tof_image: np.ndarray, 
        struct_image: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply elastic deformation to images."""
        try:
            # Generate random displacement field
            shape = tof_image.shape
            displacement = np.random.randn(*shape, len(shape)) * self.elastic_alpha
            
            # Smooth the displacement field
            for i in range(len(shape)):
                displacement[..., i] = gaussian_filter(
                    displacement[..., i], 
                    self.elastic_sigma
                )
            
            # Create coordinate grids
            coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
            coords = np.stack(coords, axis=-1)
            
            # Apply displacement
            new_coords = coords + displacement
            
            # Deform images
            deformed_tof = map_coordinates(tof_image, new_coords.T, order=1)
            deformed_struct = map_coordinates(struct_image, new_coords.T, order=1)
            deformed_mask = map_coordinates(mask, new_coords.T, order=0)  # Nearest neighbor for masks
            
            return deformed_tof, deformed_struct, deformed_mask
            
        except Exception as e:
            logger.warning(f"Elastic deformation failed: {e}")
            return tof_image, struct_image, mask


class AdvancedAugmentation:
    """
    Advanced augmentation techniques for medical images.
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.5
    ):
        """
        Initialize advanced augmentation.
        
        Args:
            mixup_alpha: Alpha parameter for mixup
            cutmix_alpha: Alpha parameter for cutmix
            cutmix_prob: Probability of applying cutmix
            mixup_prob: Probability of applying mixup
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.mixup_prob = mixup_prob
    
    def mixup(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.
        
        Args:
            image1: First image
            image2: Second image
            mask1: First mask
            mask2: Second mask
            
        Returns:
            Tuple of (mixed_image, mixed_mask, lambda)
        """
        if random.random() > self.mixup_prob:
            return image1, mask1, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Mix images and masks
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_mask = lam * mask1 + (1 - lam) * mask2
        
        return mixed_image, mixed_mask, lam
    
    def cutmix(
        self, 
        image1: torch.Tensor, 
        image2: torch.Tensor,
        mask1: torch.Tensor,
        mask2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply cutmix augmentation.
        
        Args:
            image1: First image
            image2: Second image
            mask1: First mask
            mask2: Second mask
            
        Returns:
            Tuple of (mixed_image, mixed_mask, lambda)
        """
        if random.random() > self.cutmix_prob:
            return image1, mask1, 1.0
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Get image dimensions
        H, W, D = image1.shape[-3:]
        
        # Calculate cut dimensions
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cut_d = int(D * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)
        
        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)
        
        # Apply cutmix
        mixed_image = image1.clone()
        mixed_mask = mask1.clone()
        mixed_image[..., bby1:bby2, bbx1:bbx2, bbz1:bbz2] = image2[..., bby1:bby2, bbx1:bbx2, bbz1:bbz2]
        mixed_mask[..., bby1:bby2, bbx1:bbx2, bbz1:bbz2] = mask2[..., bby1:bby2, bbx1:bbx2, bbz1:bbz2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (W * H * D))
        
        return mixed_image, mixed_mask, lam


def create_augmentation(config: Dict[str, Any]) -> MedicalImageAugmentation:
    """
    Create augmentation pipeline from configuration.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Configured augmentation pipeline
    """
    return MedicalImageAugmentation(
        rotation_range=config.get("rotation_range", 15.0),
        translation_range=config.get("translation_range", 0.1),
        scale_range=config.get("scale_range", (0.9, 1.1)),
        brightness_range=config.get("brightness_range", 0.2),
        contrast_range=config.get("contrast_range", 0.2),
        noise_std=config.get("noise_std", 0.01),
        elastic_alpha=config.get("elastic_alpha", 1000.0),
        elastic_sigma=config.get("elastic_sigma", 30.0),
        elastic_probability=config.get("elastic_probability", 0.5),
        flip_probability=config.get("flip_probability", 0.5),
        apply_probability=config.get("apply_probability", 0.8)
    )
