"""
Loss Functions for Medical Image Segmentation

This module provides comprehensive loss functions specifically designed
for medical image segmentation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss for medical image segmentation.
    
    The Dice loss is particularly effective for imbalanced datasets
    and small object segmentation.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        reduction: str = "mean",
        ignore_index: Optional[int] = None
    ):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Dice loss
        """
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Calculate Dice loss
        dice_loss = 1.0 - dice
        
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class DiceCELoss(nn.Module):
    """
    Combined Dice and Cross-Entropy Loss.
    
    This loss combines the benefits of Dice loss (good for imbalanced data)
    and Cross-Entropy loss (good for overall optimization).
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        reduction: str = "mean"
    ):
        """
        Initialize Dice-CE Loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            ce_weight: Weight for Cross-Entropy loss component
            smooth: Smoothing factor for Dice loss
            reduction: Reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth, reduction=reduction)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Combined loss
        """
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Focal loss
        """
        # Apply sigmoid
        inputs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for medical image segmentation.
    
    Tversky loss allows for different weights for false positives and false negatives.
    """
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        reduction: str = "mean"
    ):
        """
        Initialize Tversky Loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Tversky loss
        """
        # Apply sigmoid
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate true positives, false positives, false negatives
        tp = (inputs * targets).sum(dim=1)
        fp = (inputs * (1 - targets)).sum(dim=1)
        fn = ((1 - inputs) * targets).sum(dim=1)
        
        # Calculate Tversky coefficient
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Calculate Tversky loss
        tversky_loss = 1.0 - tversky
        
        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for medical image segmentation.
    
    This loss focuses on the boundary regions which are often the most
    challenging to segment accurately.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize Boundary Loss.
        
        Args:
            reduction: Reduction method
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Boundary loss
        """
        # Apply sigmoid
        inputs = torch.sigmoid(inputs)
        
        # Calculate boundary maps
        boundary_targets = self._get_boundary_map(targets)
        boundary_inputs = self._get_boundary_map(inputs)
        
        # Calculate boundary loss
        boundary_loss = F.binary_cross_entropy(boundary_inputs, boundary_targets, reduction='none')
        
        if self.reduction == "mean":
            return boundary_loss.mean()
        elif self.reduction == "sum":
            return boundary_loss.sum()
        else:
            return boundary_loss
    
    def _get_boundary_map(self, mask: torch.Tensor) -> torch.Tensor:
        """Extract boundary map from mask."""
        # Convert to numpy for boundary detection
        mask_np = mask.detach().cpu().numpy()
        boundary_maps = []
        
        for i in range(mask_np.shape[0]):
            boundary_map = np.zeros_like(mask_np[i, 0])
            
            for j in range(mask_np.shape[1]):
                # Use morphological operations to find boundaries
                from scipy import ndimage
                structure = ndimage.generate_binary_structure(3, 1)
                eroded = ndimage.binary_erosion(mask_np[i, j], structure)
                boundary = mask_np[i, j] - eroded
                boundary_map += boundary
            
            boundary_maps.append(torch.from_numpy(boundary_map).unsqueeze(0))
        
        return torch.cat(boundary_maps, dim=0).to(mask.device)


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    
    This loss combines multiple loss functions to leverage their
    individual strengths for better segmentation performance.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.4,
        ce_weight: float = 0.3,
        focal_weight: float = 0.2,
        boundary_weight: float = 0.1,
        dice_smooth: float = 1e-5,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0
    ):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss
            ce_weight: Weight for Cross-Entropy loss
            focal_weight: Weight for Focal loss
            boundary_weight: Weight for Boundary loss
            dice_smooth: Smoothing factor for Dice loss
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Combined loss
        """
        dice_loss = self.dice_loss(inputs, targets)
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        boundary_loss = self.boundary_loss(inputs, targets)
        
        total_loss = (
            self.dice_weight * dice_loss +
            self.ce_weight * ce_loss +
            self.focal_weight * focal_loss +
            self.boundary_weight * boundary_loss
        )
        
        return total_loss


def create_loss_function(loss_config: dict) -> nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        loss_config: Loss configuration dictionary
        
    Returns:
        Loss function
    """
    loss_name = loss_config.get("name", "dice_ce")
    
    if loss_name == "dice":
        return DiceLoss(
            smooth=loss_config.get("smooth", 1e-5),
            reduction=loss_config.get("reduction", "mean")
        )
    elif loss_name == "dice_ce":
        return DiceCELoss(
            dice_weight=loss_config.get("dice_weight", 0.5),
            ce_weight=loss_config.get("ce_weight", 0.5),
            smooth=loss_config.get("smooth", 1e-5)
        )
    elif loss_name == "focal":
        return FocalLoss(
            alpha=loss_config.get("alpha", 1.0),
            gamma=loss_config.get("gamma", 2.0)
        )
    elif loss_name == "tversky":
        return TverskyLoss(
            alpha=loss_config.get("alpha", 0.3),
            beta=loss_config.get("beta", 0.7),
            smooth=loss_config.get("smooth", 1e-5)
        )
    elif loss_name == "combined":
        return CombinedLoss(
            dice_weight=loss_config.get("dice_weight", 0.4),
            ce_weight=loss_config.get("ce_weight", 0.3),
            focal_weight=loss_config.get("focal_weight", 0.2),
            boundary_weight=loss_config.get("boundary_weight", 0.1)
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
