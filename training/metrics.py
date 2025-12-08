"""
Evaluation Metrics for Medical Image Segmentation

This module provides comprehensive evaluation metrics for medical image
segmentation tasks, including Dice coefficient, IoU, Hausdorff distance, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
import logging

logger = logging.getLogger(__name__)


class DiceScore(nn.Module):
    """
    Dice Score metric for medical image segmentation.
    
    The Dice score measures the overlap between predicted and ground truth masks.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize Dice Score.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            threshold: Threshold for binary conversion
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Dice score
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)
        
        # Calculate Dice score
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        if self.reduction == "mean":
            return dice.mean()
        elif self.reduction == "sum":
            return dice.sum()
        else:
            return dice


class IoUScore(nn.Module):
    """
    Intersection over Union (IoU) metric.
    
    IoU measures the overlap between predicted and ground truth masks.
    """
    
    def __init__(
        self,
        smooth: float = 1e-5,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize IoU Score.
        
        Args:
            smooth: Smoothing factor
            threshold: Threshold for binary conversion
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            IoU score
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Flatten tensors
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        if self.reduction == "mean":
            return iou.mean()
        elif self.reduction == "sum":
            return iou.sum()
        else:
            return iou


class HausdorffDistance(nn.Module):
    """
    Hausdorff Distance metric for medical image segmentation.
    
    Hausdorff distance measures the maximum distance between the boundaries
    of predicted and ground truth masks.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize Hausdorff Distance.
        
        Args:
            threshold: Threshold for binary conversion
            reduction: Reduction method
        """
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Hausdorff distance
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Convert to numpy for Hausdorff distance calculation
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        hausdorff_distances = []
        
        for i in range(inputs_np.shape[0]):
            # Get boundary points
            pred_boundary = self._get_boundary_points(inputs_np[i, 0])
            target_boundary = self._get_boundary_points(targets_np[i, 0])
            
            if len(pred_boundary) == 0 or len(target_boundary) == 0:
                hausdorff_distances.append(0.0)
                continue
            
            # Calculate Hausdorff distance
            try:
                hd = max(
                    directed_hausdorff(pred_boundary, target_boundary)[0],
                    directed_hausdorff(target_boundary, pred_boundary)[0]
                )
                hausdorff_distances.append(hd)
            except:
                hausdorff_distances.append(0.0)
        
        hausdorff_tensor = torch.tensor(hausdorff_distances, device=inputs.device)
        
        if self.reduction == "mean":
            return hausdorff_tensor.mean()
        elif self.reduction == "sum":
            return hausdorff_tensor.sum()
        else:
            return hausdorff_tensor
    
    def _get_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from mask."""
        from scipy import ndimage
        
        # Find boundary using morphological operations
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure)
        boundary = mask - eroded
        
        # Get coordinates of boundary points
        boundary_points = np.where(boundary > 0)
        return np.column_stack(boundary_points)


class SurfaceDistance(nn.Module):
    """
    Surface Distance metric for medical image segmentation.
    
    Surface distance measures the average distance between the surfaces
    of predicted and ground truth masks.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize Surface Distance.
        
        Args:
            threshold: Threshold for binary conversion
            reduction: Reduction method
        """
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Surface distance
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Convert to numpy for surface distance calculation
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        surface_distances = []
        
        for i in range(inputs_np.shape[0]):
            # Calculate surface distance
            try:
                sd = self._calculate_surface_distance(inputs_np[i, 0], targets_np[i, 0])
                surface_distances.append(sd)
            except:
                surface_distances.append(0.0)
        
        surface_distance_tensor = torch.tensor(surface_distances, device=inputs.device)
        
        if self.reduction == "mean":
            return surface_distance_tensor.mean()
        elif self.reduction == "sum":
            return surface_distance_tensor.sum()
        else:
            return surface_distance_tensor
    
    def _calculate_surface_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate surface distance between two masks."""
        # Calculate distance transforms
        pred_dt = distance_transform_edt(1 - pred)
        target_dt = distance_transform_edt(1 - target)
        
        # Get surface points
        pred_surface = (pred > 0) & (pred_dt <= 1)
        target_surface = (target > 0) & (target_dt <= 1)
        
        if np.sum(pred_surface) == 0 or np.sum(target_surface) == 0:
            return 0.0
        
        # Calculate average surface distance
        pred_to_target = np.mean(pred_dt[target_surface])
        target_to_pred = np.mean(target_dt[pred_surface])
        
        return (pred_to_target + target_to_pred) / 2.0


class VolumeSimilarity(nn.Module):
    """
    Volume Similarity metric for medical image segmentation.
    
    Volume similarity measures the relative difference in volume between
    predicted and ground truth masks.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize Volume Similarity.
        
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
            Volume similarity
        """
        # Apply sigmoid
        inputs = torch.sigmoid(inputs)
        
        # Calculate volumes
        pred_volume = inputs.sum(dim=(2, 3, 4))
        target_volume = targets.sum(dim=(2, 3, 4))
        
        # Calculate volume similarity
        volume_similarity = 1.0 - torch.abs(pred_volume - target_volume) / (pred_volume + target_volume + 1e-8)
        
        if self.reduction == "mean":
            return volume_similarity.mean()
        elif self.reduction == "sum":
            return volume_similarity.sum()
        else:
            return volume_similarity


class Sensitivity(nn.Module):
    """
    Sensitivity (Recall) metric for medical image segmentation.
    
    Sensitivity measures the proportion of true positives that are correctly identified.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize Sensitivity.
        
        Args:
            threshold: Threshold for binary conversion
            reduction: Reduction method
        """
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Sensitivity
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Calculate true positives and false negatives
        tp = (inputs * targets).sum(dim=(2, 3, 4))
        fn = ((1 - inputs) * targets).sum(dim=(2, 3, 4))
        
        # Calculate sensitivity
        sensitivity = tp / (tp + fn + 1e-8)
        
        if self.reduction == "mean":
            return sensitivity.mean()
        elif self.reduction == "sum":
            return sensitivity.sum()
        else:
            return sensitivity


class Specificity(nn.Module):
    """
    Specificity metric for medical image segmentation.
    
    Specificity measures the proportion of true negatives that are correctly identified.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        reduction: str = "mean"
    ):
        """
        Initialize Specificity.
        
        Args:
            threshold: Threshold for binary conversion
            reduction: Reduction method
        """
        super().__init__()
        self.threshold = threshold
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Specificity
        """
        # Apply sigmoid and threshold
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Calculate true negatives and false positives
        tn = ((1 - inputs) * (1 - targets)).sum(dim=(2, 3, 4))
        fp = (inputs * (1 - targets)).sum(dim=(2, 3, 4))
        
        # Calculate specificity
        specificity = tn / (tn + fp + 1e-8)
        
        if self.reduction == "mean":
            return specificity.mean()
        elif self.reduction == "sum":
            return specificity.sum()
        else:
            return specificity


class MetricCalculator:
    """
    Comprehensive metric calculator for medical image segmentation.
    
    This class provides a unified interface for calculating multiple metrics
    and generating comprehensive evaluation reports.
    """
    
    def __init__(self, metrics: List[str] = None):
        """
        Initialize Metric Calculator.
        
        Args:
            metrics: List of metrics to calculate
        """
        if metrics is None:
            metrics = ["dice", "iou", "hausdorff", "surface_distance", "volume_similarity"]
        
        self.metrics = {}
        
        for metric_name in metrics:
            if metric_name == "dice":
                self.metrics[metric_name] = DiceScore()
            elif metric_name == "iou":
                self.metrics[metric_name] = IoUScore()
            elif metric_name == "hausdorff":
                self.metrics[metric_name] = HausdorffDistance()
            elif metric_name == "surface_distance":
                self.metrics[metric_name] = SurfaceDistance()
            elif metric_name == "volume_similarity":
                self.metrics[metric_name] = VolumeSimilarity()
            elif metric_name == "sensitivity":
                self.metrics[metric_name] = Sensitivity()
            elif metric_name == "specificity":
                self.metrics[metric_name] = Specificity()
            else:
                logger.warning(f"Unknown metric: {metric_name}")
    
    def calculate_metrics(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Calculate all metrics.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        for metric_name, metric_fn in self.metrics.items():
            try:
                value = metric_fn(inputs, targets)
                results[metric_name] = value.item() if isinstance(value, torch.Tensor) else value
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {e}")
                results[metric_name] = 0.0
        
        return results
    
    def calculate_batch_metrics(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Calculate metrics for a batch of predictions.
        
        Args:
            inputs: Predicted logits (B, C, H, W, D)
            targets: Ground truth masks (B, C, H, W, D)
            
        Returns:
            Dictionary of batch metric values
        """
        batch_results = {}
        
        for metric_name, metric_fn in self.metrics.items():
            try:
                values = metric_fn(inputs, targets)
                if isinstance(values, torch.Tensor):
                    batch_results[metric_name] = values.detach().cpu().numpy()
                else:
                    batch_results[metric_name] = values
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_name}: {e}")
                batch_results[metric_name] = np.zeros(inputs.shape[0])
        
        return batch_results
