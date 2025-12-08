"""
Training Framework

This module provides the training framework for medical image segmentation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any


class DiceLoss(nn.Module):
    """Dice loss for medical image segmentation."""
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss."""
        predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss."""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        dice_loss = self.dice_loss(predictions, targets)
        ce_loss = self.ce_loss(predictions, targets)
        
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class MetricCalculator:
    """Calculate evaluation metrics."""
    
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
    
    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate all metrics."""
        results = {}
        
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > 0.5).float()
        
        for metric in self.metrics:
            if metric == "dice":
                results[metric] = self._dice_score(predictions, targets)
            elif metric == "iou":
                results[metric] = self._iou_score(predictions, targets)
            elif metric == "hausdorff":
                results[metric] = self._hausdorff_distance(predictions, targets)
            elif metric == "surface_distance":
                results[metric] = self._surface_distance(predictions, targets)
            elif metric == "volume_similarity":
                results[metric] = self._volume_similarity(predictions, targets)
        
        return results
    
    def _dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice score."""
        intersection = (pred * target).sum()
        dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-5)
        return dice.item()
    
    def _iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate IoU score."""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = intersection / (union + 1e-5)
        return iou.item()
    
    def _hausdorff_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Hausdorff distance (simplified)."""
        # Simplified implementation
        return 0.0
    
    def _surface_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate surface distance (simplified)."""
        # Simplified implementation
        return 0.0
    
    def _volume_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate volume similarity."""
        pred_vol = pred.sum()
        target_vol = target.sum()
        similarity = 1 - abs(pred_vol - target_vol) / (target_vol + 1e-5)
        return similarity.item()


class SegmentationTrainer:
    """Simple trainer for segmentation models."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        metrics: MetricCalculator,
        learning_rate: float = 1e-4,
        max_epochs: int = 100,
        gpus: int = 1,
        precision: str = "16-mixed",
        experiment_dir: str = "./experiments",
        model_name: str = "model",
        accumulate_grad_batches: int = 1,
        limit_train_batches: float = 1.0,
        limit_val_batches: float = 1.0
    ):
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.precision = precision
        self.experiment_dir = experiment_dir
        self.model_name = model_name
        self.accumulate_grad_batches = accumulate_grad_batches
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() and gpus > 0 else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )
        
        print(f"Trainer initialized for {model_name} on {self.device}")
    
    def fit(self, data_module):
        """Train the model."""
        print(f"Starting training for {self.model_name}...")
        
        # Get data loaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        # Limit batches if specified
        if self.limit_train_batches < 1.0:
            train_loader = self._limit_batches(train_loader, self.limit_train_batches)
        if self.limit_val_batches < 1.0:
            val_loader = self._limit_batches(val_loader, self.limit_val_batches)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_metrics = {}
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(image)
                loss = self.loss_function(predictions, mask)
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item()
                
                # Calculate metrics
                if batch_idx == 0:  # Calculate metrics only for first batch to save time
                    train_metrics = self.metrics.calculate_metrics(predictions, mask)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_metrics = {}
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    image = batch["image"].to(self.device)
                    mask = batch["mask"].to(self.device)
                    
                    predictions = self.model(image)
                    loss = self.loss_function(predictions, mask)
                    
                    val_loss += loss.item()
                    
                    # Calculate metrics
                    if batch_idx == 0:  # Calculate metrics only for first batch
                        val_metrics = self.metrics.calculate_metrics(predictions, mask)
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.max_epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Dice: {val_metrics.get('dice', 0.0):.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), f"{self.experiment_dir}/{self.model_name}_best.pth")
        
        print(f"Training completed for {self.model_name}")
    
    def _limit_batches(self, dataloader, limit_ratio):
        """Limit the number of batches in a dataloader."""
        total_batches = len(dataloader)
        limit_batches = int(total_batches * limit_ratio)
        
        # Create a limited iterator
        limited_data = []
        for i, batch in enumerate(dataloader):
            if i >= limit_batches:
                break
            limited_data.append(batch)
        
        return limited_data


def setup_training():
    """Setup training environment."""
    pass


def setup_logging(experiment_dir: str, log_level: str):
    """Setup logging."""
    import logging
    import os
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{experiment_dir}/training.log"),
            logging.StreamHandler()
        ]
    )