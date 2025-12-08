"""
Tests for training framework
"""

import pytest
import torch
import torch.nn as nn
from training import DiceLoss, DiceCELoss, DiceScore, IoUScore


class TestLossFunctions:
    """Test loss function implementations"""
    
    def test_dice_loss(self):
        """Test Dice loss calculation"""
        loss_fn = DiceLoss()
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.item() <= 1
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_dice_ce_loss(self):
        """Test combined Dice-CE loss"""
        loss_fn = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_perfect_prediction_dice_loss(self):
        """Test Dice loss with perfect predictions"""
        loss_fn = DiceLoss()
        
        # Perfect predictions (should give loss = 0)
        predictions = torch.ones(1, 1, 16, 16, 16) * 10  # High confidence
        targets = torch.ones(1, 1, 16, 16, 16)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() < 0.1  # Should be very close to 0
    
    def test_worst_prediction_dice_loss(self):
        """Test Dice loss with worst predictions"""
        loss_fn = DiceLoss()
        
        # Worst predictions (should give loss = 1)
        predictions = torch.ones(1, 1, 16, 16, 16) * (-10)  # High confidence wrong
        targets = torch.ones(1, 1, 16, 16, 16)
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() > 0.9  # Should be very close to 1


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_dice_score(self):
        """Test Dice score calculation"""
        metric_fn = DiceScore()
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        score = metric_fn(predictions, targets)
        
        assert isinstance(score, torch.Tensor)
        assert score.item() >= 0
        assert score.item() <= 1
        assert not torch.isnan(score)
        assert not torch.isinf(score)
    
    def test_iou_score(self):
        """Test IoU score calculation"""
        metric_fn = IoUScore()
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 32, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()
        
        score = metric_fn(predictions, targets)
        
        assert isinstance(score, torch.Tensor)
        assert score.item() >= 0
        assert score.item() <= 1
        assert not torch.isnan(score)
        assert not torch.isinf(score)
    
    def test_perfect_prediction_metrics(self):
        """Test metrics with perfect predictions"""
        dice_fn = DiceScore()
        iou_fn = IoUScore()
        
        # Perfect predictions
        predictions = torch.ones(1, 1, 16, 16, 16) * 10
        targets = torch.ones(1, 1, 16, 16, 16)
        
        dice_score = dice_fn(predictions, targets)
        iou_score = iou_fn(predictions, targets)
        
        assert dice_score.item() > 0.9  # Should be very close to 1
        assert iou_score.item() > 0.9   # Should be very close to 1
    
    def test_no_overlap_metrics(self):
        """Test metrics with no overlap"""
        dice_fn = DiceScore()
        iou_fn = IoUScore()
        
        # No overlap predictions
        predictions = torch.ones(1, 1, 16, 16, 16) * (-10)
        targets = torch.ones(1, 1, 16, 16, 16)
        
        dice_score = dice_fn(predictions, targets)
        iou_score = iou_fn(predictions, targets)
        
        assert dice_score.item() < 0.1  # Should be very close to 0
        assert iou_score.item() < 0.1   # Should be very close to 0


class TestTrainingComponents:
    """Test training components"""
    
    def test_loss_gradient_flow(self):
        """Test that loss gradients flow properly"""
        loss_fn = DiceCELoss()
        
        # Create dummy predictions and targets
        predictions = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
        targets = torch.randint(0, 2, (2, 1, 16, 16, 16)).float()
        
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        assert predictions.grad is not None
        assert not torch.isnan(predictions.grad).any()
        assert not torch.isinf(predictions.grad).any()
    
    def test_metric_consistency(self):
        """Test that metrics are consistent across different inputs"""
        dice_fn = DiceScore()
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            predictions = torch.randn(batch_size, 1, 16, 16, 16)
            targets = torch.randint(0, 2, (batch_size, 1, 16, 16, 16)).float()
            
            score = dice_fn(predictions, targets)
            
            assert isinstance(score, torch.Tensor)
            assert score.item() >= 0
            assert score.item() <= 1


if __name__ == "__main__":
    pytest.main([__file__])
