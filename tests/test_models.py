"""
Tests for model implementations
"""

import pytest
import torch
import torch.nn as nn
from models import ModelFactory, ModelManager


class TestModelFactory:
    """Test ModelFactory functionality"""
    
    def test_available_models(self):
        """Test that all expected models are available"""
        available_models = ModelFactory.get_available_models()
        expected_models = [
            "unet", "unet3d", "attention_unet", "nnu_net",
            "unetr", "unetr_plus", "swin_unetr", "primus", "slim_unetr",
            "es_unet", "rwkv_unet", "mamba_unet",
            "stacked_unet", "multiscale_unet"
        ]
        
        for model in expected_models:
            assert model in available_models, f"Model {model} not found in available models"
    
    def test_create_unet(self):
        """Test U-Net model creation"""
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 64,
            "depth": 4
        }
        
        model = ModelFactory.create_model("unet", config)
        assert isinstance(model, nn.Module)
        assert model.in_channels == 1
        assert model.out_channels == 1
    
    def test_create_unetr(self):
        """Test UNETR model creation"""
        config = {
            "img_size": [128, 128, 128],
            "in_channels": 1,
            "out_channels": 1,
            "embed_dim": 768,
            "patch_size": 16,
            "num_heads": 12
        }
        
        model = ModelFactory.create_model("unetr", config)
        assert isinstance(model, nn.Module)
    
    def test_invalid_model(self):
        """Test that invalid model names raise error"""
        with pytest.raises(ValueError):
            ModelFactory.create_model("invalid_model", {})
    
    def test_model_forward_pass(self):
        """Test that models can perform forward pass"""
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 32,
            "depth": 3
        }
        
        model = ModelFactory.create_model("unet", config)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 1, 64, 64, 64)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestModelManager:
    """Test ModelManager functionality"""
    
    def test_add_model(self):
        """Test adding models to manager"""
        manager = ModelManager()
        
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 64,
            "depth": 4
        }
        
        model = ModelFactory.create_model("unet", config)
        manager.add_model("test_unet", model, config)
        
        assert "test_unet" in manager.list_models()
        assert manager.get_model("test_unet") == model
        assert manager.get_config("test_unet") == config
    
    def test_get_nonexistent_model(self):
        """Test getting non-existent model raises error"""
        manager = ModelManager()
        
        with pytest.raises(KeyError):
            manager.get_model("nonexistent")
    
    def test_remove_model(self):
        """Test removing models from manager"""
        manager = ModelManager()
        
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 64,
            "depth": 4
        }
        
        model = ModelFactory.create_model("unet", config)
        manager.add_model("test_unet", model, config)
        
        assert "test_unet" in manager.list_models()
        
        manager.remove_model("test_unet")
        assert "test_unet" not in manager.list_models()


@pytest.mark.gpu
class TestGPUModels:
    """Test models on GPU if available"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_gpu_forward(self):
        """Test model forward pass on GPU"""
        config = {
            "in_channels": 1,
            "out_channels": 1,
            "base_features": 32,
            "depth": 3
        }
        
        model = ModelFactory.create_model("unet", config)
        model = model.cuda()
        model.eval()
        
        # Create dummy input on GPU
        x = torch.randn(1, 1, 32, 32, 32).cuda()
        
        with torch.no_grad():
            output = model(x)
        
        assert output.device.type == "cuda"
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
