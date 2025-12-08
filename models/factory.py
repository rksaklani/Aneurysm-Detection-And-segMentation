"""
Model Factory

This module provides a factory pattern for creating and managing
different segmentation models in a unified interface.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

from .unet import UNet, UNet3D, AttentionUNet
from .lightweight_unet3d import LightweightUNet3D
from .unetr import UNETR, UNETRPlusPlus
from .swin_unetr import SwinUNETR
from .nnu_net import NNU_Net
from .transformer_models import Primus, SlimUNETR
from .enhanced_models import ES_UNet, RWKV_UNet, Mamba_UNet
from .multi_scale import StackedUNet, MultiScaleUNet

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating segmentation models.
    
    This class provides a unified interface for creating different
    segmentation models with consistent configuration and initialization.
    """
    
    # Model registry
    MODELS = {
        # Classic models
        "unet": LightweightUNet3D,  # Use lightweight 3D UNet
        "unet2d": UNet,  # 2D UNet for 2D images
        "unet3d": UNet3D,
        "lightweight_unet3d": LightweightUNet3D,
        "attention_unet": AttentionUNet,
        "nnu_net": NNU_Net,
        
        # Transformer models
        "unetr": UNETR,
        "unetr_plus": UNETRPlusPlus,
        "swin_unetr": SwinUNETR,
        "primus": Primus,
        "slim_unetr": SlimUNETR,
        
        # Enhanced models
        "es_unet": ES_UNet,
        "rwkv_unet": RWKV_UNet,
        "mamba_unet": Mamba_UNet,
        
        # Multi-scale models
        "stacked_unet": StackedUNet,
        "multiscale_unet": MultiScaleUNet,
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        config: Dict[str, Any],
        **kwargs
    ) -> nn.Module:
        """
        Create a segmentation model from configuration.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Initialized model
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODELS.keys())}")
        
        model_class = cls.MODELS[model_name]
        
        # Merge config with kwargs
        model_config = {**config, **kwargs}
        
        try:
            model = model_class(**model_config)
            logger.info(f"Created {model_name} model with config: {model_config}")
            return model
        except Exception as e:
            logger.error(f"Failed to create {model_name} model: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available models."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls.MODELS[model_name]
        
        # Get model signature
        import inspect
        signature = inspect.signature(model_class.__init__)
        
        info = {
            "name": model_name,
            "class": model_class.__name__,
            "module": model_class.__module__,
            "parameters": list(signature.parameters.keys()),
            "docstring": model_class.__doc__
        }
        
        return info
    
    @classmethod
    def validate_config(cls, model_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific model.
        
        Args:
            model_name: Name of the model
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = cls.MODELS[model_name]
        
        try:
            # Try to create model with config
            model = model_class(**config)
            return True
        except Exception as e:
            logger.warning(f"Configuration validation failed for {model_name}: {e}")
            return False


class ModelManager:
    """
    Manager class for handling multiple models and their configurations.
    """
    
    def __init__(self):
        self.models = {}
        self.configs = {}
    
    def add_model(
        self,
        name: str,
        model: nn.Module,
        config: Dict[str, Any]
    ) -> None:
        """
        Add a model to the manager.
        
        Args:
            name: Name of the model
            model: Model instance
            config: Model configuration
        """
        self.models[name] = model
        self.configs[name] = config
        logger.info(f"Added model: {name}")
    
    def get_model(self, name: str) -> nn.Module:
        """Get a model by name."""
        if name not in self.models:
            raise KeyError(f"Model {name} not found")
        return self.models[name]
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get model configuration by name."""
        if name not in self.configs:
            raise KeyError(f"Configuration for model {name} not found")
        return self.configs[name]
    
    def list_models(self) -> list:
        """List all available models."""
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
            del self.configs[name]
            logger.info(f"Removed model: {name}")
    
    def clear(self) -> None:
        """Clear all models."""
        self.models.clear()
        self.configs.clear()
        logger.info("Cleared all models")


def create_benchmark_models(config: Dict[str, Any]) -> ModelManager:
    """
    Create all benchmark models from configuration.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        ModelManager with all models
    """
    manager = ModelManager()
    
    models_config = config.get("models", [])
    
    for model_config in models_config:
        model_name = model_config["name"]
        model_params = model_config.get("config", {})
        
        try:
            model = ModelFactory.create_model(model_name, model_params)
            manager.add_model(model_name, model, model_params)
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            continue
    
    return manager


class ModelManager:
    """
    Manager class for handling multiple models.
    """
    
    def __init__(self):
        self.models = {}
        self.configs = {}
    
    def add_model(self, name: str, model: nn.Module, config: Dict[str, Any]):
        """Add a model to the manager."""
        self.models[name] = model
        self.configs[name] = config
    
    def get_model(self, name: str) -> nn.Module:
        """Get a model by name."""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        return self.models[name]
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get model configuration by name."""
        if name not in self.configs:
            raise ValueError(f"Config for model {name} not found")
        return self.configs[name]
    
    def list_models(self) -> List[str]:
        """List all model names."""
        return list(self.models.keys())
    
    def remove_model(self, name: str):
        """Remove a model from the manager."""
        if name in self.models:
            del self.models[name]
        if name in self.configs:
            del self.configs[name]


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get summary information about a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary containing model summary
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = param_size + buffer_size
    
    summary = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size / (1024 * 1024),
        "model_class": model.__class__.__name__,
        "model_module": model.__class__.__module__
    }
    
    return summary


def compare_models(models: Dict[str, nn.Module]) -> Dict[str, Any]:
    """
    Compare multiple models.
    
    Args:
        models: Dictionary of model name to model instance
        
    Returns:
        Comparison results
    """
    comparison = {}
    
    for name, model in models.items():
        comparison[name] = get_model_summary(model)
    
    return comparison
