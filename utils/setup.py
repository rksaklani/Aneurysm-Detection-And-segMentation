"""
Setup and Configuration Utilities

This module provides utilities for setting up the training environment,
including seed setting, logging configuration, and training setup.
"""

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def setup_logging(
    experiment_dir: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        experiment_dir: Directory for experiment logs
        log_level: Logging level
        log_file: Optional log file name
    """
    # Create experiment directory
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"training_{timestamp}.log"
    
    log_path = experiment_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured - Level: {log_level}, File: {log_path}")


def setup_training(
    config: Dict[str, Any],
    experiment_dir: str,
    model_name: str
) -> Dict[str, Any]:
    """
    Setup training environment.
    
    Args:
        config: Training configuration
        experiment_dir: Experiment directory
        model_name: Name of the model
        
    Returns:
        Training setup dictionary
    """
    # Create model directory
    model_dir = Path(experiment_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup training parameters
    training_setup = {
        "model_dir": str(model_dir),
        "checkpoint_dir": str(model_dir / "checkpoints"),
        "log_dir": str(model_dir / "logs"),
        "results_dir": str(model_dir / "results"),
        "config": config
    }
    
    # Create subdirectories
    for dir_path in ["checkpoint_dir", "log_dir", "results_dir"]:
        Path(training_setup[dir_path]).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training setup completed for {model_name}")
    logger.info(f"Model directory: {training_setup['model_dir']}")
    
    return training_setup


def create_experiment_dir(base_dir: str, config: Dict[str, Any]) -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        config: Experiment configuration
        
    Returns:
        Experiment directory path
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name
    experiment_name = config.get("experiment_name", "medical_segmentation_benchmark")
    experiment_name = f"{experiment_name}_{timestamp}"
    
    # Create experiment directory
    experiment_dir = Path(base_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    
    return str(experiment_dir)


def save_config(config: Dict[str, Any], output_dir: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as YAML
    yaml_path = output_dir / "config.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save as JSON
    json_path = output_dir / "config.json"
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to: {yaml_path} and {json_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    logger.info(f"Configuration loaded from: {config_path}")
    
    return config


def setup_wandb(
    project_name: str,
    experiment_name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None
) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name
        config: Configuration dictionary
        entity: W&B entity (optional)
    """
    try:
        import wandb
        
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            entity=entity
        )
        
        logger.info(f"W&B logging initialized - Project: {project_name}, Experiment: {experiment_name}")
        
    except ImportError:
        logger.warning("Weights & Biases not installed. Skipping W&B setup.")
    except Exception as e:
        logger.error(f"Failed to setup W&B: {e}")


def setup_tensorboard(log_dir: str) -> None:
    """
    Setup TensorBoard logging.
    
    Args:
        log_dir: TensorBoard log directory
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logging initialized - Log dir: {log_dir}")
        
        return writer
        
    except ImportError:
        logger.warning("TensorBoard not installed. Skipping TensorBoard setup.")
        return None
    except Exception as e:
        logger.error(f"Failed to setup TensorBoard: {e}")
        return None


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for training.
    
    Args:
        device: Device specification (optional)
        
    Returns:
        PyTorch device
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    
    return device


def setup_data_parallel(model: torch.nn.Module, device_ids: Optional[List[int]] = None) -> torch.nn.Module:
    """
    Setup data parallel training.
    
    Args:
        model: Model to wrap
        device_ids: List of device IDs (optional)
        
    Returns:
        Data parallel model
    """
    if torch.cuda.device_count() > 1:
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"Data parallel training enabled on devices: {device_ids}")
    
    return model


def setup_mixed_precision(use_amp: bool = True) -> bool:
    """
    Setup automatic mixed precision.
    
    Args:
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Whether AMP is enabled
    """
    if use_amp and torch.cuda.is_available():
        logger.info("Automatic mixed precision enabled")
        return True
    else:
        logger.info("Automatic mixed precision disabled")
        return False


def setup_gradient_clipping(grad_clip_val: Optional[float] = None) -> Optional[float]:
    """
    Setup gradient clipping.
    
    Args:
        grad_clip_val: Gradient clipping value
        
    Returns:
        Gradient clipping value
    """
    if grad_clip_val is not None and grad_clip_val > 0:
        logger.info(f"Gradient clipping enabled with value: {grad_clip_val}")
        return grad_clip_val
    else:
        logger.info("Gradient clipping disabled")
        return None


def setup_early_stopping(
    patience: int = 10,
    min_delta: float = 0.001,
    monitor: str = "val_loss",
    mode: str = "min"
) -> Dict[str, Any]:
    """
    Setup early stopping configuration.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor
        mode: Whether to minimize or maximize the metric
        
    Returns:
        Early stopping configuration
    """
    early_stopping_config = {
        "patience": patience,
        "min_delta": min_delta,
        "monitor": monitor,
        "mode": mode
    }
    
    logger.info(f"Early stopping configured: {early_stopping_config}")
    
    return early_stopping_config
