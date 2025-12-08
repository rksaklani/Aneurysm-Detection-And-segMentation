"""
I/O Utilities

This module provides utilities for file I/O operations.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert OmegaConf DictConfig to regular dict if needed
    try:
        from omegaconf import DictConfig, OmegaConf
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
    except ImportError:
        pass
    
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        # Default to JSON
        with open(f"{filepath}.json", 'w') as f:
            json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    elif filepath.suffix.lower() == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {filepath.suffix}")


def create_experiment_dir(base_dir: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    
    if config and "experiment_name" in config:
        experiment_name = config["experiment_name"]
    
    experiment_dir = Path(base_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return str(experiment_dir)
