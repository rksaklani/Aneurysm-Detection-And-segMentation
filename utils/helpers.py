"""
Helper Utilities

This module provides various helper functions.
"""

import torch
import time
from typing import Optional


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get the appropriate device for computation."""
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
