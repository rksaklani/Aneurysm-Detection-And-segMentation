"""
Visualization Utilities

This module provides utilities for plotting and visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None
):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot losses
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot metrics if available
    if train_metrics and val_metrics:
        for i, (metric_name, train_values) in enumerate(train_metrics.items()):
            if i >= 3:  # Only plot first 3 metrics
                break
            
            row = (i + 1) // 2
            col = (i + 1) % 2
            
            axes[row, col].plot(train_values, label=f'Train {metric_name}')
            if metric_name in val_metrics:
                axes[row, col].plot(val_metrics[metric_name], label=f'Val {metric_name}')
            axes[row, col].set_title(f'{metric_name.title()} Score')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric_name.title())
            axes[row, col].legend()
            axes[row, col].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None
):
    """Plot metrics comparison across models."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        model_names = list(results.keys())
        metric_values = [results[model].get(metric, 0.0) for model in model_names]
        
        bars = axes[i].bar(model_names, metric_values)
        axes[i].set_title(f'{metric.title()} Comparison')
        axes[i].set_ylabel(metric.title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
