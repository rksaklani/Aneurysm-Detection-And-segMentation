"""
Evaluation Framework

This module provides evaluation and benchmarking capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import json


class Evaluator:
    """Evaluator for segmentation models."""
    
    def __init__(
        self,
        metrics: List[str],
        save_predictions: bool = False,
        save_visualizations: bool = False,
        output_dir: str = "./evaluation",
        max_samples: int = None
    ):
        self.metrics = metrics
        self.save_predictions = save_predictions
        self.save_visualizations = save_visualizations
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate a single model."""
        print(f"Evaluating {model_name}...")
        
        model.eval()
        device = next(model.parameters()).device
        
        all_metrics = {metric: [] for metric in self.metrics}
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if self.max_samples and sample_count >= self.max_samples:
                    break
                
                # Move batch to device
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)
                
                # Forward pass
                predictions = model(image)
                predictions = torch.sigmoid(predictions)
                predictions = (predictions > 0.5).float()
                
                # Calculate metrics
                batch_metrics = self._calculate_metrics(predictions, mask)
                
                for metric, value in batch_metrics.items():
                    all_metrics[metric].append(value)
                
                sample_count += image.size(0)
        
        # Calculate average metrics
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[metric] = np.mean(values)
        
        print(f"Evaluation completed for {model_name}")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return avg_metrics
    
    def _calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate metrics for a batch."""
        metrics = {}
        
        for metric in self.metrics:
            if metric == "dice":
                metrics[metric] = self._dice_score(predictions, targets)
            elif metric == "iou":
                metrics[metric] = self._iou_score(predictions, targets)
            elif metric == "hausdorff":
                metrics[metric] = self._hausdorff_distance(predictions, targets)
            elif metric == "surface_distance":
                metrics[metric] = self._surface_distance(predictions, targets)
            elif metric == "volume_similarity":
                metrics[metric] = self._volume_similarity(predictions, targets)
        
        return metrics
    
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
        # Simplified implementation - return 0 for now
        return 0.0
    
    def _surface_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate surface distance (simplified)."""
        # Simplified implementation - return 0 for now
        return 0.0
    
    def _volume_similarity(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate volume similarity."""
        pred_vol = pred.sum()
        target_vol = target.sum()
        similarity = 1 - abs(pred_vol - target_vol) / (target_vol + 1e-5)
        return similarity.item()


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self, results: Dict[str, Dict[str, float]]):
        self.results = results
    
    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_report(self, filepath: str):
        """Generate HTML report."""
        html_content = self._generate_html_report()
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def summary(self) -> str:
        """Generate text summary."""
        summary = "Benchmark Results Summary:\n"
        summary += "=" * 50 + "\n"
        
        for model_name, metrics in self.results.items():
            summary += f"\n{model_name}:\n"
            for metric, value in metrics.items():
                summary += f"  {metric}: {value:.4f}\n"
        
        return summary
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical Image Segmentation Benchmark Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Medical Image Segmentation Benchmark Results</h1>
            <table>
                <tr>
                    <th>Model</th>
        """
        
        # Add metric headers
        if self.results:
            first_model = list(self.results.keys())[0]
            for metric in self.results[first_model].keys():
                html += f"<th>{metric}</th>"
        
        html += "</tr>"
        
        # Add data rows
        for model_name, metrics in self.results.items():
            html += f"<tr><td class='metric'>{model_name}</td>"
            for metric, value in metrics.items():
                html += f"<td>{value:.4f}</td>"
            html += "</tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html