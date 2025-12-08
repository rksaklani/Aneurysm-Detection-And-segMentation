"""
Benchmark Results and Reporting

This module provides comprehensive benchmarking and reporting utilities
for medical image segmentation models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkResults:
    """
    Comprehensive benchmark results for medical image segmentation models.
    
    This class provides methods for storing, analyzing, and reporting
    benchmark results across multiple models and metrics.
    """
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """
        Initialize benchmark results.
        
        Args:
            results: Dictionary of model results
        """
        self.results = results
        self.models = list(results.keys())
        self.metrics = self._extract_metrics()
        
        logger.info(f"Initialized benchmark results for {len(self.models)} models")
        logger.info(f"Available metrics: {self.metrics}")
    
    def _extract_metrics(self) -> List[str]:
        """Extract available metrics from results."""
        if not self.results:
            return []
        
        # Get metrics from first model
        first_model = list(self.results.keys())[0]
        first_results = self.results[first_model]
        
        if "metrics" in first_results:
            return list(first_results["metrics"].keys())
        else:
            return []
    
    def get_model_results(self, model_name: str) -> Dict[str, Any]:
        """Get results for a specific model."""
        if model_name not in self.results:
            raise KeyError(f"Model {model_name} not found in results")
        return self.results[model_name]
    
    def get_metric_values(self, metric_name: str) -> Dict[str, float]:
        """Get metric values across all models."""
        metric_values = {}
        
        for model_name, model_results in self.results.items():
            if "metrics" in model_results and metric_name in model_results["metrics"]:
                metric_values[model_name] = model_results["metrics"][metric_name]
            else:
                logger.warning(f"Metric {metric_name} not found for model {model_name}")
                metric_values[model_name] = 0.0
        
        return metric_values
    
    def get_best_model(self, metric_name: str) -> Tuple[str, float]:
        """Get the best model for a specific metric."""
        metric_values = self.get_metric_values(metric_name)
        
        if not metric_values:
            raise ValueError(f"No values found for metric {metric_name}")
        
        best_model = max(metric_values, key=metric_values.get)
        best_value = metric_values[best_model]
        
        return best_model, best_value
    
    def get_ranking(self, metric_name: str) -> List[Tuple[str, float]]:
        """Get ranking of models for a specific metric."""
        metric_values = self.get_metric_values(metric_name)
        
        if not metric_values:
            return []
        
        # Sort by metric value (descending for higher is better metrics)
        ranking = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        
        return ranking
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all models and metrics."""
        summary = {
            "num_models": len(self.models),
            "num_metrics": len(self.metrics),
            "models": self.models,
            "metrics": self.metrics
        }
        
        # Calculate statistics for each metric
        metric_stats = {}
        for metric in self.metrics:
            values = list(self.get_metric_values(metric).values())
            if values:
                metric_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values)
                }
        
        summary["metric_statistics"] = metric_stats
        
        return summary
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all models and metrics."""
        data = []
        
        for model_name in self.models:
            model_results = self.results[model_name]
            row = {"Model": model_name}
            
            if "metrics" in model_results:
                for metric_name, metric_value in model_results["metrics"].items():
                    row[metric_name] = metric_value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save(self, file_path: str) -> None:
        """Save benchmark results to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Saved benchmark results to: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> "BenchmarkResults":
        """Load benchmark results from file."""
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        return cls(results)
    
    def generate_report(self, output_path: str) -> None:
        """Generate comprehensive HTML report."""
        report = BenchmarkReport(self)
        report.generate_html(output_path)
    
    def summary(self) -> str:
        """Generate text summary of benchmark results."""
        summary_stats = self.get_summary_statistics()
        
        # Create summary text
        summary_text = f"""
Benchmark Results Summary:
- Number of models: {summary_stats['num_models']}
- Number of metrics: {summary_stats['num_metrics']}
- Models: {', '.join(summary_stats['models'])}
- Metrics: {', '.join(summary_stats['metrics'])}

Model Rankings:
"""
        
        # Add rankings for each metric
        for metric in self.metrics:
            ranking = self.get_ranking(metric)
            if ranking:
                best_model, best_value = ranking[0]
                summary_text += f"\n{metric.upper()}:\n"
                summary_text += f"  Best: {best_model} ({best_value:.4f})\n"
                
                for i, (model, value) in enumerate(ranking[:3]):
                    summary_text += f"  {i+1}. {model}: {value:.4f}\n"
        
        return summary_text


class BenchmarkReport:
    """
    HTML report generator for benchmark results.
    
    This class generates comprehensive HTML reports with visualizations
    and detailed analysis of benchmark results.
    """
    
    def __init__(self, benchmark_results: BenchmarkResults):
        """
        Initialize benchmark report.
        
        Args:
            benchmark_results: Benchmark results to report
        """
        self.results = benchmark_results
        self.models = benchmark_results.models
        self.metrics = benchmark_results.metrics
    
    def generate_html(self, output_path: str) -> None:
        """Generate HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._generate_html_content()
        
        # Save HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated benchmark report: {output_path}")
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Medical Image Segmentation Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .ranking {{
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }}
        .best-model {{
            background-color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2c5aa0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Medical Image Segmentation Benchmark Report</h1>
        
        {self._generate_summary_section()}
        
        {self._generate_metrics_section()}
        
        {self._generate_ranking_section()}
        
        {self._generate_comparison_table()}
        
        <footer>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_summary_section(self) -> str:
        """Generate summary section."""
        summary_stats = self.results.get_summary_statistics()
        
        return f"""
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Number of models:</strong> {summary_stats['num_models']}</p>
            <p><strong>Number of metrics:</strong> {summary_stats['num_metrics']}</p>
            <p><strong>Models evaluated:</strong> {', '.join(summary_stats['models'])}</p>
            <p><strong>Metrics used:</strong> {', '.join(summary_stats['metrics'])}</p>
        </div>
        """
    
    def _generate_metrics_section(self) -> str:
        """Generate metrics section."""
        html = "<h2>Metrics Analysis</h2>"
        
        for metric in self.metrics:
            metric_values = self.results.get_metric_values(metric)
            best_model, best_value = self.results.get_best_model(metric)
            
            html += f"""
            <div class="metric-section">
                <h3>{metric.upper()}</h3>
                <div class="best-model">
                    Best Model: {best_model} ({best_value:.4f})
                </div>
                <p>Mean: {np.mean(list(metric_values.values())):.4f}</p>
                <p>Std: {np.std(list(metric_values.values())):.4f}</p>
                <p>Min: {np.min(list(metric_values.values())):.4f}</p>
                <p>Max: {np.max(list(metric_values.values())):.4f}</p>
            </div>
            """
        
        return html
    
    def _generate_ranking_section(self) -> str:
        """Generate ranking section."""
        html = "<h2>Model Rankings</h2>"
        
        for metric in self.metrics:
            ranking = self.results.get_ranking(metric)
            
            html += f"""
            <div class="metric-section">
                <h3>{metric.upper()}</h3>
                <div class="ranking">
            """
            
            for i, (model, value) in enumerate(ranking):
                html += f"<p>{i+1}. {model}: {value:.4f}</p>"
            
            html += "</div></div>"
        
        return html
    
    def _generate_comparison_table(self) -> str:
        """Generate comparison table."""
        df = self.results.create_comparison_table()
        
        html = "<h2>Comparison Table</h2>"
        html += "<table>"
        
        # Header
        html += "<tr>"
        for col in df.columns:
            html += f"<th>{col}</th>"
        html += "</tr>"
        
        # Rows
        for _, row in df.iterrows():
            html += "<tr>"
            for col in df.columns:
                if col == "Model":
                    html += f"<td>{row[col]}</td>"
                else:
                    html += f"<td class='metric-value'>{row[col]:.4f}</td>"
            html += "</tr>"
        
        html += "</table>"
        
        return html


def create_benchmark_summary(results: BenchmarkResults) -> str:
    """
    Create a comprehensive summary of benchmark results.
    
    Args:
        results: Benchmark results
        
    Returns:
        Summary string
    """
    summary = []
    
    # Overall summary
    summary.append("=" * 50)
    summary.append("MEDICAL IMAGE SEGMENTATION BENCHMARK SUMMARY")
    summary.append("=" * 50)
    
    # Basic statistics
    summary_stats = results.get_summary_statistics()
    summary.append(f"Models evaluated: {summary_stats['num_models']}")
    summary.append(f"Metrics used: {summary_stats['num_metrics']}")
    summary.append("")
    
    # Model rankings
    summary.append("MODEL RANKINGS:")
    summary.append("-" * 30)
    
    for metric in results.metrics:
        ranking = results.get_ranking(metric)
        if ranking:
            summary.append(f"\n{metric.upper()}:")
            for i, (model, value) in enumerate(ranking):
                summary.append(f"  {i+1}. {model}: {value:.4f}")
    
    # Best models
    summary.append("\nBEST MODELS:")
    summary.append("-" * 20)
    
    for metric in results.metrics:
        try:
            best_model, best_value = results.get_best_model(metric)
            summary.append(f"{metric.upper()}: {best_model} ({best_value:.4f})")
        except ValueError:
            summary.append(f"{metric.upper()}: No data available")
    
    summary.append("=" * 50)
    
    return "\n".join(summary)
