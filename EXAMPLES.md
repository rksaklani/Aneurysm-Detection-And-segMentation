# Medical Image Segmentation Benchmarking - Examples

This document provides comprehensive examples for using the medical image segmentation benchmarking framework.

## 🚀 Quick Start Examples

### Example 1: Basic Benchmarking

```bash
# Run basic benchmarking with default settings
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/basic_benchmark
```

### Example 2: Custom Model Selection

```bash
# Train specific models
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/custom_models \
    --models unet unetr swin_unetr \
    --batch_size 2 \
    --max_epochs 100
```

### Example 3: High-Performance Training

```bash
# Use multiple GPUs and larger batch size
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/high_performance \
    --models unet unetr swin_unetr primus \
    --batch_size 4 \
    --max_epochs 200 \
    --gpus 2
```

## 🔧 Configuration Examples

### Example 1: Basic Configuration

```yaml
# configs/basic_config.yaml
dataset:
  name: "ADAM"
  data_path: "/path/to/ADAM_release_subjs"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 2
  patch_size: [128, 128, 128]
  overlap: 0.5

models:
  - name: "unet"
    config:
      in_channels: 1
      out_channels: 1
      base_features: 64
      depth: 4

training:
  max_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-5

loss:
  name: "dice_ce"
  dice_weight: 0.5
  ce_weight: 0.5

metrics:
  - name: "dice"
  - name: "iou"
  - name: "hausdorff_distance"
```

### Example 2: Advanced Configuration

```yaml
# configs/advanced_config.yaml
dataset:
  name: "ADAM"
  data_path: "/path/to/ADAM_release_subjs"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 4
  patch_size: [96, 96, 96]
  overlap: 0.5
  load_patches: true

preprocessing:
  normalize_method: "z_score"
  clip_percentiles: [0.5, 99.5]
  bias_correction: true
  noise_reduction: true
  contrast_enhancement: false

augmentation:
  enabled: true
  rotation_range: 15
  translation_range: 0.1
  scale_range: [0.9, 1.1]
  brightness_range: 0.2
  contrast_range: 0.2
  elastic_alpha: 1000
  elastic_sigma: 30
  elastic_probability: 0.5
  flip_probability: 0.5
  apply_probability: 0.8

models:
  - name: "unet"
    config:
      in_channels: 1
      out_channels: 1
      base_features: 64
      depth: 4
      dropout: 0.1
      bilinear: true
      
  - name: "unetr"
    config:
      img_size: [96, 96, 96]
      in_channels: 1
      out_channels: 1
      embed_dim: 768
      patch_size: 16
      num_heads: 12
      num_layers: 12
      mlp_ratio: 4.0
      qkv_bias: true
      dropout: 0.1
      attn_dropout: 0.1
      drop_path: 0.1
      
  - name: "swin_unetr"
    config:
      img_size: [96, 96, 96]
      in_channels: 1
      out_channels: 1
      depths: [2, 2, 2, 2]
      num_heads: [3, 6, 12, 24]
      feature_size: 48
      norm_name: "instance"
      drop_rate: 0.0
      attn_drop_rate: 0.0
      dropout_path_rate: 0.0
      normalize: true
      use_checkpoint: false
      spatial_dims: 3

training:
  max_epochs: 200
  learning_rate: 1e-4
  weight_decay: 1e-5
  scheduler: "cosine"
  warmup_epochs: 10
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  precision: "16-mixed"

loss:
  name: "dice_ce"
  dice_weight: 0.5
  ce_weight: 0.5
  smooth: 1e-5

metrics:
  - name: "dice"
  - name: "iou"
  - name: "hausdorff_distance"
  - name: "surface_distance"
  - name: "volume_similarity"
  - name: "sensitivity"
  - name: "specificity"

evaluation:
  save_predictions: true
  save_visualizations: true
  threshold: 0.5
  post_process: true

logging:
  project_name: "medical_segmentation_benchmark"
  experiment_name: "advanced_benchmark"
  log_dir: "./logs"
  save_dir: "./checkpoints"
  log_every_n_steps: 10
  val_check_interval: 1.0

hardware:
  gpus: 2
  num_nodes: 1
  strategy: "auto"

seed: 42
deterministic: true
benchmark: false
```

### Example 3: Research Configuration

```yaml
# configs/research_config.yaml
dataset:
  name: "ADAM"
  data_path: "/path/to/ADAM_release_subjs"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 1
  patch_size: [192, 192, 192]
  overlap: 0.5
  load_patches: false

preprocessing:
  normalize_method: "robust"
  clip_percentiles: [1.0, 99.0]
  bias_correction: true
  noise_reduction: true
  contrast_enhancement: true

augmentation:
  enabled: true
  rotation_range: 20
  translation_range: 0.15
  scale_range: [0.85, 1.15]
  brightness_range: 0.3
  contrast_range: 0.3
  elastic_alpha: 1500
  elastic_sigma: 40
  elastic_probability: 0.7
  flip_probability: 0.5
  apply_probability: 0.9

models:
  - name: "unet"
    config:
      in_channels: 1
      out_channels: 1
      base_features: 32
      depth: 5
      dropout: 0.2
      bilinear: false
      
  - name: "attention_unet"
    config:
      in_channels: 1
      out_channels: 1
      base_features: 64
      depth: 4
      dropout: 0.1
      
  - name: "unetr"
    config:
      img_size: [192, 192, 192]
      in_channels: 1
      out_channels: 1
      embed_dim: 768
      patch_size: 16
      num_heads: 12
      num_layers: 12
      mlp_ratio: 4.0
      qkv_bias: true
      dropout: 0.1
      attn_dropout: 0.1
      drop_path: 0.1
      
  - name: "swin_unetr"
    config:
      img_size: [192, 192, 192]
      in_channels: 1
      out_channels: 1
      depths: [2, 2, 2, 2]
      num_heads: [3, 6, 12, 24]
      feature_size: 48
      norm_name: "instance"
      drop_rate: 0.0
      attn_drop_rate: 0.0
      dropout_path_rate: 0.0
      normalize: true
      use_checkpoint: true
      spatial_dims: 3

training:
  max_epochs: 300
  learning_rate: 5e-5
  weight_decay: 1e-4
  scheduler: "cosine"
  warmup_epochs: 20
  gradient_clip_val: 0.5
  accumulate_grad_batches: 2
  precision: "16-mixed"

loss:
  name: "combined"
  dice_weight: 0.4
  ce_weight: 0.3
  focal_weight: 0.2
  boundary_weight: 0.1

metrics:
  - name: "dice"
  - name: "iou"
  - name: "hausdorff_distance"
  - name: "surface_distance"
  - name: "volume_similarity"
  - name: "sensitivity"
  - name: "specificity"

evaluation:
  save_predictions: true
  save_visualizations: true
  threshold: 0.5
  post_process: true

logging:
  project_name: "medical_segmentation_research"
  experiment_name: "research_benchmark"
  log_dir: "./logs"
  save_dir: "./checkpoints"
  log_every_n_steps: 5
  val_check_interval: 0.5

hardware:
  gpus: 4
  num_nodes: 1
  strategy: "ddp"

seed: 42
deterministic: true
benchmark: false
```

## 🏃‍♂️ Training Examples

### Example 1: Single Model Training

```bash
# Train only U-Net
python main.py \
    --config configs/basic_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/unet_only \
    --models unet \
    --max_epochs 100 \
    --learning_rate 1e-4
```

### Example 2: Multi-GPU Training

```bash
# Train with multiple GPUs
python main.py \
    --config configs/advanced_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/multi_gpu \
    --models unet unetr swin_unetr \
    --gpus 2 \
    --batch_size 4
```

### Example 3: Long Training Run

```bash
# Extended training for research
python main.py \
    --config configs/research_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/long_training \
    --models unet unetr swin_unetr primus \
    --max_epochs 500 \
    --learning_rate 5e-5 \
    --gpus 4
```

## 📊 Evaluation Examples

### Example 1: Basic Evaluation

```bash
# Evaluate trained models
python main.py \
    --config configs/basic_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/evaluation \
    --models unet unetr \
    --eval_only
```

### Example 2: Evaluation with Visualizations

```bash
# Evaluate with prediction saving and visualizations
python main.py \
    --config configs/advanced_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/evaluation_with_viz \
    --models unet unetr swin_unetr \
    --eval_only \
    --save_predictions \
    --save_visualizations
```

### Example 3: Custom Evaluation

```python
# Custom evaluation script
from evaluation import Evaluator, BenchmarkResults
from models import ModelFactory
from data import ADAMDataModule
import torch

# Load model
model = ModelFactory.create_model("unet", {
    "in_channels": 1,
    "out_channels": 1,
    "base_features": 64,
    "depth": 4
})

# Load weights
model.load_state_dict(torch.load("experiments/unet_only/unet_best.pth"))
model.eval()

# Create data module
data_module = ADAMDataModule(
    data_path="/path/to/ADAM_release_subjs",
    batch_size=1,
    patch_size=(128, 128, 128)
)

# Create evaluator
evaluator = Evaluator(
    metrics=["dice", "iou", "hausdorff_distance"],
    save_predictions=True,
    save_visualizations=True
)

# Evaluate
results = evaluator.evaluate_model(
    model=model,
    data_loader=data_module.test_dataloader(),
    model_name="unet"
)

print(f"Dice Score: {results['metrics']['dice']:.4f}")
print(f"IoU Score: {results['metrics']['iou']:.4f}")
```

## 🔍 Analysis Examples

### Example 1: Load and Analyze Results

```python
from evaluation import BenchmarkResults
import matplotlib.pyplot as plt
import pandas as pd

# Load benchmark results
results = BenchmarkResults.load("experiments/benchmark_results.json")

# Get summary statistics
summary = results.get_summary_statistics()
print(f"Number of models: {summary['num_models']}")
print(f"Number of metrics: {summary['num_metrics']}")

# Get best model for each metric
for metric in results.metrics:
    best_model, best_score = results.get_best_model(metric)
    print(f"{metric.upper()}: {best_model} ({best_score:.4f})")

# Get rankings
for metric in results.metrics:
    ranking = results.get_ranking(metric)
    print(f"\n{metric.upper()} Ranking:")
    for i, (model, score) in enumerate(ranking):
        print(f"  {i+1}. {model}: {score:.4f}")
```

### Example 2: Create Comparison Plots

```python
import matplotlib.pyplot as plt
import seaborn as sns
from evaluation import BenchmarkResults

# Load results
results = BenchmarkResults.load("experiments/benchmark_results.json")

# Create comparison table
df = results.create_comparison_table()

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Dice score comparison
axes[0, 0].bar(df['Model'], df['dice'])
axes[0, 0].set_title('Dice Score Comparison')
axes[0, 0].set_ylabel('Dice Score')

# IoU score comparison
axes[0, 1].bar(df['Model'], df['iou'])
axes[0, 1].set_title('IoU Score Comparison')
axes[0, 1].set_ylabel('IoU Score')

# Hausdorff distance comparison
axes[1, 0].bar(df['Model'], df['hausdorff_distance'])
axes[1, 0].set_title('Hausdorff Distance Comparison')
axes[1, 0].set_ylabel('Hausdorff Distance')

# Surface distance comparison
axes[1, 1].bar(df['Model'], df['surface_distance'])
axes[1, 1].set_title('Surface Distance Comparison')
axes[1, 1].set_ylabel('Surface Distance')

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Example 3: Statistical Analysis

```python
import numpy as np
from scipy import stats
from evaluation import BenchmarkResults

# Load results
results = BenchmarkResults.load("experiments/benchmark_results.json")

# Get metric values
dice_values = results.get_metric_values("dice")
iou_values = results.get_metric_values("iou")

# Calculate statistics
dice_stats = {
    "mean": np.mean(list(dice_values.values())),
    "std": np.std(list(dice_values.values())),
    "min": np.min(list(dice_values.values())),
    "max": np.max(list(dice_values.values())),
    "median": np.median(list(dice_values.values()))
}

iou_stats = {
    "mean": np.mean(list(iou_values.values())),
    "std": np.std(list(iou_values.values())),
    "min": np.min(list(iou_values.values())),
    "max": np.max(list(iou_values.values())),
    "median": np.median(list(iou_values.values()))
}

print("Dice Score Statistics:")
for stat, value in dice_stats.items():
    print(f"  {stat}: {value:.4f}")

print("\nIoU Score Statistics:")
for stat, value in iou_stats.items():
    print(f"  {stat}: {value:.4f}")

# Correlation analysis
dice_list = list(dice_values.values())
iou_list = list(iou_values.values())
correlation, p_value = stats.pearsonr(dice_list, iou_list)

print(f"\nDice-IoU Correlation: {correlation:.4f} (p-value: {p_value:.4f})")
```

## 🎯 Customization Examples

### Example 1: Custom Model

```python
# models/custom_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_features, 3, padding=1),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, base_features, 3, padding=1),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(base_features, base_features, 3, padding=1),
            nn.BatchNorm3d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_features, out_channels, 1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Register in factory
from models.factory import ModelFactory
ModelFactory.MODELS["my_custom_model"] = MyCustomModel
```

### Example 2: Custom Loss Function

```python
# training/custom_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.alpha * dice + self.beta * focal

# Use in configuration
loss:
  name: "my_custom_loss"
  alpha: 0.6
  beta: 0.4
```

### Example 3: Custom Metric

```python
# training/custom_metric.py
import torch
import torch.nn as nn

class MyCustomMetric(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > self.threshold).float()
        
        # Calculate custom metric
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum() - intersection
        
        return intersection / (union + 1e-8)

# Add to metrics list
metrics:
  - name: "my_custom_metric"
    threshold: 0.5
```

## 📞 Troubleshooting Examples

### Example 1: Memory Issues

```bash
# Reduce batch size and use gradient accumulation
python main.py \
    --config configs/basic_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/memory_optimized \
    --models unet \
    --batch_size 1 \
    --accumulate_grad_batches 4
```

### Example 2: Dataset Issues

```python
# Validate dataset
from data.utils import validate_dataset

validation_results = validate_dataset("/path/to/ADAM_release_subjs")
print(f"Valid subjects: {len(validation_results['valid_subjects'])}")
print(f"Invalid subjects: {len(validation_results['invalid_subjects'])}")

for invalid in validation_results['invalid_subjects']:
    print(f"Invalid subject: {invalid}")
```

### Example 3: Model Loading Issues

```python
# Test model creation
from models.factory import ModelFactory

try:
    model = ModelFactory.create_model("unet", {
        "in_channels": 1,
        "out_channels": 1,
        "base_features": 64,
        "depth": 4
    })
    print("Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"Model creation failed: {e}")
```

These examples provide comprehensive guidance for using the medical image segmentation benchmarking framework effectively.
