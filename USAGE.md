# Medical Image Segmentation Benchmarking - Usage Guide

This guide provides comprehensive instructions for using the medical image segmentation benchmarking framework.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd benchmark_medical

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Ensure your ADAM dataset is organized as follows:
```
ADAM_release_subjs/
├── 10001/
│   ├── TOF.nii
│   ├── struct.nii
│   └── aneurysms.nii
├── 10002/
│   ├── TOF.nii
│   ├── struct.nii
│   └── aneurysms.nii
└── ...
```

### 3. Quick Benchmarking

```bash
# Run complete benchmarking pipeline
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr swin_unetr \
    --batch_size 2 \
    --max_epochs 100
```

## 📊 Available Models

### Classic Models
- **U-Net**: `unet` - CNN encoder-decoder with skip connections
- **U-Net 3D**: `unet3d` - 3D variant of U-Net
- **Attention U-Net**: `attention_unet` - U-Net with attention gates
- **nnU-Net**: `nnu_net` - Self-configuring U-Net pipeline

### Transformer Models
- **UNETR**: `unetr` - Transformer encoder + U-Net decoder
- **UNETR++**: `unetr_plus` - Enhanced UNETR with nested connections
- **SwinUNETR**: `swin_unetr` - Swin Transformer + U-Net
- **Primus**: `primus` - Pure Transformer architecture
- **Slim UNETR**: `slim_unetr` - Lightweight Transformer

### Enhanced Models
- **ES-UNet**: `es_unet` - Enhanced U-Net with attention
- **RWKV-UNet**: `rwkv_unet` - CNN + RWKV hybrid
- **Mamba-UNet**: `mamba_unet` - U-Net + State Space Models

### Multi-scale Models
- **Stacked UNet**: `stacked_unet` - Multiple U-Net layers
- **Multi-scale UNet**: `multiscale_unet` - Multi-resolution processing

## 🔧 Configuration

### Basic Configuration

Create a configuration file (`config.yaml`):

```yaml
# Dataset Configuration
dataset:
  name: "ADAM"
  data_path: "/path/to/ADAM_release_subjs"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  batch_size: 2
  patch_size: [128, 128, 128]
  overlap: 0.5

# Model Configuration
models:
  - name: "unet"
    config:
      in_channels: 1
      out_channels: 1
      base_features: 64
      depth: 4
      
  - name: "unetr"
    config:
      img_size: [128, 128, 128]
      in_channels: 1
      out_channels: 1
      embed_dim: 768
      patch_size: 16
      num_heads: 12

# Training Configuration
training:
  max_epochs: 200
  learning_rate: 1e-4
  weight_decay: 1e-5
  scheduler: "cosine"
  precision: "16-mixed"

# Loss Configuration
loss:
  name: "dice_ce"
  dice_weight: 0.5
  ce_weight: 0.5
  smooth: 1e-5

# Metrics Configuration
metrics:
  - name: "dice"
  - name: "iou"
  - name: "hausdorff_distance"
  - name: "surface_distance"
  - name: "volume_similarity"
```

### Advanced Configuration

For more control, you can customize:

```yaml
# Preprocessing
preprocessing:
  normalize_method: "z_score"  # or "min_max", "robust"
  clip_percentiles: [0.5, 99.5]
  bias_correction: true
  noise_reduction: true
  contrast_enhancement: false

# Augmentation
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

# Hardware
hardware:
  gpus: 1
  num_nodes: 1
  strategy: "auto"

# Logging
logging:
  project_name: "medical_segmentation_benchmark"
  experiment_name: "adam_benchmark"
  log_dir: "./logs"
  save_dir: "./checkpoints"
```

## 🏃‍♂️ Running Experiments

### 1. Complete Benchmarking

```bash
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr swin_unetr \
    --batch_size 2 \
    --max_epochs 200 \
    --learning_rate 1e-4 \
    --gpus 1
```

### 2. Training Only

```bash
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr \
    --max_epochs 100
```

### 3. Evaluation Only

```bash
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr \
    --eval_only \
    --save_predictions \
    --save_visualizations
```

### 4. Single Model Training

```bash
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet \
    --max_epochs 100
```

## 📈 Monitoring and Logging

### Weights & Biases Integration

```bash
# Enable W&B logging
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --wandb_project "medical_segmentation_benchmark" \
    --wandb_entity "your_entity"
```

### TensorBoard Integration

```bash
# View training progress
tensorboard --logdir ./experiments/your_experiment/logs
```

## 📊 Results and Evaluation

### Benchmark Results

After training, you'll find:

```
experiments/
├── your_experiment/
│   ├── config.yaml
│   ├── benchmark_results.json
│   ├── benchmark_report.html
│   ├── unet_best.pth
│   ├── unetr_best.pth
│   └── swin_unetr_best.pth
```

### Viewing Results

1. **HTML Report**: Open `benchmark_report.html` in your browser
2. **JSON Results**: Load `benchmark_results.json` for programmatic access
3. **Model Checkpoints**: Use `*_best.pth` files for inference

### Custom Evaluation

```python
from evaluation import BenchmarkResults, Evaluator
from models import ModelFactory

# Load benchmark results
results = BenchmarkResults.load("experiments/your_experiment/benchmark_results.json")

# Get best model for Dice score
best_model, best_score = results.get_best_model("dice")
print(f"Best model: {best_model} (Dice: {best_score:.4f})")

# Get model rankings
ranking = results.get_ranking("dice")
for i, (model, score) in enumerate(ranking):
    print(f"{i+1}. {model}: {score:.4f}")
```

## 🔍 Customization

### Adding New Models

1. Create model class in `models/` directory
2. Add to `models/__init__.py`
3. Register in `models/factory.py`

```python
# Example: Custom model
class MyCustomModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__()
        # Your model implementation
        
    def forward(self, x):
        # Your forward pass
        return x

# Register in factory
ModelFactory.MODELS["my_custom_model"] = MyCustomModel
```

### Custom Loss Functions

```python
# Example: Custom loss
class MyCustomLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, inputs, targets):
        # Your loss implementation
        return loss

# Use in configuration
loss:
  name: "my_custom_loss"
  weight: 1.0
```

### Custom Metrics

```python
# Example: Custom metric
class MyCustomMetric(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets):
        # Your metric implementation
        return metric_value

# Add to metrics list
metrics:
  - name: "my_custom_metric"
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --batch_size 1
   
   # Use gradient accumulation
   --accumulate_grad_batches 4
   ```

2. **Dataset Loading Issues**
   ```bash
   # Check dataset structure
   python data/utils.py --data_path /path/to/dataset --validate
   ```

3. **Model Loading Issues**
   ```bash
   # Check model configuration
   python models/factory.py --model_name unet --validate_config
   ```

### Performance Optimization

1. **Memory Optimization**
   - Use smaller patch sizes
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Speed Optimization**
   - Increase batch size if memory allows
   - Use multiple GPUs
   - Enable data loading optimization

## 📚 Examples

### Example 1: Basic Benchmarking

```bash
# Run basic benchmarking with default settings
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/basic_benchmark
```

### Example 2: Advanced Configuration

```bash
# Run with custom configuration
python main.py \
    --config configs/advanced_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/advanced_benchmark \
    --models unet unetr swin_unetr primus \
    --batch_size 4 \
    --max_epochs 300 \
    --learning_rate 5e-5 \
    --gpus 2
```

### Example 3: Evaluation Only

```bash
# Evaluate pre-trained models
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments/evaluation \
    --models unet unetr \
    --eval_only \
    --save_predictions \
    --save_visualizations
```

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the configuration examples
3. Check the logs in your experiment directory
4. Create an issue on the repository

## 🎯 Best Practices

1. **Start Small**: Begin with a single model and small dataset
2. **Monitor Training**: Use W&B or TensorBoard for monitoring
3. **Validate Configuration**: Test your configuration before long runs
4. **Save Checkpoints**: Always save model checkpoints
5. **Document Experiments**: Keep track of your experiments and results
