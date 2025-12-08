# 🏥 Unified Medical Image Segmentation Benchmark

## Single Script for Complete Model Evolution Analysis

This unified script replaces all separate scripts (`main.py`, `main_optimized.py`, `quick_start.py`, `run_benchmark.py`) with a single comprehensive solution.

### 🚀 Quick Start

```bash
# Basic usage
python unified_benchmark.py --data_path /path/to/adam/dataset

# Quick test (2 models, minimal settings)
python unified_benchmark.py --data_path /path/to/adam/dataset --quick_test

# All 15 models (may timeout on slower systems)
python unified_benchmark.py --data_path /path/to/adam/dataset --all_models

# Custom selection
python unified_benchmark.py --data_path /path/to/adam/dataset --models unet unetr primus mamba_unet

# Extended time limit
python unified_benchmark.py --data_path /path/to/adam/dataset --max_time_minutes 45
```

### 📊 Implemented Models (15 Total)

**Classic Models (2015-2021):**
- `unet` - Original U-Net (2015) 
- `unet3d` - 3D U-Net variant
- `lightweight_unet3d` - Memory-optimized U-Net
- `attention_unet` - U-Net with attention gates
- `nnu_net` - Self-configuring U-Net (2018)

**Transformer Models (2021-2023):**
- `unetr` - UNETR (2021) - Vision Transformer + U-Net decoder
- `unetr_plus` - UNETR++ (2022) - Enhanced with nested connections
- `swin_unetr` - SwinUNETR (2022) - Swin Transformer + U-Net
- `primus` - Pure Transformer (2023) - No CNN components
- `slim_unetr` - Lightweight Transformer (2023)

**Next-Generation Models (2023-2025):**
- `es_unet` - Enhanced U-Net with attention + deep supervision
- `rwkv_unet` - CNN + RWKV hybrid for long-range modeling
- `mamba_unet` - U-Net + State Space Models (Mamba)
- `stacked_unet` - Multiple stacked U-Nets for rich context
- `multiscale_unet` - Multi-scale processing for various resolutions

### 🧠 3D Medical Image Support

**ADAM Dataset Structure:**
```
/path/to/adam/dataset/
├── 10025/
│   ├── aneurysms.nii.gz    # Ground truth mask
│   ├── orig/
│   │   ├── TOF.nii.gz      # Time-of-Flight MRA
│   │   └── struct.nii.gz   # Structural MRI
│   └── pre/
│       ├── TOF.nii.gz      # Preprocessed TOF
│       └── struct.nii.gz   # Preprocessed struct
├── 10026/
└── ...
```

**Automatic Processing:**
- ✅ 3D NIfTI file loading with error handling
- ✅ Z-score normalization for medical images  
- ✅ Automatic patch extraction (32³ for timeout prevention)
- ✅ Binary mask processing for aneurysm segmentation
- ✅ Multi-input support (TOF + structural images)

### ⏱️ Timeout Prevention Features

**Aggressive Timeout Management:**
- **Total runtime:** 25 minutes maximum
- **Per model:** 5 minutes maximum  
- **Per batch:** 10 seconds maximum
- **Per dataset item:** 5 seconds maximum

**Memory Management:**
- Memory usage limited to 60% of available RAM
- Automatic cleanup every 3 batches
- Force cleanup on errors
- GPU memory management with `torch.cuda.empty_cache()`

**Training Optimizations:**
- Max 10 batches per epoch for speed
- Max 5 validation batches
- Early stopping with patience=2
- Reduced patch sizes (32³ instead of 128³)
- Simplified model architectures

### 🔧 Features

**Dataset Validation:**
- Automatic ADAM dataset structure validation
- Subject file availability checking
- Aneurysm vs. healthy subject detection
- Missing file reporting

**Progress Tracking:**
- Rich console output with progress bars
- Real-time memory usage monitoring
- Remaining time estimation
- Model completion status

**Error Recovery:**
- Comprehensive exception handling
- Timeout recovery for individual models
- Memory cleanup on errors  
- Detailed error logging

**Results Export:**
- JSON results with full metrics
- Summary report in text format
- Progress logs with timestamps
- Model comparison tables

### 📈 Output

**Metrics Calculated:**
- Dice Score (primary metric)
- IoU (Intersection over Union)
- Accuracy
- Training time per model

**Files Generated:**
```
benchmark_results/
├── benchmark_results.json    # Complete results
├── benchmark_summary.txt     # Human-readable summary
└── logs/                     # Detailed logs
```

### 🎯 Usage Examples

**Research Comparison:**
```bash
# Compare all transformer models
python unified_benchmark.py \
  --data_path /data/adam \
  --models unetr unetr_plus swin_unetr primus slim_unetr \
  --max_time_minutes 30
```

**Evolution Analysis:**
```bash  
# Run complete evolution sequence
python unified_benchmark.py \
  --data_path /data/adam \
  --all_models \
  --max_time_minutes 60 \
  --output_dir ./evolution_results
```

**Quick Validation:**
```bash
# Fast validation run
python unified_benchmark.py \
  --data_path /data/adam \
  --quick_test \
  --max_subjects 3
```

### 🔧 System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space
- CPU execution supported

**Recommended:**
- Python 3.9+  
- 16GB RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB disk space

**Installation:**
```bash
pip install -r requirements_unified.txt
```

### 🚨 Troubleshooting

**Common Issues:**

1. **Timeout Errors:**
   - Use `--quick_test` for faster execution
   - Reduce `--max_subjects` 
   - Increase `--max_time_minutes`

2. **Memory Issues:**
   - Close other applications
   - Use smaller batch sizes
   - Enable swap if available

3. **Dataset Errors:**
   - Check ADAM dataset structure
   - Verify NIfTI file permissions
   - Ensure sufficient disk space

4. **Model Failures:**
   - Check CUDA availability for GPU
   - Verify PyTorch installation
   - Try CPU-only execution

### 📞 Support

This unified script replaces all previous separate scripts and provides:
- ✅ Complete model evolution sequence
- ✅ 3D medical image compatibility  
- ✅ Timeout-proof execution
- ✅ Memory management
- ✅ Error recovery
- ✅ Progress tracking
- ✅ Comprehensive results

Perfect for research, benchmarking, and supervisor demonstrations! 🎓
