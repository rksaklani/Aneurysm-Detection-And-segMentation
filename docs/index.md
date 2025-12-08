# Medical Image Segmentation Benchmarking Framework

Welcome to the Medical Image Segmentation Benchmarking Framework documentation.

## Quick Start

See the [README](../README.md) for installation and usage instructions.

## Documentation

- [Usage Guide](../USAGE.md)
- [Unified Usage](../UNIFIED_USAGE.md)
- [Examples](../EXAMPLES.md)
- [Contributing](../CONTRIBUTING.md)
- [Citation](../CITATION.md)

## Features

- ✅ All 15 models from evolution sequence (2015-2025)
- 🧠 3D medical image support (ADAM dataset)
- ⏱️ Comprehensive timeout prevention
- 🔧 Memory management & error recovery
- 📊 Automatic progress tracking
- 🎨 Rich visualizations and reports

## Single Command Usage

```bash
# Quick test (2 models, ~5 minutes)
python unified_benchmark.py --data_path data/adam_dataset/raw --quick_test

# Full benchmark (All 15 models, ~60 minutes)
python unified_benchmark.py --data_path data/adam_dataset/raw --all_models --max_time_minutes 60
```

## Results

The framework generates comprehensive results including:
- Model performance comparison charts
- Evolution timeline visualizations
- Training curves and metrics
- JSON results and HTML reports
