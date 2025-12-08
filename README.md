# 🏥 Medical Image Segmentation Benchmarking Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/your-username/medical-segmentation-benchmark/workflows/CI/badge.svg)](https://github.com/your-username/medical-segmentation-benchmark/actions)
[![Documentation](https://readthedocs.org/projects/medical-segmentation-benchmark/badge/?version=latest)](https://medical-segmentation-benchmark.readthedocs.io/)

> **A comprehensive research-grade framework for benchmarking state-of-the-art medical image segmentation models on the ADAM (Aneurysm Detection And segMentation) dataset.**

## 🎯 Overview

This framework provides a systematic evaluation of multiple segmentation architectures for intracranial aneurysm detection and segmentation from Time-of-Flight Magnetic Resonance Angiographs (TOF-MRAs). It implements cutting-edge models from classic U-Net architectures to advanced transformer-based approaches, providing researchers with a unified platform for comparative analysis.

### ✨ Key Features

- 🧠 **Comprehensive Model Library**: 15+ state-of-the-art segmentation models
- 📊 **Advanced Evaluation**: 10+ medical imaging metrics with statistical analysis
- 🔬 **Research-Grade Quality**: Reproducible experiments with proper documentation
- ⚡ **High Performance**: Multi-GPU support, mixed precision training, memory optimization
- 📈 **Experiment Tracking**: Weights & Biases, TensorBoard, and custom logging
- 🐳 **Docker Support**: Containerized deployment for easy reproducibility
- 📚 **Extensive Documentation**: Tutorials, examples, and API documentation

## 📊 Supported Models

### 🏛️ Classic Models
- **U-Net (2015)**: CNN encoder-decoder with skip connections
- **U-Net 3D**: 3D variant for volumetric medical images
- **Attention U-Net**: U-Net with attention gates for improved focus
- **nnU-Net (2018/2021)**: Self-configuring U-Net pipeline

### 🤖 Transformer Models
- **UNETR (2021)**: Transformer encoder + U-Net decoder
- **UNETR++ (2022/2023)**: Enhanced UNETR with nested connections
- **SwinUNETR**: Swin Transformer + U-Net architecture
- **Primus**: Pure Transformer architecture
- **Slim UNETR++**: Lightweight Transformer variant

### 🚀 Next-Generation Models (2023-2025)
- **ES-UNet**: Enhanced U-Net with attention mechanisms
- **RWKV-UNet**: CNN + RWKV hybrid architecture
- **Mamba-UNet**: U-Net + State Space Models
- **Stacked UNet**: Multi-layer U-Net architecture
- **Multi-scale UNet**: Multi-resolution processing

## 🏗️ Project Structure

```
medical-segmentation-benchmark/
├── 📁 data/                    # Data loading and preprocessing
│   ├── dataset.py             # ADAM dataset implementation
│   ├── preprocessing.py       # Medical image preprocessing
│   ├── augmentation.py        # Data augmentation pipeline
│   └── utils.py               # Data utilities
├── 📁 models/                  # Model implementations
│   ├── unet.py                # U-Net variants
│   ├── unetr.py               # UNETR implementations
│   ├── swin_unetr.py          # SwinUNETR
│   ├── transformer_models.py  # Advanced transformers
│   ├── enhanced_models.py     # Next-gen models
│   └── factory.py             # Model factory
├── 📁 training/                # Training framework
│   ├── trainer.py             # Training loop
│   ├── losses.py              # Loss functions
│   ├── metrics.py             # Evaluation metrics
│   └── callbacks.py           # Training callbacks
├── 📁 evaluation/              # Evaluation system
│   ├── evaluator.py           # Model evaluation
│   ├── benchmark.py           # Benchmarking tools
│   └── visualization.py       # Result visualization
├── 📁 configs/                 # Configuration files
├── 📁 utils/                   # Utility functions
├── 📁 tests/                   # Test suite
├── 📁 docs/                    # Documentation
└── 📁 examples/                # Example scripts
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/medical-segmentation-benchmark.git
cd medical-segmentation-benchmark

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
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
│   └── ...
```

### 3. Run Benchmarking

```bash
# Quick start with default settings
python run_benchmark.py \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr swin_unetr

# Advanced configuration
python main.py \
    --config configs/benchmark_config.yaml \
    --data_path /path/to/ADAM_release_subjs \
    --output_dir ./experiments \
    --models unet unetr swin_unetr primus \
    --batch_size 2 \
    --max_epochs 100 \
    --gpus 1
```

### 4. View Results

- **HTML Report**: Open `experiments/your_experiment/benchmark_report.html`
- **JSON Results**: Load `experiments/your_experiment/benchmark_results.json`
- **Model Checkpoints**: Use `experiments/your_experiment/*_best.pth`

## 📈 Evaluation Metrics

### 🎯 Segmentation Metrics
- **Dice Score**: Overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Standard segmentation metric
- **Hausdorff Distance**: Maximum distance between boundaries
- **Surface Distance**: Average distance between surfaces
- **Volume Similarity**: Relative volume difference

### 📊 Statistical Analysis
- **Confidence Intervals**: Statistical significance testing
- **Model Rankings**: Performance comparison across models
- **Correlation Analysis**: Relationship between different metrics
- **Performance Distributions**: Statistical characterization

## 🔬 Research Applications

### 🏥 Clinical Applications
- **Intracranial Aneurysm Detection**: Primary application
- **Brain Tumor Segmentation**: Extended applications
- **Vascular Analysis**: Related medical imaging tasks
- **Radiology Workflows**: Clinical decision support

### 🧪 Research Areas
- **Medical Image Analysis**: Core research area
- **Deep Learning**: Methodological contributions
- **Computer Vision**: Technical advancements
- **Biomedical Engineering**: Applied research

## 📚 Documentation

- 📖 **[Usage Guide](USAGE.md)**: Comprehensive usage instructions
- 🎯 **[Examples](EXAMPLES.md)**: Detailed examples and tutorials
- 🔧 **[API Documentation](https://medical-segmentation-benchmark.readthedocs.io/)**: Complete API reference
- 🤝 **[Contributing](CONTRIBUTING.md)**: How to contribute to the project
- 📄 **[Citation](CITATION.md)**: How to cite this work

## 🐳 Docker Support

```bash
# Build and run with Docker
docker build -t medical-segmentation-benchmark .
docker run -it --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/experiments:/app/experiments \
    medical-segmentation-benchmark

# Use Docker Compose
docker-compose up benchmark
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/ -v -m "not slow"  # Fast tests only
pytest tests/ -v -m "gpu"       # GPU tests only

# Run with coverage
make test-coverage
```

## 📊 Performance

### ⚡ Training Performance
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Mixed Precision**: 16-bit training for faster convergence
- **Memory Optimization**: Efficient data loading and processing
- **Gradient Accumulation**: Handle large batch sizes with limited memory

### 🎯 Model Performance
- **State-of-the-Art Results**: Competitive performance on ADAM dataset
- **Comprehensive Evaluation**: Multiple metrics for thorough assessment
- **Statistical Analysis**: Confidence intervals and significance testing
- **Reproducible Results**: Fixed seeds and deterministic training

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🎯 Areas for Contribution
- **New Models**: Implement latest state-of-the-art architectures
- **Advanced Metrics**: Add specialized medical imaging metrics
- **Performance Optimization**: Improve training and inference speed
- **Documentation**: Enhance tutorials and examples

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{medical_segmentation_benchmark,
  title={Medical Image Segmentation Benchmarking Framework: A Comprehensive Evaluation of State-of-the-Art Models for Intracranial Aneurysm Detection and Segmentation},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/your-username/medical-segmentation-benchmark},
  doi={10.5281/zenodo.XXXXXXX},
  license={MIT}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ADAM Challenge Organizers** for providing the dataset
- **PyTorch Team** for the excellent deep learning framework
- **Medical Imaging Community** for validation and feedback
- **Open Source Contributors** for their valuable contributions

## 📞 Support

- 📧 **Email**: [support@domain.com](mailto:support@domain.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/medical-segmentation-benchmark/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/medical-segmentation-benchmark/discussions)
- 📚 **Documentation**: [Read the Docs](https://medical-segmentation-benchmark.readthedocs.io/)

---

**⭐ Star this repository if you find it useful!**

*Last updated: January 2024*
