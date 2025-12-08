# 📊 Benchmark Results

This directory contains example benchmark results demonstrating the complete medical image segmentation evolution analysis.

## 📁 Current Files

- **`benchmark_results.json`** - Detailed results for all 15 models (example output)
- **`benchmark_summary.txt`** - Human-readable performance summary (example output)  
- **`README.md`** - This documentation file

## 🏆 Example Results Overview

**Best Performing Models** (from example run):

| Rank | Model | Architecture Era | Dice Score | Time |
|------|-------|------------------|------------|------|
| 🥇 | **UNet3D** | Classic CNN (2015) | **0.3863** | 2.1min |
| 🥈 | **Mamba-UNet** | Next-Gen (2025) | **0.2461** | 1.3min |
| 🥉 | **Primus** | Pure Transformer (2023) | **0.1541** | 1.3min |

## 🔄 Your Results

When you run the benchmark, **new result files will be generated** and will replace these examples:

```bash
# Run benchmark - generates new results
python unified_benchmark.py --data_path data/adam_dataset/raw --all_models

# Results saved to:
# - benchmark_results.json    (detailed JSON)
# - benchmark_summary.txt     (summary table)
```

## 📈 Result File Structure

### JSON Results (`benchmark_results.json`)
```json
{
  "experiment_info": {
    "start_time": "2025-10-27T18:29:45",
    "total_time_minutes": 23.899,
    "models_completed": 14
  },
  "dataset_info": {
    "total_subjects": 6,
    "subjects_with_aneurysms": 6
  },
  "model_results": {
    "unet3d": {
      "dice_score": 0.3863,
      "training_time_minutes": 2.1,
      "status": "completed"
    }
  }
}
```

### Summary Table (`benchmark_summary.txt`)
```
Medical Image Segmentation Benchmark Results
=================================================

Model Results:
------------------------------
unet3d               | Dice: 0.3863 | Time: 2.1min
mamba_unet           | Dice: 0.2461 | Time: 1.3min
...

Best Model: unet3d
Best Dice Score: 0.3863
Models Completed: 14/15
```

## ⚠️ Important Notes

- **Example results:** Current files are examples from development system
- **Your results may vary** based on dataset, hardware, and random initialization
- **Reproducibility:** Set `--seed 42` for more consistent results
- **Performance depends on:** GPU type, memory, dataset size

## 🎯 Expected Performance Ranges

Based on ADAM aneurysm dataset:

- **Excellent (>0.3):** UNet3D, advanced 3D CNNs
- **Good (0.1-0.3):** Transformer hybrids, next-gen models  
- **Learning (0.05-0.1):** Basic transformers, attention models
- **Poor (<0.05):** 2D methods, misconfigured models

## 📜 Citation

If you use these benchmark results in research:

```bibtex
@software{medical_segmentation_benchmark,
  title={Unified Medical Image Segmentation Benchmark: Complete Evolution Analysis (2015-2025)},
  author={Medical AI Research Team}, 
  year={2025},
  url={https://github.com/ansulx/Medical-Segmentation-Benchmark}
}
```
