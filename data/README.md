# 🗂️ Dataset Directory Structure

This directory contains the expected structure for medical imaging datasets used in the benchmark. **Actual dataset files are not included in the repository** due to size and privacy constraints.

## 📁 Expected Structure

```
data/
├── adam_dataset/
│   ├── raw/
│   │   ├── README.md          # Dataset information
│   │   ├── 10025/
│   │   │   ├── aneurysms.nii.gz      # Ground truth masks
│   │   │   ├── orig/
│   │   │   │   ├── TOF.nii.gz        # Time-of-Flight MRA
│   │   │   │   └── struct.nii.gz     # Structural MRI (optional)
│   │   │   └── pre/
│   │   │       ├── TOF.nii.gz        # Preprocessed TOF
│   │   │       └── struct.nii.gz     # Preprocessed structural
│   │   ├── 10026/ ... 10030/
│   │   └── [additional subjects]/
│   └── processed/
│       └── README.md          # Processed data information
├── brats_dataset/             # Future: Brain tumor segmentation
└── other_datasets/            # Future: Additional medical datasets
```

## 📋 Dataset Requirements

### ADAM Dataset (Aneurysm Detection and Morphology)

**Download:** [ADAM Challenge Dataset](https://adam.isi.uu.nl/)

**Required files per subject:**
- `aneurysms.nii.gz` - Binary segmentation masks (ground truth)
- `orig/TOF.nii.gz` - Time-of-Flight MR Angiography (primary input)
- `orig/struct.nii.gz` - Structural MRI (optional, for multi-modal)

**Subject IDs:** 10025, 10026, 10027, 10028, 10029, 10030, etc.

### 🚀 Quick Setup

1. **Download ADAM dataset** from the official challenge website
2. **Extract** the dataset to `data/adam_dataset/raw/`
3. **Verify** structure matches the expected format above
4. **Run** the benchmark:

```bash
# Test dataset structure
python unified_benchmark.py --data_path data/adam_dataset/raw --quick_test

# Full benchmark
python unified_benchmark.py --data_path data/adam_dataset/raw --all_models
```

## ⚠️ Important Notes

- **Dataset files are excluded from Git** (see `.gitignore`)
- **Maintain folder structure** but place actual `.nii.gz` files yourself
- **Privacy compliance:** Ensure you have proper permissions for dataset usage
- **File sizes:** Medical volumes can be 100MB+ each
- **Format:** NIfTI format (`.nii.gz`) is expected for all medical images

## 🔧 Troubleshooting

**Missing dataset files?**
```bash
# Check structure
find data/adam_dataset/raw -name "*.nii.gz" | head -10

# Verify subjects
ls -la data/adam_dataset/raw/
```

**Permission issues?**
```bash
chmod -R 755 data/adam_dataset/
```

**Need help?** Check `UNIFIED_USAGE.md` for complete setup instructions.
