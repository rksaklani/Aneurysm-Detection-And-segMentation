# from pathlib import Path
# import shutil

# ADAM_ROOT = Path(r"data\adam_dataset\raw")
# OUTPUT_ROOT = Path(r"data\MRA")

# IMG_DIR = OUTPUT_ROOT / "images"
# MASK_DIR = OUTPUT_ROOT / "masks"

# IMG_DIR.mkdir(parents=True, exist_ok=True)
# MASK_DIR.mkdir(parents=True, exist_ok=True)

# idx = 1

# for subject_dir in sorted(ADAM_ROOT.iterdir()):
#     if not subject_dir.is_dir():
#         continue

#     case_dir = subject_dir / subject_dir.name

#     img_path = case_dir / "orig" / "TOF.nii.gz"
#     mask_path = case_dir / "aneurysms.nii.gz"

#     if not img_path.exists() or not mask_path.exists():
#         continue

#     img_name = f"img_{idx:04d}.nii.gz"
#     mask_name = f"mask_{idx:04d}.nii.gz"

#     shutil.copy(img_path, IMG_DIR / img_name)
#     shutil.copy(mask_path, MASK_DIR / mask_name)

#     idx += 1

# print(f"Reorganization complete. Total samples copied: {idx - 1}")
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np

# ================= CONFIG =================
ADAM_ROOT = Path(r"data\adam_dataset\raw")
OUTPUT_ROOT = Path(r"data\MRA")

IMG_DIR = OUTPUT_ROOT / "images"
MASK_DIR = OUTPUT_ROOT / "masks"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

# ================= PROCESS =================
idx = 1
skipped_empty = 0
skipped_missing = 0

for subject_dir in sorted(ADAM_ROOT.iterdir()):
    if not subject_dir.is_dir():
        continue

    case_dir = subject_dir / subject_dir.name

    img_path = case_dir / "orig" / "TOF.nii.gz"
    mask_path = case_dir / "aneurysms.nii.gz"

    # ---- existence check ----
    if not img_path.exists() or not mask_path.exists():
        skipped_missing += 1
        continue

    # ---- empty mask check ----
    mask_data = nib.load(mask_path).get_fdata()
    if np.max(mask_data) == 0:
        skipped_empty += 1
        continue

    # ---- copy valid case ----
    img_name = f"img_{idx:04d}.nii.gz"
    mask_name = f"mask_{idx:04d}.nii.gz"

    shutil.copy(img_path, IMG_DIR / img_name)
    shutil.copy(mask_path, MASK_DIR / mask_name)

    idx += 1

# ================= REPORT =================
print("✅ Reorganization complete")
print(f"Total valid samples copied : {idx - 1}")
print(f"Skipped (empty masks)      : {skipped_empty}")
print(f"Skipped (missing files)   : {skipped_missing}")
