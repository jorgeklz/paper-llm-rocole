"""
prepare_rocole.py
=================

Builds the train/val/test splits for the two tasks of the study from the
original RoCoLe dataset.

Input (expected at ./rocole/):
    rocole/Photos/*.jpg                          (1560 images)
    rocole/Annotations/RoCoLe-classes.xlsx       (columns: File, Binary.Label,
                                                  Multiclass.Label)

Output (created at ./data/splits/):
    data/splits/task_a_binary/{train,val,test}/{healthy,unhealthy}/
    data/splits/task_b_3class/{train,val,test}/{healthy,red_spider_mite,coffee_leaf_rust}/

Task A (binary):   healthy vs unhealthy (all non-healthy aggregated).
Task B (3-class):  healthy / red_spider_mite / coffee_leaf_rust
                   (rust_level_1..4 aggregated into coffee_leaf_rust).

Stratified 70/15/15 splits with a fixed random seed for reproducibility.

Usage:
    python3 data_preparation/prepare_rocole.py
"""

import os
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] pandas required: pip install pandas openpyxl")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
SRC = Path("rocole")
PHOTOS = SRC / "Photos"
ANNOT_XLSX = SRC / "Annotations" / "RoCoLe-classes.xlsx"
OUT = Path("data/splits")
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = 0.15


# ----------------------------------------------------------------------
# Label normalization
# ----------------------------------------------------------------------
def normalize_multiclass(label: str) -> str:
    """Map the original multi-class labels to the 3 classes of Task B."""
    s = str(label).strip().lower()
    if s in ("healthy", "health", "sano"):
        return "healthy"
    if "spider" in s or "mite" in s:
        return "red_spider_mite"
    # rust_level_1..4 and any other rust variant
    return "coffee_leaf_rust"


def normalize_binary(label: str) -> str:
    s = str(label).strip().lower()
    if s in ("healthy", "health", "sano", "0"):
        return "healthy"
    return "unhealthy"


# ----------------------------------------------------------------------
# Stratified split
# ----------------------------------------------------------------------
def stratified_split(files_by_class, seed=SEED):
    """Return a dict split_name -> list[(filename, class)] stratified by class."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for cls, files in sorted(files_by_class.items()):
        files = sorted(files)
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        for f in files[:n_train]:
            splits["train"].append((f, cls))
        for f in files[n_train:n_train + n_val]:
            splits["val"].append((f, cls))
        for f in files[n_train + n_val:]:
            splits["test"].append((f, cls))
    return splits


def resolve_photo_path(fname: str) -> Path:
    """Locate the image file, accepting common extension variants."""
    p = PHOTOS / fname
    if p.exists():
        return p
    stem = Path(fname).stem
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        candidate = PHOTOS / (stem + ext)
        if candidate.exists():
            return candidate
    return p  # returns a non-existent path; reported by the caller


# ----------------------------------------------------------------------
# Build one task
# ----------------------------------------------------------------------
def build_task(task_name: str, label_col: str, normalizer, df: pd.DataFrame) -> None:
    print(f"\n=== Building {task_name} ===")
    files_by_class = defaultdict(list)
    missing = 0
    for _, row in df.iterrows():
        fname = str(row["File"]).strip()
        cls = normalizer(row[label_col])
        if not resolve_photo_path(fname).exists():
            missing += 1
            continue
        files_by_class[cls].append(fname)

    print("Counts per class (raw):")
    for cls in sorted(files_by_class):
        print(f"  {cls:25s}: {len(files_by_class[cls])}")
    if missing:
        print(f"  [WARN] {missing} referenced files not found under {PHOTOS}")

    splits = stratified_split(files_by_class)

    for split_name, items in splits.items():
        counts = Counter(cls for _, cls in items)
        print(f"  {split_name:6s}: {len(items)} images -> {dict(counts)}")
        for fname, cls in items:
            dst_dir = OUT / task_name / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            src = resolve_photo_path(fname)
            dst = dst_dir / src.name
            shutil.copy2(src, dst)

    print(f"  TOTAL copied: {sum(len(v) for v in splits.values())}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    if not PHOTOS.is_dir():
        sys.exit(f"[ERROR] {PHOTOS} not found. Place RoCoLe at ./rocole/Photos/")
    if not ANNOT_XLSX.is_file():
        sys.exit(f"[ERROR] {ANNOT_XLSX} not found.")

    n_photos = (len(list(PHOTOS.glob("*.jpg"))) +
                len(list(PHOTOS.glob("*.JPG"))))
    print(f"[INFO] Images found in {PHOTOS}: {n_photos}")

    df = pd.read_excel(ANNOT_XLSX)
    print(f"[INFO] Rows in annotations: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Tolerant column-name detection
    cols = {c.lower().strip(): c for c in df.columns}
    file_col = cols.get("file") or cols.get("filename") or cols.get("image")
    bin_col = (cols.get("binary.label") or cols.get("binary_label")
               or cols.get("binary"))
    multi_col = (cols.get("multiclass.label") or cols.get("multiclass_label")
                 or cols.get("multiclass") or cols.get("class"))

    if not all([file_col, bin_col, multi_col]):
        sys.exit(f"[ERROR] Expected columns not detected. Found: {list(df.columns)}")

    df = df.rename(columns={file_col: "File"})

    if OUT.exists():
        print(f"[INFO] Removing previous {OUT}...")
        shutil.rmtree(OUT)

    build_task("task_a_binary", bin_col, normalize_binary, df)
    build_task("task_b_3class", multi_col, normalize_multiclass, df)

    print(f"\n[OK] Splits generated at {OUT.resolve()}")


if __name__ == "__main__":
    main()
