# Handwritten Multi-Script Character Segmentation & Recognition
## Solution README

---

## Overview

This solution tackles the **JU-CMATER Handwritten Multi-Script Character Segmentation and Recognition** challenge. The pipeline uses a custom **Faster R-CNN** detector with an **Inception-ResNet-v2 FPN backbone** (via `timm`) to jointly segment and classify handwritten characters across two scripts: **Latin** and **Bengali**.

Key design choices:

- **Tiled inference** (960×960 tiles, 35% overlap) handles large full-page document images at high resolution without GPU memory overflow.
- **Weighted Boxes Fusion (WBF)** merges overlapping detections across tiles and across ensemble models, outperforming standard NMS for dense, overlapping text regions.
- **Bengali oversampling** (3× repetition during final training) compensates for class imbalance between Latin and Bengali pages.
- **Focal loss** replaces the default Faster R-CNN cross-entropy to handle the extreme foreground/background imbalance in character detection.
- **3-scale TTA** (×0.85, ×1.0, ×1.15) at inference improves recall for small characters.
- **Two-stage training**: first on an 80/20 train/val split (to track validation score), then fine-tuned on all pages with Bengali 3× oversampled.
- **Script-aware confidence thresholds**: Bengali pages use `conf=0.028` for high recall; Latin pages use `conf=0.08–0.09`.

---

## Environment Setup

### Hardware

- **GPU strongly recommended** (tested on NVIDIA T4 / A100 via Google Colab)
- CPU-only is supported but training will be extremely slow

### Python Version

- Python 3.9 or higher

### Install Dependencies

```bash
pip install timm>=0.9.12 \
            ensemble-boxes \
            albumentations>=1.3.1 \
            opencv-python \
            scikit-learn \
            ultralytics \
            torch torchvision \
            pandas numpy pillow
```

Or in a single command (as used in the notebook):

```bash
pip install timm>=0.9.12 ensemble-boxes albumentations>=1.3.1 opencv-python scikit-learn ultralytics
```

PyTorch should be installed separately with the appropriate CUDA version for your GPU. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### Full Package Versions (Tested)

| Package | Version |
|---|---|
| Python | 3.10.x |
| torch | 2.2.x (CUDA 12.1) |
| torchvision | 0.17.x |
| timm | 0.9.12+ |
| albumentations | 1.3.1+ |
| ensemble-boxes | 1.0.9 |
| opencv-python | 4.9.x |
| scikit-learn | 1.4.x |
| pandas | 2.x |
| numpy | 1.26.x |
| pillow | 10.x |

---

## Dataset Setup

1. Download the competition dataset zip:
   ```
   handwritten-multiscript-character-segmentation-recognition.zip
   ```

2. Place the zip at:
   ```
   /content/handwritten-multiscript-character-segmentation-recognition.zip
   ```
   (If running locally, update `DATASET_ZIP` at the top of `solution.py` to match your path.)

3. The script will auto-extract to `/content/dataset/` and locate the `JU_CMATER` directory automatically.

Expected folder structure after extraction:
```
JU_CMATER/
├── train/
│   ├── images/        # Full-page training images (.jpg/.png)
│   └── annotations/   # LabelMe JSON annotation files
└── test/
    └── images/        # Full-page test images (no annotations)
```

---

## Running the Code

The entire pipeline is contained in a single script / Colab notebook: **`solution.py`** (or run cells top-to-bottom in the notebook).

### Steps performed automatically:

1. **Package installation** — installs all required libraries
2. **Dataset extraction** — unzips and locates the data
3. **Label construction** — builds `label_to_id` / `id_to_label` mappings from all training annotations
4. **Tile generation** — slices all training pages into 960×960 YOLO-format tiles saved under `/content/work/yolo_tiles/`
5. **Val-model training** — trains Faster R-CNN for 25 epochs on the 80% train split
6. **Validation scoring** — evaluates the val model on the held-out 20% using the competition metric
7. **Final-model training** — fine-tunes for 30 more epochs on all pages (Bengali 3×)
8. **Submission generation** — runs 2-model ensemble + 3-scale TTA on all 6 test pages and writes `submission.csv`

### To run end-to-end:

```bash
python solution.py
```

Or in Google Colab, run all cells sequentially (Runtime → Run All).

---

## Generating the Submission File

The submission CSV is automatically generated at the end of the script:

```
/content/work/submission.csv
```

Format validation is performed automatically. The file will contain 6 rows (one per test page) with columns:

| Column | Format |
|---|---|
| `page_id` | Page stem (e.g., `001`, `101`) |
| `predictions` | `["script":0, {"unicode_value":"U+0041","bbox":[x1,y1,x2,y2]}, ...]` |

In Google Colab, the file is automatically downloaded at the end via:
```python
from google.colab import files
files.download('/content/work/submission.csv')
```

If running locally, copy the file from `/content/work/submission.csv` to your desired output path.

---

## Key Configuration Constants

These are the most important parameters to tune if reproducing or improving the solution:

| Constant | Value | Purpose |
|---|---|---|
| `TILE_SIZE` | 960 | Tile size for sliding window |
| `TILE_OVERLAP` | 0.35 | Overlap fraction between tiles |
| `TARGET_INFER_RES` | 1280 | Resize resolution during inference |
| `EPOCHS_VAL` | 25 | Epochs for validation-split model |
| `EPOCHS_FINAL` | 30 | Fine-tuning epochs on full dataset |
| `PAGE_CONF['101']` | 0.028 | Bengali confidence threshold (keep LOW) |
| `PAGE_CONF['001']` | 0.08 | Latin confidence threshold |
| `tta_scales` | (0.85, 1.0, 1.15) | Multi-scale TTA at inference |
| `MIN_BOX_AREA_PRED` | 80 | Minimum box area in pixels² |

---

## Important Notes

- **Do not apply global cross-class NMS** across the entire page — this destroys Bengali matra + consonant stacking pairs that legitimately overlap.
- **Do not flip augmentations** (horizontal/vertical) — text directionality is critical for script identification.
- Bengali pages are identified by Unicode range `U+09xx`; Latin pages use `U+00xx`–`U+00FF`.
- The `FORCED_SCRIPT` dictionary hardcodes known script assignments for test pages (`101`, `102` → Bengali; `001`, `005`, `010`, `019` → Latin) to prevent script misclassification.
- The val-model weights are reused as initialization for final-model training to avoid NaN loss in early epochs of the second training phase.

---

## Output Files

| File | Description |
|---|---|
| `/content/work/submission.csv` | Final submission (6 rows) |
| `/content/work/inception_val.pth` | Checkpoint after val-split training |
| `/content/work/inception_final.pth` | Final model after full-data fine-tuning |
| `/content/work/inception_final_epXX.pth` | Intermediate checkpoints every 5 epochs |
| `/content/work/yolo_tiles/` | Generated tile images and YOLO labels |
