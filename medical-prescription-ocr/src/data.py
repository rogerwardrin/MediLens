import glob
import os
import pandas as pd
from PIL import Image

# Configurable data root: defaults to ~/ocr_data per plan.md
DATA_ROOT = os.getenv("OCR_DATA_ROOT", os.path.expanduser("~/ocr_data"))
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def build_image_index(root):
    """Map file name -> full path for every image under `root` recursively."""
    return {
        os.path.basename(p): p
        for p in glob.glob(f"{root}/**/*", recursive=True)
        if p.lower().endswith(IMG_EXT)
    }


def open_image(filename, index):
    """Open an image from the index by its base filename."""
    bn = os.path.basename(str(filename))
    if bn not in index:
        raise FileNotFoundError(f"Image '{bn}' not found in provided index.")
    return Image.open(index[bn]).convert("RGB")


def gold_records(gold_df):
    """Convert gold DataFrame into filename -> [{'medicine', 'dosage', 'frequency'}, ...]"""
    out = {}
    for fn, g in gold_df.groupby("filename"):
        out[fn] = g[["medicine", "dosage", "frequency"]].astype(str).to_dict("records")
    return out

