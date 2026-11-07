# Medical Prescription OCR

A machine learning pipeline for extracting medicine name, dosage, and frequency from prescription images (printed & handwritten) using OpenCV preprocessing, EasyOCR (with GPU acceleration), Tesseract fallback, and rule-based extraction.

> **Disclaimer:** This project is a class demonstration / educational lab project and is **not for clinical use**. Always verify with the original prescription.

---

## 1. Pipeline Architecture

```
Image ──► Preprocessing ──► OCR Engines ───────► Information Extraction ──► UI
          (OpenCV)          - EasyOCR (GPU)       (Regex + RapidFuzz)       (Streamlit)
                            - Tesseract fallback
```

### Components Implemented (Phases 0 to 3)

| Component | File | Description |
|---|---|---|
| **Phase 0: Setup** | `requirements.txt`, `.gitignore` | Environment specification, dependencies, and git exclusions. |
| **Phase 1: Data Pipeline** | `src/data.py`, `notebooks/01_data.ipynb` | Dataset download helpers, index builders, and train/dev/test image splitters. |
| **Phase 2: Preprocessing** | `src/preprocess.py`, `notebooks/02_preprocess.ipynb` | Grayscale conversion, aspect-ratio preserving resize (1600px), deskewing, Non-Local Means denoising, and adaptive binarization. |
| **Phase 3: OCR Engines** | `src/ocr.py`, `notebooks/03_ocr.ipynb` | GPU-accelerated EasyOCR line grouping, Tesseract fallback, combined confidence scoring, and fallback routing. |

---

## 2. Directory Structure

```
medical-prescription-ocr/
├── notebooks/
│   ├── 01_data.ipynb          # Data loading, Kaggle integration, split generation
│   ├── 02_preprocess.ipynb    # Visual stage inspection, OpenCV benchmarking
│   └── 03_ocr.ipynb           # EasyOCR + Tesseract fallback evaluation
├── src/
│   ├── __init__.py
│   ├── data.py                # Dataset indexing & reading utilities
│   ├── preprocess.py          # OpenCV preprocessing functions
│   └── ocr.py                 # EasyOCR, Tesseract, and fallback logic
├── data/                      # Gold label CSVs (dev_gold.csv, test_gold.csv)
├── app/                       # Streamlit application (Phase 6)
├── Docs/                      # Reports, metrics, slides
├── requirements.txt
├── .gitignore
├── plan.md
└── README.md
```

---

## 3. Environment Setup

### 3.1 Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (e.g. RTX 3050) or CPU
- Tesseract OCR engine installed:
  - **Ubuntu / WSL2:** `sudo apt update && sudo apt install -y tesseract-ocr`
  - **Windows:** Install [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract.exe` is in PATH.

### 3.2 Install Dependencies
Install PyTorch with CUDA first (visit [pytorch.org](https://pytorch.org)), then:
```bash
pip install -r requirements.txt
```

### 3.3 Kaggle Setup (for Datasets)
Place your Kaggle API key at `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 4. Running Phases 1-3

1. **Phase 1 (Data):** Run `notebooks/01_data.ipynb` to download the Kaggle synthetic dataset (`priyanshuyav13/synthetic-medical-prescription-ocr-dataset`) and generate `data/dev_gold.csv` (200 dev images) and `data/test_gold.csv` (300 test images).
2. **Phase 2 (Preprocessing):** Run `notebooks/02_preprocess.ipynb` to inspect each OpenCV step (`gray`, `resized`, `denoised`, `binary`) and verify execution speed (<2s per image).
3. **Phase 3 (OCR Engines):** Run `notebooks/03_ocr.ipynb` to test text extraction across dev images, benchmark preprocessing stages on OCR confidence, and verify the fallback logic.

