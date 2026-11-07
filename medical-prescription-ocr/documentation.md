# Medical Prescription OCR Documentation

## 1. Overview

This project is an educational machine-learning pipeline for extracting text and
structured prescription information from printed or handwritten medical
prescription images. The pipeline combines:

- OpenCV image preprocessing
- EasyOCR as the primary OCR engine
- Tesseract as a fallback OCR engine
- Regular expressions for dosage and frequency candidates
- Levenshtein-based character error rate (CER) and sequence-based word error
  rate (WER) metrics

The project is a class demonstration and is **not intended for clinical use**.
All extracted medicine names, dosages, and frequencies must be checked against
the original prescription by a qualified person.

## 2. Repository Structure

```text
medical-prescription-ocr/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── extract.py
│   ├── metrics.py
│   ├── ocr.py
│   └── preprocess.py
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_preprocess.ipynb
│   └── 03_ocr.ipynb
├── data/
│   ├── bd_vocab.csv
│   ├── bd_word_test.csv
│   ├── dev_gold.csv
│   ├── medicine_list.csv
│   ├── synthetic_drugs.csv
│   └── test_gold.csv
├── tests/
├── requirements.txt
├── README.md
└── documentation.md
```

The virtual-environment and cache directories are local development artifacts
and are not part of the application source.

## 3. Processing Pipeline

```text
Prescription image
        │
        ▼
Image loading and indexing
        │
        ▼
OpenCV preprocessing
(gray → resized → optional deskew → denoised → binary)
        │
        ▼
EasyOCR
        │
        ├── sufficiently confident result ──► OCR result
        │
        └── low confidence or short result
                    │
                    ▼
                Tesseract
                    │
                    ▼
             best scored OCR result
        │
        ▼
Rule-based field extraction
medicine names, dosages, frequencies
        │
        ▼
Evaluation with CER and WER
```

## 4. Source Modules

### `src/data.py`

Provides image and annotation helpers.

#### Configuration

- `DATA_ROOT` is read from the `OCR_DATA_ROOT` environment variable.
- If the variable is not set, the default is `~/ocr_data`.
- Supported image extensions are `.png`, `.jpg`, `.jpeg`, and `.webp`.

#### Functions

| Function | Description |
|---|---|
| `build_image_index(root)` | Recursively finds supported images and returns a mapping from base filename to full path. |
| `open_image(filename, index)` | Looks up an image by base filename, opens it with Pillow, and converts it to RGB. Raises `FileNotFoundError` when the filename is absent. |
| `gold_records(gold_df)` | Groups a gold-label DataFrame by `filename` and returns medicine, dosage, and frequency records. |

Because the index uses only the base filename as its key, duplicate filenames in
different directories overwrite earlier entries.

### `src/preprocess.py`

Converts source images into OCR-friendly grayscale variants.

| Function | Description |
|---|---|
| `to_bgr(img)` | Converts a Pillow image to a BGR NumPy array. NumPy inputs are returned unchanged. |
| `resize_long_side(gray, target=1600)` | Resizes an image so its longest side is approximately `target`, preserving aspect ratio. It skips resizing when the scale is within 5% of 1. |
| `denoise(gray)` | Applies OpenCV fast Non-Local Means denoising with fixed template and search windows. |
| `binarize(gray)` | Applies Gaussian adaptive thresholding with a 31-pixel block size and constant 15. |
| `deskew(gray)` | Estimates text orientation and rotates images only for plausible angles between 0.5 and 15 degrees. |
| `preprocess(img, target=1600, do_deskew=False)` | Runs the preprocessing stages and returns all generated arrays in a dictionary. |

The returned dictionary normally contains `gray`, `resized`, `denoised`, and
`binary`. When `do_deskew=True`, it also contains `deskewed`.

### `src/ocr.py`

Defines the OCR engines and fallback policy.

#### EasyOCR reader lifecycle

`get_reader(gpu=None)` lazily creates one EasyOCR reader and stores it in the
module-level `_reader` variable. This avoids repeatedly loading the OCR model.
When `gpu` is omitted, the function checks `torch.cuda.is_available()`. If
PyTorch cannot be imported, CPU mode is selected.

#### Line grouping

`group_lines(results)` is intended to convert EasyOCR word or phrase boxes into
human-readable lines. It calculates each box's horizontal position, vertical
center, and height, groups boxes whose vertical centers are close to one another,
and sorts each group from left to right.

#### OCR functions

| Function | Description |
|---|---|
| `run_easyocr(img, gpu=None)` | Runs EasyOCR, groups its boxes into lines, calculates mean confidence, and returns text, lines, engine name, confidence, and elapsed time. |
| `run_tesseract(img)` | Runs Tesseract with `--oem 3 --psm 6`, extracts line text, converts Tesseract's percentage confidence to a 0–1 score, and returns the same result shape. |
| `ocr_with_fallback(img, min_conf=0.5, min_chars=20)` | Keeps a sufficiently confident EasyOCR result; otherwise runs Tesseract and chooses the result with the larger confidence/length score. |

OCR result objects have this general shape:

```python
{
    "engine": "easyocr" or "tesseract",
    "lines": ["recognized line 1", "recognized line 2"],
    "text": "recognized line 1\nrecognized line 2",
    "conf": 0.0,
    "secs": 0.0,
}
```

When fallback processing occurs, `fallback_used` is set to `True` and `secs`
contains the combined time spent on both engines.

### `src/extract.py`

`extract_fields(text)` performs simple line-based extraction:

- Dosages match a number followed by `mg`, `ml`, `mcg`, `g`, or `iu`.
- Frequencies match patterns such as `2x daily`, `1 x day`, `daily`, `times`,
  `bd`, `tds`, `od`, or `hs`.
- Non-empty lines longer than three characters that contain no digits are treated
  as medicine-name candidates.

The return value is:

```python
{
    "medicine": [...],
    "dosage": [...],
    "frequency": [...],
}
```

This is a candidate extractor rather than a medical parser. It does not verify
that a medicine name is valid, associate a dosage with a particular medicine,
or resolve OCR spelling errors.

### `src/metrics.py`

| Function | Description |
|---|---|
| `cer(pred, truth)` | Calculates normalized Levenshtein character distance. Empty truth returns `0.0` for an empty prediction and `1.0` otherwise. |
| `wer(pred, truth)` | Splits both strings into words and calculates `1 - SequenceMatcher.ratio()`. It uses the same empty-truth behavior as `cer`. |

Both metrics are error rates, so lower values indicate better agreement.

## 5. Data and Notebooks

The notebooks represent the staged workflow:

1. **`01_data.ipynb`** — prepares dataset indexes and gold-label splits.
2. **`02_preprocess.ipynb`** — inspects preprocessing stages and compares image
   transformations.
3. **`03_ocr.ipynb`** — evaluates EasyOCR, Tesseract fallback behavior, and OCR
   confidence across prescription images.

The CSV files in `data/` provide gold labels, medicine vocabulary, synthetic
drug data, and word-level test data used by the notebooks and evaluation work.

## 6. Installation and Prerequisites

Requirements are listed in `requirements.txt`, including NumPy, OpenCV,
Pillow, EasyOCR, PyTesseract, PyTorch, Pandas, and
`python-Levenshtein`.

Recommended setup:

1. Use Python 3.10 or newer.
2. Install a compatible PyTorch build, with CUDA support if an NVIDIA GPU is
   available.
3. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the Tesseract OCR executable separately and ensure it is available
   on `PATH`.
5. Set `OCR_DATA_ROOT` when image data is stored outside `~/ocr_data`.

EasyOCR may download model files the first time its reader is created. GPU
availability is detected automatically unless the caller explicitly passes
`gpu=True` or `gpu=False`.

## 7. Example Usage

```python
from src.data import build_image_index, open_image
from src.preprocess import preprocess
from src.ocr import ocr_with_fallback
from src.extract import extract_fields

index = build_image_index("path/to/images")
image = open_image("prescription.png", index)
stages = preprocess(image, do_deskew=True)

ocr_result = ocr_with_fallback(stages["binary"])
fields = extract_fields(ocr_result["text"])

print(ocr_result["engine"], ocr_result["conf"])
print(fields)
```

For best results, compare OCR on multiple preprocessing stages rather than
assuming that the binary image is always superior.

## 8. Current Implementation Notes

The current `src/ocr.py` contains an indentation error in `group_lines` and
therefore cannot currently be imported or compiled. Its body also contains a
duplicated `if not results` check and references `bbox` and `text` before the
missing result-iteration logic. The intended behavior is clear from the later
grouping and sorting code, but this module must be corrected before the OCR
pipeline can run.

Other implementation limitations include:

- OCR confidence is an average and does not guarantee medical accuracy.
- The fallback score favors confidence and text length but does not measure
  semantic correctness.
- The extractor uses permissive regular expressions and may produce false
  positives or miss uncommon prescription notation.
- `build_image_index` does not detect duplicate base filenames.
- The source tree currently has no readable test modules; cached test artifacts
  are not a substitute for source tests.

## 9. Safety and Operational Guidance

This system must not be used to make prescribing, dispensing, or dosage
decisions without human verification. Preserve the original image, display the
OCR confidence and selected engine, and require a reviewer to confirm every
medicine, dosage, and frequency before the result is used.
