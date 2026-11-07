import time
import numpy as np
import pytesseract
import easyocr

_reader = None


def get_reader(gpu=None):
    """Load EasyOCR once (loading takes several seconds)."""
    global _reader
    if _reader is None:
        if gpu is None:
            try:
                import torch
                gpu = torch.cuda.is_available()
            except ImportError:
                gpu = False
        _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def group_lines(results):
    """EasyOCR returns word/phrase boxes. Join boxes on approximately the same line into text lines."""
    if not results:
        return []
    if not results:
        return []
    items = []
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({
            "x": min(xs),
            "yc": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
            "text": text,
        })
    items.sort(key=lambda d: d["yc"])
    med_h = float(np.median([d["h"] for d in items])) or 10.0

    lines, cur = [], [items[0]]
    for it in items[1:]:
        if abs(it["yc"] - np.mean([c["yc"] for c in cur])) < 0.6 * med_h:
            cur.append(it)
        else:
            lines.append(cur)
            cur = [it]
    lines.append(cur)
    return [" ".join(d["text"] for d in sorted(l, key=lambda d: d["x"])) for l in lines]


def run_easyocr(img, gpu=None):
    """Run EasyOCR on the input image array or path."""
    t = time.time()
    res = get_reader(gpu).readtext(img)
    lines = group_lines(res)
    conf = float(np.mean([r[2] for r in res])) if res else 0.0
    return {
        "engine": "easyocr",
        "lines": lines,
        "text": "\n".join(lines),
        "conf": conf,
        "secs": time.time() - t,
    }


def run_tesseract(img):
    """Run Tesseract OCR on the input image as fallback."""
    t = time.time()
    cfg = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
    confs = [
        int(c)
        for c, w in zip(data["conf"], data["text"])
        if str(c) != "-1" and w.strip()
    ]
    text = pytesseract.image_to_string(img, config=cfg)
    lines = [l for l in text.splitlines() if l.strip()]
    conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
    return {
        "engine": "tesseract",
        "lines": lines,
        "text": "\n".join(lines),
        "conf": conf,
        "secs": time.time() - t,
    }


def ocr_with_fallback(img, min_conf=0.5, min_chars=20):
    """EasyOCR first. If it is unsure or returns almost nothing, also try Tesseract
    and keep the better result."""
    first = run_easyocr(img)
    if first["conf"] >= min_conf and len(first["text"]) >= min_chars:
        return first
    second = run_tesseract(img)
    score = lambda r: r["conf"] * min(len(r["text"]), 200)
    best = max([first, second], key=score)
    best["secs"] = first["secs"] + second["secs"]  # total time spent
    best["fallback_used"] = True
    return best

