import re


def extract_fields(text):
    """Extract medicine, dosage, and frequency candidates from OCR text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    meds, dosages, freqs = [], [], []
    dose_re = re.compile(r"\b\d+\s*(mg|ml|mcg|g|iu)\b", re.I)
    freq_re = re.compile(r"\b(\d+\s*[x×]\s*\w+|\w+\s*(daily|times|bd|tds|od|hs))\b", re.I)
    for l in lines:
        m = dose_re.search(l)
        if m:
            dosages.append(m.group(0))
        f = freq_re.search(l)
        if f:
            freqs.append(f.group(0))
        # treat lines without digits as medicine name candidates
        if not any(c.isdigit() for c in l) and len(l) > 3:
            meds.append(l)
    return {"medicine": meds, "dosage": dosages, "frequency": freqs}
