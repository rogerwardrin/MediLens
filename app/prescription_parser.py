"""Turn recognised prescription text into structured, JSON-ready records.

No torch/OCR dependency, so it can be tested on plain strings:
    python prescription_parser.py --selftest

Fields extracted from one prescription line:
    form        tablet | capsule | syrup | suspension | liquid | injection | drops | ointment | cream | None
    name        medicine name (matched against a known-name list when one is given)
    dosage      strength, e.g. 500 mg or 5/80 mg
    frequency   e.g. 1+0+1, OD, BD, TDS, every 8 hours
    instructions  after_meals, before_meals, ...
Anything uncertain sets needs_review = True. Decision support only: a person must verify.
"""
import json
import re
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = "0.1"

FORM_SURFACES = {            # normalised form -> written abbreviations (lowercase); first is the short form
    "tablet": ["tab", "tablet", "tabs"],
    "capsule": ["cap", "capsule", "caps"],
    "syrup": ["syp", "syr", "syrup"],
    "suspension": ["susp", "suspension"],
    "liquid": ["liq", "liquid"],
    "injection": ["inj", "injection"],
    "drops": ["drops", "gtt"],
    "ointment": ["oint", "ointment"],
    "cream": ["cream"],
}
SURFACE_TO_FORM = {s: f for f, ss in FORM_SURFACES.items() for s in ss}
DEFAULT_FORMS = ("tablet", "capsule")      # the forms your photos actually show


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_candidates(names, forms=DEFAULT_FORMS):
    """All strings the reader should score: each name bare, and with a written form prefix.
    Returns (candidates, variants_per_name). Candidates are (surface_text, name, form) in name-major order."""
    prefixes = [("", None)]
    for f in forms:
        short = FORM_SURFACES[f][0]
        for text in (short.capitalize(), short):
            for punct in ("", "."):
                prefixes.append((f"{text}{punct} ", f))
    cands = [(p + n, n, f) for n in names for p, f in prefixes]
    return cands, len(prefixes)


# ---------------------------------------------------------------- parsing
DOSE_RE = re.compile(
    r"(?P<v>\d+(?:\.\d+)?(?:\s*[/\-]\s*\d+(?:\.\d+)?)*)\s*(?P<u>mg|mcg|µg|μg|ug|g|ml|iu|%)(?![a-z])", re.I)
SCHED_RE = re.compile(
    r"(?<![\d/.\-+])(?P<a>[0-3]|½)\s*[+\-–]\s*(?P<b>[0-3]|½)\s*[+\-–]\s*(?P<c>[0-3]|½)"
    r"(?:\s*[+\-–]\s*(?P<d>[0-3]|½))?(?![\d.])")
ABBREV = {"od": 1, "qd": 1, "bd": 2, "bid": 2, "tds": 3, "tid": 3, "qid": 4}
ABBREV_RE = re.compile(r"\b(od|qd|bd|bid|tds|tid|qid|hs|sos|prn|stat)\b", re.I)
WORDS_RE = re.compile(r"\b(once|twice|thrice)\s+(?:a\s+)?(?:daily|day)\b", re.I)
TIMES_RE = re.compile(r"\b(\d+)\s*(?:x|times)\s*(?:a\s*|per\s*)?(?:daily|day|/day)\b", re.I)
EVERY_RE = re.compile(r"\bevery\s+(\d+)\s*(?:hours?|hrs?|h)\b|\bq(\d+)h\b", re.I)
INSTR = [
    (re.compile(r"\bafter\s+(?:meals?|food)\b", re.I), "after_meals"),
    (re.compile(r"\bbefore\s+(?:meals?|food)\b", re.I), "before_meals"),
    (re.compile(r"\bwith\s+(?:meals?|food)\b", re.I), "with_meals"),
    (re.compile(r"\bempty\s+stomach\b", re.I), "empty_stomach"),
    (re.compile(r"\b(?:at\s+)?(?:bed\s?time|night)\b", re.I), "at_night"),
]
RX_RE = re.compile(r"^\s*(?:rx\.?|℞|r/)\s+", re.I)
WORD_NUM = {"once": 1, "twice": 2, "thrice": 3}


def _cut(text, m):
    return text[:m.start()] + " " + text[m.end():]


def _num(s):
    return 0.5 if s == "½" else float(s)


def parse_line(text, names=None, max_name_dist=0.34):
    """Parse one line such as 'Tab. Covam 5/80mg 1+0+1 after meals' into a record."""
    work = RX_RE.sub("", text or "")
    review = []

    # --- form: written abbreviation at the start of the line
    form, form_raw = None, None
    m = re.match(r"^\s*([A-Za-z]{2,10})\b\.?\s*(.*)$", work)
    if m and m.group(2).strip():
        tok = m.group(1).lower()
        if tok in SURFACE_TO_FORM:
            form, form_raw, work = SURFACE_TO_FORM[tok], m.group(1), m.group(2)
        elif 3 <= len(tok) <= 4:               # OCR slip such as 'Tas', 'Tak', 'lap'
            hits = sorted({f for s, f in SURFACE_TO_FORM.items() if len(s) >= 3 and edit_distance(tok, s) == 1})
            if len(hits) == 1:
                form, form_raw, work = hits[0], m.group(1), m.group(2)
                review.append("form_uncertain")

    # --- dosage (strength)
    dosage = None
    m = DOSE_RE.search(work)
    if m:
        values = [float(x) for x in re.split(r"\s*[/\-]\s*", m.group("v").strip())]
        dosage = {"values": [int(v) if v.is_integer() else v for v in values],
                  "unit": m.group("u").lower().replace("µ", "u").replace("μ", "u"), "raw": m.group(0).strip()}
        work = _cut(work, m)

    # --- frequency
    frequency = None
    m = SCHED_RE.search(work)
    if m:
        parts = [g for g in m.group("a", "b", "c", "d") if g]
        frequency = {"raw": m.group(0).strip(), "code": "+".join(parts),
                     "times_per_day": sum(_num(p) for p in parts)}
        work = _cut(work, m)
    else:
        for rx, kind in ((WORDS_RE, "words"), (TIMES_RE, "times"), (EVERY_RE, "every"), (ABBREV_RE, "abbrev")):
            m = rx.search(work)
            if not m:
                continue
            if kind == "words":
                n = WORD_NUM[m.group(1).lower()]
                frequency = {"raw": m.group(0), "code": f"{n}x_daily", "times_per_day": n}
            elif kind == "times":
                frequency = {"raw": m.group(0), "code": f"{m.group(1)}x_daily", "times_per_day": int(m.group(1))}
            elif kind == "every":
                h = int(m.group(1) or m.group(2))
                frequency = {"raw": m.group(0), "code": f"every_{h}h", "times_per_day": round(24 / h, 2) if h else None}
            else:
                a = m.group(1).lower()
                code = {"hs": "bedtime", "sos": "as_needed", "prn": "as_needed", "stat": "once_now"}.get(a, a)
                frequency = {"raw": m.group(0), "code": code, "times_per_day": ABBREV.get(a)}
            work = _cut(work, m)
            break

    # --- instructions
    instructions = []
    for rx, label in INSTR:
        m = rx.search(work)
        if m:
            instructions.append(label)
            work = _cut(work, m)

    # --- leftover: name (letters) and a bare strength number such as 'starcox 90'
    cleaned = re.sub(r"[^A-Za-z0-9'\- ]", " ", work)
    tokens = [t.strip("-'") for t in cleaned.split()]
    tokens = [t for t in tokens if t]
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    name_raw = " ".join(words)
    if dosage is None and nums:
        dosage = {"values": [int(nums[0])], "unit": None, "raw": nums[0]}
        review.append("dosage_unit_missing")

    # --- name: snap to the known list when given
    name, verified, cands = name_raw, False, []
    if names and name_raw:
        scored = sorted(((edit_distance(name_raw.lower(), n.lower()) / max(len(n), len(name_raw)), n) for n in names))
        cands = [n for _, n in scored[:3]]
        if scored[0][0] <= max_name_dist:
            name, verified = scored[0][1], True
    if not name_raw:
        review.append("name_missing")
    elif not verified:
        review.append("name_unverified")

    return {"raw_text": text, "name": name or None, "name_verified": verified, "name_candidates": cands,
            "form": form, "form_raw": form_raw, "dosage": dosage, "frequency": frequency,
            "instructions": instructions, "needs_review": bool(review), "review_reasons": review}


def build_prescription_json(items, source_image=None):
    doc = {"schema_version": SCHEMA_VERSION,
           "source_image": source_image,
           "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "disclaimer": "Automatically extracted. Verify every field against the original prescription.",
           "medicines": items,
           "needs_review": any(i.get("needs_review") for i in items) or not items}
    return doc


# ---------------------------------------------------------------- self-test
def _selftest():
    names = ["Covam", "Risek", "Azitma", "Atcam", "Ascoril", "Ceftriaxone", "Napa", "Caricef", "My-D", "Starcox"]

    def check(line, **want):
        r = parse_line(line, names)
        for k, v in want.items():
            got = r[k]
            if isinstance(v, dict):
                assert isinstance(got, dict) and all(got.get(a) == b for a, b in v.items()), (line, k, got, v)
            else:
                assert got == v, (line, k, got, v)
        return r

    check("tab covam 5-80mg", form="tablet", name="Covam", dosage={"values": [5, 80], "unit": "mg"}, needs_review=False)
    check("Tab. Covam 5/80mg", form="tablet", name="Covam", dosage={"values": [5, 80], "unit": "mg"})
    check("cap risek 20mg", form="capsule", name="Risek", dosage={"values": [20], "unit": "mg"}, frequency=None)
    check("tab azitma 500mg", form="tablet", name="Azitma", dosage={"values": [500], "unit": "mg"})
    check("Cap. Caricef 40 mg BD", form="capsule", name="Caricef", frequency={"code": "bd", "times_per_day": 2})
    check("Tab Napa 500mg 1+0+1 after meals", name="Napa", frequency={"code": "1+0+1", "times_per_day": 2},
          instructions=["after_meals"])
    check("Tab Atcam 1-1-1", name="Atcam", frequency={"code": "1+1+1", "times_per_day": 3})
    check("Syp Ascoril 5ml TDS", form="syrup", name="Ascoril", dosage={"values": [5], "unit": "ml"},
          frequency={"times_per_day": 3})
    check("Inj Ceftriaxone 1g OD", form="injection", name="Ceftriaxone", dosage={"values": [1], "unit": "g"},
          frequency={"code": "od"})
    check("Rx Tab Atcam twice daily", form="tablet", name="Atcam", frequency={"code": "2x_daily", "times_per_day": 2})
    check("Tab Napa every 8 hours", frequency={"code": "every_8h", "times_per_day": 3.0})
    check("tab my d", form="tablet", name="My-D")
    check("starcox 90", form=None, name="Starcox", dosage={"values": [90], "unit": None}, needs_review=True)
    r = check("Tas Covam 20mg", form="tablet", needs_review=True)          # OCR slip in the form word
    assert "form_uncertain" in r["review_reasons"]
    r = check("lap Risek", form="capsule", needs_review=True)
    r = check("Tab Qwertyxyz 10mg", name_verified=False, needs_review=True)  # not in the list -> raw text kept, flagged
    assert r["name"] == "Qwertyxyz"
    r = check("Az", form=None)                                               # short name is not mistaken for a form
    doc = build_prescription_json([parse_line("tab covam 5-80mg 1+0+1", names)], "x.png")
    json.loads(json.dumps(doc))
    print("all parser self-tests passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        line = " ".join(a for a in sys.argv[1:]) or "Tab. Covam 5/80mg 1+0+1 after meals"
        print(json.dumps(parse_line(line), indent=2, ensure_ascii=False))
