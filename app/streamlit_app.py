"""MediLens word-reader demo.   Run:  streamlit run streamlit_app.py"""
import json

import streamlit as st
from PIL import Image, ImageOps, ImageDraw
from page_reader import PageReader

import word_reader as wr
from prescription_parser import build_prescription_json, parse_line, dose_review_reasons

st.set_page_config(page_title="MediLens", page_icon="💊", layout="centered")
st.title("MediLens: medicine name reader")
st.caption("Upload cropped single-line images (e.g. 'Tab. Covam'). Results are decision support only; "
           "always verify against the original prescription.")


@st.cache_resource(show_spinner="Loading model...")
def get_reader():
    return wr.WordReader()


try:
    reader = get_reader()
    @st.cache_resource(show_spinner="Loading detector...")
    def get_page_reader():
        return PageReader(get_reader())
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Set MEDILENS_TRAINER / MEDILENS_CKPT / MEDILENS_CONFIG / MEDILENS_VOCAB "
            "environment variables, or edit the defaults at the top of word_reader.py.")
    st.stop()

with st.sidebar:
    mode = st.radio("Input type", ["Full prescription", "Cropped word"])    
    st.header("Settings")
    thr = st.slider("Flag threshold (margin)", 0.0, 6.0, wr.DEFAULT_THRESHOLD, 0.1,
                help="Gap between the best and second-best known name. Higher = flag more for review.")
    st.write(f"Device: `{reader.device}` | Known names: {len(reader.vocab)}")

files = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if files and mode == "Full prescription":
    page_reader = get_page_reader()
    items = []
    for f in files:
        im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        boxes, crops, readings = page_reader.read(im, threshold=thr)
        ok = [(c, r, b) for b, c, r in zip(boxes, crops, readings) if not r.flagged]
        bad = [(c, r, b) for b, c, r in zip(boxes, crops, readings) if r.flagged]
        st.subheader(f.name)
        st.write(f"**{len(ok)}** medicine names accepted | {len(bad)} boxes unverified")
        shown = im.copy(); d = ImageDraw.Draw(shown)
        for b, r in zip(boxes, readings):
            d.rectangle(b, outline=(0, 160, 0) if not r.flagged else (200, 0, 0), width=3)
        st.image(shown, caption="green = accepted, red = unverified", use_container_width=True)
        for c, r, b in ok:
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                c1.image(c, use_container_width=True)
                c2.success(f"**{r.name}**   |   form: {r.form or 'not written'}")
                c2.caption(f"Raw OCR: {r.raw} | margin {r.margin:.2f}")
            items.append({"margin": round(r.margin, 3), "raw_text": r.raw, "name": r.name,
                          "name_candidates": r.candidates, "form": r.form, "dosage": None,
                          "needs_review": True, "bbox": b, "source_file": f.name})
        with st.expander(f"Unverified boxes with a readable guess ({len(bad)} total)"):
            for c, r, b in bad:
                if len(r.raw) >= 3:                      # display filter only, hides most junk boxes
                    st.image(c, width=220)
                    st.caption(f"raw: {r.reason or ''} | closest: {', '.join(r.candidates)} | margin {r.margin:.2f}")
    doc = build_prescription_json(items, source_image=None)
    with st.expander("JSON for the database"):
        st.json(doc)
    st.download_button("Download JSON", json.dumps(doc, indent=2, ensure_ascii=False),
                       file_name="prescription.json", mime="application/json", key="dl_page")

if files and mode == "Cropped word":
    images = [ImageOps.exif_transpose(Image.open(f)) for f in files]
    readings = reader.read(images, threshold=thr, gate = True)
    items = []
    st.write(f"**{len(readings)}** images, **{sum(r.flagged for r in readings)}** need review")
    for f, im, r in zip(files, images, readings):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            c1.image(im, caption=f.name, use_container_width=True)
            form = r.form or "not written"
            if r.flagged:
                c2.warning("Unverified. Needs review")
                c2.markdown(f"Raw OCR (rough guess): **{r.raw or '(nothing read)'}**")
                c2.markdown("Closest known names: " + ", ".join(f"`{c}`" for c in r.candidates))
            else:
                c2.success(f"**{r.name}**   |   form: {form}")
                c2.markdown(f"Raw OCR: `{r.raw}`")
            c2.caption(f"Margin {r.margin:.2f} (flagged below {thr:.2f}) | raw-vs-match disagreement {r.disagreement:.2f}")
        p = parse_line(r.raw)                       # dose / frequency / instructions from the raw OCR text
        dose_flags = dose_review_reasons(p)
        ##
        if dose_flags:
            c2.caption("Dose check: " + ", ".join(dose_flags))
        items.append({"margin": round(r.margin, 3), "raw_text": r.raw,
                      "name": None if r.flagged else r.name,
                      "name_candidates": r.candidates, "form": None if r.flagged else r.form,
                      "dosage": None, "dosage_text_raw": p["dosage"]["raw"] if p["dosage"] else None, "frequency": p["frequency"],
                      "instructions": p["instructions"], "digit_fixes": p["digit_fixes"],
                      "needs_review": True, "dose_review_reasons": dose_flags,
                      "source_file": f.name})
    doc = build_prescription_json(items, source_image=None)
    with st.expander("JSON for the database"):
        st.json(doc)
    st.download_button("Download JSON", json.dumps(doc, indent=2, ensure_ascii=False),
                       file_name="prescription.json", mime="application/json")
