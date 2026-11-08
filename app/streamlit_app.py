"""MediLens word-reader demo.   Run:  streamlit run streamlit_app.py"""
import json

import streamlit as st
from PIL import Image

import word_reader as wr
from prescription_parser import build_prescription_json

st.set_page_config(page_title="MediLens", page_icon="💊", layout="centered")
st.title("MediLens: medicine name reader")
st.caption("Upload cropped single-line images (e.g. 'Tab. Covam'). Results are decision support only; "
           "always verify against the original prescription.")


@st.cache_resource(show_spinner="Loading model...")
def get_reader():
    return wr.WordReader()


try:
    reader = get_reader()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Set MEDILENS_TRAINER / MEDILENS_CKPT / MEDILENS_CONFIG / MEDILENS_VOCAB "
            "environment variables, or edit the defaults at the top of word_reader.py.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    thr = st.slider("Flag threshold (margin)", 0.0, 6.0, wr.DEFAULT_THRESHOLD, 0.1,
                help="Gap between the best and second-best known name. Higher = flag more for review.")
    st.write(f"Device: `{reader.device}` | Known names: {len(reader.vocab)}")

files = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if files:
    images = [Image.open(f) for f in files]
    readings = reader.read(images, threshold=thr)
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
        items.append({"margin": round(r.margin, 3),"raw_text": r.raw, "name": None if r.flagged else r.name,
                      "name_candidates": r.candidates, "form": None if r.flagged else r.form,
                      "dosage": None, "frequency": None, "instructions": [],
                      "needs_review": r.flagged, "source_file": f.name})
    doc = build_prescription_json(items, source_image=None)
    with st.expander("JSON for the database"):
        st.json(doc)
    st.download_button("Download JSON", json.dumps(doc, indent=2, ensure_ascii=False),
                       file_name="prescription.json", mime="application/json")
