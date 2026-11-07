"""MediLens word-reader demo.   Run:  streamlit run streamlit_app.py"""
import streamlit as st
from PIL import Image

import word_reader as wr

st.set_page_config(page_title="MediLens", page_icon="💊", layout="centered")
st.title("MediLens: medicine name reader")
st.caption("Upload cropped single-word images. Results are decision support only; "
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
    thr = st.slider("Flag threshold (disagreement)", 0.1, 0.8, wr.DEFAULT_THRESHOLD, 0.05,
                    help="Raw OCR vs vocabulary match, normalized edit distance. "
                         "Lower = flag more images for review.")
    st.write(f"Device: `{reader.device}` | Known names: {len(reader.vocab)}")

files = st.file_uploader("Word images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if files:
    images = [Image.open(f) for f in files]
    readings = reader.read(images, threshold=thr)
    n_flag = sum(r.flagged for r in readings)
    st.write(f"**{len(readings)}** images, **{n_flag}** need review")
    for f, im, r in zip(files, images, readings):
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            c1.image(im, caption=f.name, use_container_width=True)
            if r.flagged:
                c2.warning("Unverified. Needs review")
                c2.markdown(f"Raw OCR (rough guess): **{r.raw or '(nothing read)'}**")
                c2.markdown("Closest known names: " + ", ".join(f"`{c}`" for c in r.candidates))
            else:
                c2.success(f"Match: **{r.match}**")
                c2.markdown(f"Raw OCR: `{r.raw}`")
            c2.caption(f"Disagreement {r.disagreement:.2f} (threshold {thr:.2f})")
