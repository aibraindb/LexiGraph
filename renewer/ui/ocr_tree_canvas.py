


import sys, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    print(f"Adding {ROOT} to PYTHONPATH")
    sys.path.insert(0,str(ROOT))

import io
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from app.core.ocr_indexer import index_pdf_bytes, pages_as_dicts

st.set_page_config(page_title="LexiGraph • OCR Tree & Canvas", layout="wide")

st.title("📄 LexiGraph — OCR Tree & Canvas (digital-first)")
st.caption("Left: document tree (lines). Right: page preview + overlays. Click a line to highlight; toggle word/line boxes.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    render_dpi = st.slider("Render DPI", 100, 300, 144, 10)
    show_line_boxes = st.checkbox("Show line boxes", value=True)
    show_word_boxes = st.checkbox("Show word boxes", value=False)
    stroke_width = st.slider("Overlay width", 1, 5, 2)
    st.divider()
    st.caption("Tip: If the preview is empty but the tree is populated, your PDF is digital-only and preview rendering failed. Lines still work.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

# Index
pages=[]
try:
    pdf_bytes = uploaded.read()
    pages = index_pdf_bytes(pdf_bytes, render_dpi=render_dpi)
    pages = pages_as_dicts(pages)
except Exception as e:
    st.error(f"OCR index failed: {e}")
    st.stop()

st.success(f"Indexed {len(pages)} page(s).")

# Session defaults
if "cur_page" not in st.session_state: st.session_state["cur_page"] = 0
if "selected_line_id" not in st.session_state: st.session_state["selected_line_id"] = -1

# Layout
col_tree, col_view = st.columns([0.38, 0.62])

# ---------- TREE ----------
with col_tree:
    st.subheader("Document Tree")
    if not pages:
        st.info("No pages found in PDF.")
    else:
        pg = pages[st.session_state["cur_page"]]
        lines = pg.get("lines", [])
        words = pg.get("words", [])
        st.caption(f"Page {st.session_state['cur_page']+1}: {len(lines)} line(s), {len(words)} word(s)")

        # Select a line
        options = [f"[{i+1}] {(ln.get('text','') or '').splitlines()[0][:60]}" for i, ln in enumerate(lines)]
        idx_default = st.session_state["selected_line_id"]
        if not (0 <= idx_default < len(options)):
            idx_default = 0 if options else 0

        if options:
            chosen = st.selectbox("Line", options=options, index=idx_default, key="line_select")
            st.session_state["selected_line_id"] = options.index(chosen)
            cur_idx = st.session_state["selected_line_id"]
            st.text_area("Selected text", value=lines[cur_idx].get("text",""), height=120)
        else:
            st.info("No lines on this page.")

        # Page navigation
        if len(pages) > 1:
            ncol1, ncol2, ncol3 = st.columns(3)
            with ncol1:
                if st.button("⟨ Prev", use_container_width=True) and st.session_state["cur_page"] > 0:
                    st.session_state["cur_page"] -= 1
                    st.session_state["selected_line_id"] = -1
            with ncol2:
                st.write(f"Page {st.session_state['cur_page']+1} / {len(pages)}")
            with ncol3:
                if st.button("Next ⟩", use_container_width=True) and st.session_state["cur_page"] < len(pages)-1:
                    st.session_state["cur_page"] += 1
                    st.session_state["selected_line_id"] = -1

# ---------- PAGE VIEWER ----------
with col_view:
    st.subheader("Page Viewer / Annotator")
    pg = pages[st.session_state["cur_page"]]
    img_bytes = pg.get("image_bytes")

    if not img_bytes:
        st.warning("No page preview available (rendering failed). Tree still works. If needed, install PyMuPDF and re-run.")
    else:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        pdf_w, pdf_h = pg.get("size", (img.width, img.height))
        sx = (img.width / pdf_w) if pdf_w else 1.0
        sy = (img.height / pdf_h) if pdf_h else 1.0

        # overlays
        if show_line_boxes:
            for i, ln in enumerate(pg.get("lines", [])):
                x0,y0,x1,y1 = ln["bbox"]
                draw.rectangle([x0*sx,y0*sy,x1*sx,y1*sy], outline="orange", width=stroke_width)
        if show_word_boxes:
            for wd in pg.get("words", []):
                x0,y0,x1,y1 = wd["bbox"]
                draw.rectangle([x0*sx,y0*sy,x1*sx,y1*sy], outline="deepskyblue", width=1)

        # highlight selected line
        sel = st.session_state.get("selected_line_id", -1)
        if 0 <= sel < len(pg.get("lines", [])):
            x0,y0,x1,y1 = pg["lines"][sel]["bbox"]
            draw.rectangle([x0*sx,y0*sy,x1*sx,y1*sy], outline="limegreen", width=max(stroke_width,3))

        st.image(img, caption="Rendered page with overlays")

        # Optional: simple lasso box drawing (freehand rectangle)
        st.caption("Draw a rectangle to propose a correction (lasso).")
        cnv = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=2,
            stroke_color="#00FF00",
            background_image=img,
            update_streamlit=True,
            height=img.height,
            width=img.width,
            drawing_mode="rect",
            key="page_canvas"
        )
        if cnv.json_data and cnv.json_data.get("objects"):
            # take last rect
            last = cnv.json_data["objects"][-1]
            if last.get("type") == "rect":
                # normalize to PDF coords
                rx, ry = last.get("left",0), last.get("top",0)
                rw, rh = last.get("width",0), last.get("height",0)
                pdf_box = (rx/sx, ry/sy, (rx+rw)/sx, (ry+rh)/sy)
                st.info(f"Lasso PDF bbox: {tuple(round(v,2) for v in pdf_box)}")
