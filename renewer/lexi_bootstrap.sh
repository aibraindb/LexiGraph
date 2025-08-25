#!/usr/bin/env bash
set -euo pipefail

# 1) Ensure venv is active
python --version >/dev/null 2>&1 || { echo "Python not found on PATH"; exit 1; }

# 2) Create folders
mkdir -p app/core ui data

# 3) Write requirements (minimal & Mac-safe)
cat > requirements-ocr-ui.txt << 'REQ'
streamlit==1.36.0
streamlit-drawable-canvas==0.9.3
pdfminer.six==20240706
pdfplumber==0.11.4
Pillow==10.3.0
scikit-learn==1.5.2
numpy>=1.26,<3
REQ

# 4) app/__init__.py
cat > app/__init__.py << 'PY'
# makes "app" a package
PY

# 5) app/core/__init__.py
cat > app/core/__init__.py << 'PY'
# core package
PY

# 6) app/core/ocr_indexer.py  (DIGITAL-PDF first; image/OCR stubbed)
cat > app/core/ocr_indexer.py << 'PY'
from __future__ import annotations
import io, math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import pdfplumber
from PIL import Image, ImageDraw

@dataclass
class Line:
    bbox: Tuple[float,float,float,float]   # (x0,y0,x1,y1) in PDF coords
    text: str
    words: List[int] = field(default_factory=list)

@dataclass
class Word:
    bbox: Tuple[float,float,float,float]
    text: str

@dataclass
class PageDict:
    number: int
    size: Tuple[float,float]               # (width,height) in PDF coords
    image_bytes: Optional[bytes]           # PNG bytes if rendered; else None
    lines: List[Line] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)

def _render_page_image(pp: pdfplumber.page.Page, dpi: int = 144) -> Optional[bytes]:
    """
    Try to render a page preview using pdfplumber's rasterizer.
    This path does NOT require PyMuPDF. If rendering fails, return None.
    """
    try:
        # pdfplumber Page has .to_image(resolution=)
        pi = pp.to_image(resolution=dpi)  # returns pdfplumber.display.PageImage
        # PageImage has .original (PIL Image)
        img = pi.original.convert("RGB")
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        # Safe fallback: no background image, tree will still work
        return None

def _group_words_to_lines(words: List[Dict], y_tol: float = 3.0, min_chars: int = 2) -> List[Line]:
    """Group words into text lines by y proximity; returns Line list."""
    if not words:
        return []
    # Normalize: words from pdfplumber.extract_words() have x0,y0,x1,y1,text
    items = []
    for i, w in enumerate(words):
        try:
            x0=float(w["x0"]); y0=float(w["top"]); x1=float(w["x1"]); y1=float(w["bottom"])
            txt=str(w.get("text","") or "")
            items.append((i, x0,y0,x1,y1, txt))
        except Exception:
            continue
    if not items:
        return []
    # Sort by y, then x
    items.sort(key=lambda t: (t[2], t[1]))
    lines: List[Line] = []
    cur: List[Tuple[int,float,float,float,float,str]] = []
    def flush():
        if not cur:
            return
        xs = [c[1] for c in cur]; ys0 = [c[2] for c in cur]; xs1 = [c[3] for c in cur]; ys1 = [c[4] for c in cur]
        text = " ".join([c[5] for c in cur]).strip()
        if len(text) >= min_chars:
            lines.append(Line(
                bbox=(min(xs), min(ys0), max(xs1), max(ys1)),
                text=text
            ))
    # Greedy grouping
    baseline = items[0][2]
    for rec in items:
        _, x0,y0,x1,y1, txt = rec
        if abs(y0 - baseline) <= y_tol:
            cur.append(rec)
        else:
            flush()
            cur = [rec]
            baseline = y0
    flush()
    return lines

def _words_from_plumber(pp: pdfplumber.page.Page) -> List[Word]:
    res = []
    try:
        ws = pp.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["top","bottom"]
        )
        for w in ws:
            x0=float(w["x0"]); x1=float(w["x1"])
            y0=float(w["top"]); y1=float(w["bottom"])
            txt=str(w.get("text","") or "")
            res.append(Word(bbox=(x0,y0,x1,y1), text=txt))
    except Exception:
        pass
    return res

def index_pdf_bytes(pdf_bytes: bytes, render_dpi: int = 144) -> List[PageDict]:
    """
    Build a structured OCR-like index for a PDF.
    DIGITAL-FIRST: uses pdfplumber to read embedded text and approximate lines.
    If page image rendering fails, we still return lines/words without image_bytes.
    """
    pages: List[PageDict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, pp in enumerate(pdf.pages):
            w,h = pp.width, pp.height
            img_bytes = _render_page_image(pp, dpi=render_dpi)
            # words & lines
            words = _words_from_plumber(pp)
            # for grouping, pdfplumber provides a separate extract_words;
            # we rebuild lines from that set to keep consistent coords
            raw_pw = pp.extract_words(use_text_flow=True, extra_attrs=["top","bottom"])
            lines = _group_words_to_lines(raw_pw, y_tol=3.0, min_chars=2)
            pages.append(PageDict(
                number=i,
                size=(w,h),
                image_bytes=img_bytes,
                lines=lines,
                words=words
            ))
    return pages

# convenience
def pages_as_dicts(pages: List[PageDict]) -> List[Dict]:
    out=[]
    for p in pages:
        out.append({
            "number": p.number,
            "size": p.size,
            "image_bytes": p.image_bytes,
            "lines": [ {"bbox":l.bbox,"text":l.text, "words":l.words} for l in p.lines ],
            "words": [ {"bbox":w.bbox,"text":w.text} for w in p.words ],
        })
    return out
PY

# 7) ui/ocr_tree_canvas.py (Tree + preview + overlays + selection sync)
cat > ui/ocr_tree_canvas.py << 'PY'
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
PY

# 8) Install and run
pip install -r requirements-ocr-ui.txt

echo ""
echo "✅ Install done."
echo "Run:  streamlit run ui/ocr_tree_canvas.py"
