#!/usr/bin/env bash
set -euo pipefail

mkdir -p app/core ui scripts data

############################################
# app/core/ocr_indexer.py
############################################
cat > app/core/ocr_indexer.py << 'PY'
from __future__ import annotations
import io, json, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# Prefer pdfplumber (pure Python on top of pdfminer) to avoid native lib drama.
# Fallback to PyMuPDF only if present.
_PLUMBER_OK = False
try:
    import pdfplumber
    _PLUMBER_OK = True
except Exception:
    _PLUMBER_OK = False

_FITZ_OK = False
try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except Exception:
    _FITZ_OK = False

from PIL import Image, ImageDraw


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Line:
    id: str
    text: str
    bbox: Tuple[float, float, float, float]  # (x0,y0,x1,y1) in PDF points
    conf: float = 1.0
    source: str = "pdf"
    page_index: int = 0

@dataclass
class Page:
    number: int           # 1-based page number
    width: float
    height: float
    image_bytes: Optional[bytes]  # background PNG to show in UI
    lines: List[Line]

@dataclass
class OCRIndex:
    pages: List[Page]


# -----------------------------
# Helpers
# -----------------------------
def _to_png_placeholder(width_pt: float, height_pt: float, dpi: int = 144) -> bytes:
    # A white page placeholder if we cannot rasterize PDF page
    w = int(width_pt * dpi / 72)
    h = int(height_pt * dpi / 72)
    if w <= 0 or h <= 0:
        w, h = 800, 1000
    img = Image.new("RGB", (w, h), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def _plumber_render_png(ppage, dpi: int = 144) -> Optional[bytes]:
    try:
        # pdfplumber’s to_image may require ImageMagick/Wand in some envs.
        # If it fails, we return None and let placeholder kick in.
        im = ppage.to_image(resolution=dpi)
        bio = io.BytesIO()
        im.original.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return None

def _fitz_render_png(fpage, dpi: int = 144) -> Optional[bytes]:
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = fpage.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None

def _normalize_lines(lines: List[Line]) -> List[Dict[str, Any]]:
    out = []
    for ln in lines:
        out.append({
            "id": ln.id,
            "text": ln.text,
            "bbox": ln.bbox,
            "conf": ln.conf,
            "source": ln.source,
            "page_index": ln.page_index
        })
    return out

def _page_to_dict(pg: Page) -> Dict[str, Any]:
    return {
        "number": pg.number,
        "width": pg.width,
        "height": pg.height,
        "image_bytes_b64": None,  # UI will not inline by default
        "lines": _normalize_lines(pg.lines)
    }

# -----------------------------
# Indexers
# -----------------------------
def _index_with_pdfplumber(pdf_bytes: bytes,
                           min_chars_in_line: int = 2,
                           line_merge_tol: int = 3) -> OCRIndex:
    pages: List[Page] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, p in enumerate(pdf.pages):
            width, height = float(p.width), float(p.height)
            # Words: list of dicts with x0, y0, x1, y1, text, top, bottom, baseline, etc.
            try:
                words = p.extract_words(keep_blank_chars=False, use_text_flow=True)
            except Exception:
                # fallback: single block
                words = []

            # group words into lines by 'top' with a tolerance
            lines: List[Line] = []
            if words:
                # sort by top then x0
                words = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))
                # simple clustering by 'top' ~ line
                current: List[Dict[str, Any]] = []
                last_top = None
                for w in words:
                    t = float(w.get("top", 0))
                    if last_top is None or abs(t - last_top) <= line_merge_tol:
                        current.append(w)
                        last_top = t if last_top is None else (last_top + t) / 2.0
                    else:
                        if current:
                            lines.append(_words_to_line(current, pi))
                        current = [w]
                        last_top = t
                if current:
                    lines.append(_words_to_line(current, pi))

            # Filter tiny lines
            lines = [ln for ln in lines if len(ln.text.strip()) >= min_chars_in_line]

            # Render background
            bg = _plumber_render_png(p) or _to_png_placeholder(width, height)

            pages.append(Page(
                number=pi+1, width=width, height=height,
                image_bytes=bg, lines=lines
            ))
    return OCRIndex(pages=pages)

def _words_to_line(cluster: List[Dict[str, Any]], page_index: int) -> Line:
    x0 = min(float(w["x0"]) for w in cluster)
    y0 = min(float(w["top"]) for w in cluster)
    x1 = max(float(w["x1"]) for w in cluster)
    y1 = max(float(w.get("bottom", w.get("y1", w.get("top", 0)+10))) for w in cluster)
    text = " ".join(w.get("text", "") for w in cluster if w.get("text"))
    lid = f"p{page_index}_l{abs(hash((x0,y0,x1,y1,text)))%1_000_000}"
    return Line(id=lid, text=text, bbox=(x0,y0,x1,y1), page_index=page_index)

def _index_with_fitz(pdf_bytes: bytes,
                     min_chars_in_line: int = 2) -> OCRIndex:
    # Use PyMuPDF text blocks
    pages: List[Page] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pi, fpage in enumerate(doc):
        width, height = float(fpage.rect.width), float(fpage.rect.height)
        # Extract text blocks
        lines: List[Line] = []
        try:
            blocks = fpage.get_text("blocks")  # x0,y0,x1,y1, text, block_no, ...
            for b in blocks:
                x0, y0, x1, y1, txt, *_ = b
                if not isinstance(txt, str):
                    continue
                for raw_line in txt.splitlines():
                    if len(raw_line.strip()) >= min_chars_in_line:
                        lid = f"p{pi}_l{abs(hash((x0,y0,x1,y1,raw_line)))%1_000_000}"
                        lines.append(Line(
                            id=lid, text=raw_line, bbox=(x0,y0,x1,y1),
                            page_index=pi
                        ))
        except Exception:
            pass

        bg = _fitz_render_png(fpage) or _to_png_placeholder(width, height)
        pages.append(Page(number=pi+1, width=width, height=height, image_bytes=bg, lines=lines))
    return OCRIndex(pages=pages)

# -----------------------------
# Public API
# -----------------------------
def index_pdf_bytes(pdf_bytes: bytes,
                    min_chars_in_line: int = 2,
                    line_merge_tol: int = 3) -> OCRIndex:
    """
    Return OCRIndex with pages -> lines (each with bbox).
    Uses pdfplumber by default; falls back to PyMuPDF if available.
    """
    if _PLUMBER_OK:
        return _index_with_pdfplumber(pdf_bytes,
                                      min_chars_in_line=min_chars_in_line,
                                      line_merge_tol=line_merge_tol)
    if _FITZ_OK:
        return _index_with_fitz(pdf_bytes, min_chars_in_line=min_chars_in_line)
    raise RuntimeError("No OCR backend available (pdfplumber / PyMuPDF missing). Install pdfplumber.")

def page_to_dict(pg: Page) -> Dict[str, Any]:
    return _page_to_dict(pg)

PY


############################################
# ui/ocr_tree_canvas.py
############################################
cat > ui/ocr_tree_canvas.py << 'PY'
import io, base64, json
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from app.core.ocr_indexer import index_pdf_bytes, page_to_dict, OCRIndex

st.set_page_config(page_title="Lexi — OCR Tree & Canvas (HITL)", layout="wide")
st.title("📄 Lexi — OCR Page Viewer / Annotator")

# --- Sidebar controls
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("PDF", type=["pdf"])
min_chars = st.sidebar.slider("Min chars per line", 1, 10, 2, 1)
merge_tol = st.sidebar.slider("Line merge tolerance (px)", 1, 10, 3, 1)

if "ocr_idx" not in st.session_state:
    st.session_state["ocr_idx"] = None
if "cur_page" not in st.session_state:
    st.session_state["cur_page"] = 0
if "objects" not in st.session_state:
    # canvas objects cache per page index
    st.session_state["objects"] = {}

def _png_to_bgimg(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes))

def _to_canvas_rect(line, page_w, page_h, active=False):
    # st_canvas expects left, top, width, height in pixels (we feed 1:1 from image pixels)
    x0, y0, x1, y1 = line["bbox"]
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    return {
        "type": "rect",
        "left": float(x0),
        "top": float(y0),
        "width": float(w),
        "height": float(h),
        "stroke": "#ff2d55" if not active else "#3b82f6",
        "fill": "rgba(255,45,85,0.08)" if not active else "rgba(59,130,246,0.12)",
        "strokeWidth": 2,
        "uuid": line["id"],
        "text": line.get("text","")
    }

def _build_tree(pg_dict):
    # very simple tree: just lines for now (can nest into blocks later)
    lines = pg_dict.get("lines", [])
    st.subheader("Document Tree (current page)")
    if not lines:
        st.info("No lines detected.")
        return None
    # selection
    options = [f"[{i+1}] {ln.get('text','')[:60].replace('\n',' ')}" for i, ln in enumerate(lines)]
    idx = st.selectbox("Lines", options=options, index=0, key=f"line_select_{st.session_state['cur_page']}")
    sel = lines[int(idx.split(']')[0][1:]) - 1]
    return sel

def _draw_canvas(pg_dict, initial_rects, background_img):
    # Canvas sized to page image
    W, H = background_img.size
    canvas_res = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_width=2,
        stroke_color="#ff2d55",
        background_image=background_img,
        update_streamlit=True,
        height=H,
        width=W,
        drawing_mode="transform",  # move/resize existing rects
        initial_drawing={"objects": initial_rects, "background": "#FFFFFF"},
        key=f"canvas_{st.session_state['cur_page']}",
    )
    return canvas_res

def _persist_canvas(page_idx, canvas_res):
    if canvas_res and canvas_res.json_data:
        try:
            jd = json.loads(canvas_res.json_data)
            st.session_state["objects"][page_idx] = jd.get("objects", [])
        except Exception:
            pass

# --- Main
col_left, col_right = st.columns([0.33, 0.67])

with col_left:
    st.header("Pages")
    if uploaded:
        try:
            idx: OCRIndex = index_pdf_bytes(uploaded.read(), min_chars_in_line=min_chars, line_merge_tol=merge_tol)
            st.session_state["ocr_idx"] = idx
            st.success(f"Indexed {len(idx.pages)} page(s).")
        except Exception as e:
            st.error(f"OCR index failed: {e}")
    else:
        st.info("Upload a PDF to begin.")

    if st.session_state["ocr_idx"]:
        pages = st.session_state["ocr_idx"].pages
        st.session_state["cur_page"] = st.number_input("Page", 1, len(pages), 1, 1) - 1

        # Build tree (line list)
        pg_dict = page_to_dict(pages[st.session_state["cur_page"]])
        selected_line = _build_tree(pg_dict)
    else:
        pages = []
        selected_line = None

with col_right:
    st.header("Page Viewer / Annotator")

    if not pages:
        st.info("Waiting for PDF…")
    else:
        pg = pages[st.session_state["cur_page"]]
        bg = _png_to_bgimg(pg.image_bytes) if pg.image_bytes else Image.new("RGB", (800,1000), "white")
        pg_dict = page_to_dict(pg)

        # Build initial rects (highlight selected)
        stored = st.session_state["objects"].get(st.session_state["cur_page"])
        if stored is None:
            rects = []
            for ln in pg_dict.get("lines", []):
                rects.append(_to_canvas_rect(ln, pg.width, pg.height,
                                             active=(selected_line and ln["id"]==selected_line["id"])))
        else:
            rects = stored

        canvas_res = _draw_canvas(pg_dict, rects, bg)
        _persist_canvas(st.session_state["cur_page"], canvas_res)

        # Two-way link: clicking a line re-renders with active style next rerun.
        if selected_line:
            st.caption(f"Selected: {selected_line['id']} — “{selected_line.get('text','')[:120]}”")

# Save current page objects to disk (optional hook)
out_dir = Path("data/ocr_cache")
out_dir.mkdir(parents=True, exist_ok=True)
if st.session_state.get("ocr_idx"):
    pid = st.session_state["cur_page"]
    objs = st.session_state["objects"].get(pid, [])
    (out_dir / f"page_{pid+1:03d}.json").write_text(json.dumps(objs, indent=2))
PY


############################################
# scripts/nuke_and_build.py
############################################
cat > scripts/nuke_and_build.py << 'PY'
import shutil, json
from pathlib import Path

ROOT = Path(".")
for p in ["data/ocr_cache"]:
    d = ROOT / p
    if d.exists():
        shutil.rmtree(d)
        print("Removed", d)
    d.mkdir(parents=True, exist_ok=True)
    print("Created", d)

print("Done.")
PY


############################################
# README.md
############################################
cat > README.md << 'MD'
# Lexi — OCR Tree & Canvas (HITL)

This drop focuses on **reliable bounding boxes + lassoing + HITL** without Tesseract.

## What’s inside

- `app/core/ocr_indexer.py` — indexes PDFs into pages/lines with bboxes using **pdfplumber** (preferred) and falls back to **PyMuPDF** if available. Produces a clean dict structure the UI consumes.
- `ui/ocr_tree_canvas.py` — Streamlit UI: **Tree ⇄ Canvas two-way linking**, move/resize boxes, save to `data/ocr_cache/page_XXX.json`.
- `scripts/nuke_and_build.py` — reset caches.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit streamlit-drawable-canvas pdfplumber pillow
# (Optional) If you want PyMuPDF fallback:
# pip install pymupdf
