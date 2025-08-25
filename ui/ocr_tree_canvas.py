# ui/ocr_tree_canvas.py
# -----------------------------------------------------------------------------
# OCR-like PDF line boxing & HITL editor without hard OCR deps.
# - Extracts line boxes using PyMuPDF if available; else approximates with PDFMiner.
# - Renders page images with PyMuPDF if available; else white canvas fallback.
# - Left pane tree (Page -> Lines), right pane canvas with boxes (transformable).
# - Two-way selection sync. Lasso add. Undo/Redo. NMS & filters to reduce box spam.
# -----------------------------------------------------------------------------

import io
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw

import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Optional OCR fallback (doctr) for image-only pages
try:
    from doctr.models import ocr_predictor
    from doctr.io import DocumentFile
    DOCTR_OK = True
except Exception:
    DOCTR_OK = False


# Optional: PyMuPDF for robust page rendering / text boxes
try:
    import fitz  # type: ignore
    FITZ_OK = True
except Exception:
    FITZ_OK = False

# ------------------------- Utils & Geometry ----------------------------------

def iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax1-ax0)) * max(0.0, (ay1-ay0))
    area_b = max(0.0, (bx1-bx0)) * max(0.0, (by1-by0))
    denom = area_a + area_b - inter + 1e-9
    return inter / denom

def nms_boxes(boxes: List[Tuple[float,float,float,float]], scores: List[float], iou_thresh: float=0.4):
    """Non-maximum suppression. Returns indices to keep."""
    if not boxes:
        return []
    idxs = np.argsort(np.array(scores))[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious < iou_thresh]
    return keep

def clamp(v, a, b):
    return max(a, min(b, v))

# ------------------------- Data Models ---------------------------------------

@dataclass
class Line:
    id: int
    text: str
    bbox: Tuple[float,float,float,float]  # (x0,y0,x1,y1) in page pixel coords
    score: float = 1.0

@dataclass
class PageData:
    page_index: int
    width: int
    height: int
    image_bytes: Optional[bytes]  # PNG bytes
    lines: List[Line]

# ------------------------- Extraction Backends -------------------------------

def _render_page_image_fitz(doc: "fitz.Document", page_index: int, scale: float=2.0) -> Tuple[bytes, int, int]:
    page = doc.load_page(page_index)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    bts = io.BytesIO()
    img.save(bts, format="PNG")
    return bts.getvalue(), img.width, img.height

def _extract_lines_fitz(doc: "fitz.Document", page_index: int, scale: float=2.0,
                        min_chars: int=3, min_h: int=10) -> List[Line]:
    """Use PyMuPDF's text blocks -> lines with coordinates scaled to rendered image."""
    page = doc.load_page(page_index)
    # Get raw text lines
    # Use "rawdict" for detailed structure
    data = page.get_text("rawdict")
    # Rendering scale
    mat = fitz.Matrix(scale, scale)
    lines: List[Line] = []
    line_id = 0
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 0 means text block
            continue
        for line in block.get("lines", []):
            # Combine spans
            spans = line.get("spans", [])
            if not spans: 
                continue
            txt = "".join([s.get("text", "") for s in spans]).strip()
            if len(txt) < min_chars:
                continue
            # bbox for the line
            # Spans have bbox; we can union them
            x0s, y0s, x1s, y1s = [], [], [], []
            for s in spans:
                x0,y0,x1,y1 = s.get("bbox", [0,0,0,0])
                x0s.append(x0); y0s.append(y0); x1s.append(x1); y1s.append(y1)
            if not x0s: 
                continue
            bx0, by0 = min(x0s), min(y0s)
            bx1, by1 = max(x1s), max(y1s)
            # scale bbox to image space
            rect = fitz.Rect(bx0, by0, bx1, by1) * mat
            w = rect.x1 - rect.x0
            h = rect.y1 - rect.y0
            if h < min_h or w < 5:
                continue
            lines.append(Line(id=line_id, text=txt, bbox=(rect.x0, rect.y0, rect.x1, rect.y1), score=float(w*h)))
            line_id += 1
    # NMS to reduce duplicates/overlaps
    if lines:
        boxes = [ln.bbox for ln in lines]
        scores = [ln.score for ln in lines]
        keep = nms_boxes(boxes, scores, iou_thresh=0.3)
        lines = [lines[i] for i in keep]
    return lines

def _render_blank(width: int=1000, height: int=1300) -> Tuple[bytes, int, int]:
    img = Image.new("RGB", (width, height), "white")
    bts = io.BytesIO(); img.save(bts, format="PNG")
    return bts.getvalue(), width, height

def _ocr_lines_doctr(pil_img: Image.Image, min_chars: int = 3, min_h: int = 10) -> List[Line]:
    """
    Run doctr OCR on a PIL image and return Line[] in image pixel coords.
    """
    if not DOCTR_OK:
        return []
    # doctr expects numpy RGB
    img_np = np.array(pil_img.convert("RGB"))
    predictor = ocr_predictor(pretrained=True)  # CPU by default
    doc = DocumentFile.from_images([img_np])
    result = predictor(doc)
    lines: List[Line] = []
    li = 0
    # doctr gives relative boxes (0..1). We'll union words into lines.
    for page in result.pages:
        # page.blocks -> lines -> words
        for block in page.blocks:
            for line in block.lines:
                # join words and union their boxes
                txt = " ".join([w.value for w in line.words]).strip()
                if len(txt) < min_chars:
                    continue
                xs, ys, xe, ye = [], [], [], []
                for w in line.words:
                    # w.geometry: ((x0,y0),(x1,y1)) normalized to 0..1
                    (x0, y0), (x1, y1) = w.geometry
                    xs.append(x0); ys.append(y0); xe.append(x1); ye.append(y1)
                if not xs:
                    continue
                # convert to pixels
                W, H = pil_img.size
                bx0 = float(min(xs) * W); by0 = float(min(ys) * H)
                bx1 = float(max(xe) * W);  by1 = float(max(ye) * H)
                if (by1 - by0) < min_h:
                    continue
                lines.append(Line(id=li, text=txt, bbox=(bx0, by0, bx1, by1), score=(bx1-bx0)*(by1-by0)))
                li += 1
    # NMS to reduce overlaps
    if lines:
        boxes = [ln.bbox for ln in lines]
        scores = [ln.score for ln in lines]
        keep = nms_boxes(boxes, scores, iou_thresh=0.3)
        lines = [lines[i] for i in keep]
    return lines

def extract_pages(pdf_bytes: bytes, use_fitz: bool=True, scale: float=2.0,
                  min_chars: int=3, min_line_h: int=10,
                  max_lines_per_page: int=400, use_ocr_fallback: bool=True) -> List[PageData]:
    """Best-effort page renderer + line boxes. Uses PyMuPDF if available."""
    pages: List[PageData] = []
    if use_fitz and FITZ_OK:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pi in range(doc.page_count):
                try:
                    img_b, w, h = _render_page_image_fitz(doc, pi, scale=scale)
                except Exception:
                    img_b, w, h = _render_blank()
# inside the per-page loop in extract_pages(...)

                try:
                    lines = _extract_lines_fitz(doc, pi, scale=scale, min_chars=min_chars, min_h=min_line_h)
                except Exception:
                    lines = []
                
                # NEW: OCR fallback if no lines and toggle is on
                if (not lines) and use_ocr_fallback and FITZ_OK:
                    # we already rendered the page image above; reuse it
                    if 'img_b' not in locals():
                        try:
                            img_b, w, h = _render_page_image_fitz(doc, pi, scale=scale)
                        except Exception:
                            img_b, w, h = _render_blank()
                    pil_img = Image.open(io.BytesIO(img_b)).convert("RGB")
                    try:
                        lines = _ocr_lines_doctr(pil_img, min_chars=min_chars, min_h=min_line_h)
                    except Exception:
                        lines = []

                # cap lines
                if len(lines) > max_lines_per_page:
                    # keep largest areas first
                    lines = sorted(lines, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[:max_lines_per_page]
                pages.append(PageData(page_index=pi, width=w, height=h, image_bytes=img_b, lines=lines))
            return pages
        except Exception as e:
            st.warning(f"PyMuPDF backend failed, fallback to blank render. ({e})")

    # Fallback: blank pages, no lines (still editable with lasso)
    # We’ll try to count pages via a very light parse using pdf header heuristics
    # but if unknown, just render 1 page.
    img_b, w, h = _render_blank()
    pages.append(PageData(page_index=0, width=w, height=h, image_bytes=img_b, lines=[]))
    return pages

# ------------------------- Session & History ---------------------------------

def ensure_state():
    if "pdf_name" not in st.session_state: st.session_state.pdf_name = None
    if "pages" not in st.session_state: st.session_state.pages: List[PageData] = []
    if "cur_page" not in st.session_state: st.session_state.cur_page = 0
    if "selected_line_id" not in st.session_state: st.session_state.selected_line_id = 0
    if "history" not in st.session_state: st.session_state.history = []  # list of (pages json)
    if "future" not in st.session_state: st.session_state.future = []

def push_history():
    # compact history to last 20 steps
    snapshot = json.dumps([[asdict(ln) for ln in pg.lines] for pg in st.session_state.pages])
    st.session_state.history.append(snapshot)
    if len(st.session_state.history) > 20:
        st.session_state.history.pop(0)
    # clear redo
    st.session_state.future.clear()

def undo():
    if not st.session_state.history:
        return
    current = json.dumps([[asdict(ln) for ln in pg.lines] for pg in st.session_state.pages])
    st.session_state.future.append(current)
    snap = st.session_state.history.pop()
    restore = json.loads(snap)
    for pi, line_rows in enumerate(restore):
        if pi >= len(st.session_state.pages): break
        recs = []
        for r in line_rows:
            recs.append(Line(id=int(r["id"]), text=r["text"], bbox=tuple(r["bbox"]), score=float(r.get("score",1.0))))
        st.session_state.pages[pi].lines = recs

def redo():
    if not st.session_state.future:
        return
    push_history()
    snap = st.session_state.future.pop()
    restore = json.loads(snap)
    for pi, line_rows in enumerate(restore):
        if pi >= len(st.session_state.pages): break
        recs = []
        for r in line_rows:
            recs.append(Line(id=int(r["id"]), text=r["text"], bbox=tuple(r["bbox"]), score=float(r.get("score",1.0))))
        st.session_state.pages[pi].lines = recs

# ------------------------- Canvas Builders -----------------------------------

def page_image(pg: PageData) -> Image.Image:
    if pg.image_bytes:
        return Image.open(io.BytesIO(pg.image_bytes)).convert("RGB")
    return Image.new("RGB", (pg.width, pg.height), "white")

def to_fabric_rect(line: Line, stroke: str="#e11", fill="rgba(255,0,0,0.10)", selected=False, label=None) -> Dict[str,Any]:
    x0,y0,x1,y1 = line.bbox
    obj = {
        "type": "rect", "left": float(x0), "top": float(y0),
        "width": float(x1-x0), "height": float(y1-y0),
        "fill": fill, "stroke": stroke, "strokeWidth": 2,
        "selectable": True, "hasControls": True, "hasBorders": True,
        "data": {"line_id": line.id}
    }
    if selected:
        obj["stroke"] = "#0a0"
        obj["fill"] = "rgba(0,255,0,0.10)"
    if label is not None:
        # add a small label by drawing on background later; canvas text objects
        # get heavy to manage, so we draw numbers into BG image (controlled below)
        obj["data"]["label"] = int(label)
    return obj

def fabric_to_bbox(obj: Dict[str, Any]) -> Tuple[float,float,float,float]:
    x = float(obj.get("left", 0.0))
    y = float(obj.get("top", 0.0))
    w = float(obj.get("width", 0.0))
    h = float(obj.get("height", 0.0))
    return (x, y, x+w, y+h)

def draw_numbers_on_image(img: Image.Image, lines: List[Line], number_map: Dict[int,int]):
    """Draw small circled numbers at top-left of each line bbox."""
    if not number_map: 
        return img
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for ln in lines:
        if ln.id not in number_map:
            continue
        n = number_map[ln.id]
        x0,y0,_,_ = ln.bbox
        r = 10
        dr.ellipse((x0-2*r, y0-2*r, x0, y0), outline=(255,0,0), width=2)
        dr.text((x0-2*r+4, y0-2*r+2), str(n), fill=(0,0,0))
    return im

# ------------------------- App ------------------------------------------------

st.set_page_config(page_title="LexiGraph — OCR Tree & Canvas", layout="wide")
st.title("📄 LexiGraph — OCR Tree, Canvas & HITL")

ensure_state()

with st.sidebar:
    st.header("Load")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    scale = st.slider("Render scale (fitz)", 1.0, 3.0, 2.0, 0.1)
    min_chars = st.slider("Min chars per line", 1, 10, 3, 1)
    min_h = st.slider("Min line height (px)", 6, 30, 10, 1)
    max_lines = st.slider("Max boxes / page", 50, 1000, 400, 10)
    use_ocr_fallback = st.checkbox("Use OCR fallback if no embedded text", value=True)
    if st.button("Index / Rebuild"):
        if not pdf:
            st.warning("Upload a PDF first.")
        else:
            st.session_state.pdf_name = pdf.name
            st.info("Extracting page images & line boxes…")
            pages = extract_pages(pdf.read(), use_fitz=True, scale=scale,
                                  min_chars=min_chars, min_line_h=min_h,
                                  max_lines_per_page=max_lines)
            st.session_state.pages = pages
            st.session_state.cur_page = 0
            st.session_state.selected_line_id = 0
            push_history()
            st.success(f"Indexed {len(pages)} page(s).")

    st.header("History")
    cols = st.columns(2)
    if cols[0].button("Undo"):
        undo()
    if cols[1].button("Redo"):
        redo()

    st.header("Lasso / Add")
    st.write("• Switch canvas to **Rect (lasso)** to draw a new box.\n"
             "• Use **Bind Lasso → Selected line** to attach your last drawn box.")
    st.session_state.setdefault("last_lasso_box", None)
    if st.button("Bind Lasso → Selected line"):
        if st.session_state.get("last_lasso_box") is None:
            st.warning("Draw a new rectangle in Rect mode first.")
        else:
            pg = st.session_state.pages[st.session_state.cur_page]
            li = st.session_state.selected_line_id or 0
            # if no lines yet, create a new line record
            if li < 0 or li >= len(pg.lines):
                new_id = (max([ln.id for ln in pg.lines]) + 1) if pg.lines else 0
                pg.lines.append(Line(id=new_id, text="", bbox=st.session_state.last_lasso_box, score=1.0))
                st.session_state.selected_line_id = new_id
            else:
                # update existing line bbox
                ln = pg.lines[li]
                ln.bbox = st.session_state.last_lasso_box
            push_history()
            st.success("Bound lasso rectangle.")

# --- Layout: Tree (left) & Canvas (right)
col_tree, col_canvas = st.columns([0.35, 0.65], gap="large")

# LEFT: Tree / selection
with col_tree:
    st.header("Document Tree")
    if not st.session_state.pages:
        st.info("Upload a PDF and click **Index / Rebuild** in the sidebar.")
    else:
        st.caption(f"File: {st.session_state.pdf_name or '(untitled)'}")
        # Page picker
        page_options = [f"Page {p.page_index+1} ({len(p.lines)} lines)" for p in st.session_state.pages]
        cur_label = page_options[st.session_state.cur_page]
        new_page_label = st.selectbox("Page", options=page_options, index=st.session_state.cur_page, key="page_select")
        st.session_state.cur_page = page_options.index(new_page_label)

        # Lines picker (index-based, safe)
        pg = st.session_state.pages[st.session_state.cur_page]
        lines = pg.lines
        st.caption(f"Current page size: {pg.width} × {pg.height}")
        if not lines:
            st.warning("No lines detected on this page. You can draw rectangles in Rect mode to add boxes.")
        else:
            opt_ids = list(range(len(lines)))
            # lines: list[dict] already computed for current page

            def fmt_line(idx: int) -> str:
                """Human label for a line id -> '[#] first 60 chars'."""
                try:
                    li = lines[idx] if 0 <= idx < len(lines) else {}
                except Exception:
                    li = {}
                txt = (li.get("text") or "").strip()
                first = txt.splitlines()[0] if txt else ""
                return f"[{idx+1}] {first[:60]}"
            
            # options are 0..N-1
            opt_ids = list(range(len(lines)))
            
            # previous selection; clamp safely
            prev_sel = int(st.session_state.get("selected_line_id", 0))
            sel_default = prev_sel if 0 <= prev_sel < len(opt_ids) else (0 if opt_ids else 0)
            
            if not opt_ids:
                st.info("No lines detected on this page. Try OCR fallback or adjust the thresholds.")
                chosen_idx = None
            else:
                chosen_idx = st.selectbox(
                    "Line",
                    options=opt_ids,
                    index=sel_default,
                    format_func=fmt_line,
                    key="line_select",
                )
                # Keep session state in sync
                if chosen_idx is not None:
                    st.session_state["selected_line_id"] = int(chosen_idx)


            # clamp selection
            sel_default = st.session_state.selected_line_id
            if sel_default is None or sel_default < 0 or sel_default >= len(opt_ids):
                sel_default = 0
            chosen_idx = st.selectbox("Line", options=opt_ids, index=sel_default, format_func=fmt_line, key="line_select")
            if chosen_idx is not None:
                st.session_state.selected_line_id = int(chosen_idx)

            st.subheader("Selected Text")
            st.text_area("",
                         value=(lines[st.session_state.selected_line_id].text if lines else ""),
                         height=140)

# RIGHT: Canvas
with col_canvas:
    st.header("Page Viewer / Annotator")

    if not st.session_state.pages:
        st.info("Upload and index a PDF first.")
    else:
        pg = st.session_state.pages[st.session_state.cur_page]
        bg = page_image(pg)

        # Optional numbered labels on image
        number_map = {ln.id: idx+1 for idx, ln in enumerate(pg.lines[:200])}
        bg_num = draw_numbers_on_image(bg, pg.lines, number_map)

        # Build initial fabric
        init_objects = []
        for i, ln in enumerate(pg.lines):
            init_objects.append(to_fabric_rect(
                ln,
                selected=(i == st.session_state.selected_line_id),
                label=number_map.get(ln.id)
            ))

        # Canvas mode switch
        mode = st.radio("Mode", ["Transform", "Rect (lasso add)"], horizontal=True)
        drawing_mode = "transform" if mode == "Transform" else "rect"

        # Render canvas
        canvas_res = st_canvas(
            background_image=bg_num,
            drawing_mode=drawing_mode,
            key="canvas",
            height=pg.height if pg.height <= 1800 else 1800,  # cap height for browser perf
            width=pg.width if pg.width <= 1400 else 1400,
            initial_drawing={"version": "4.4.0", "objects": init_objects},
            update_streamlit=True,
            stroke_color="#00ff00",
            stroke_width=2,
        )

        # Capture new objects (for lasso add)
        if canvas_res.json_data:
            objs = canvas_res.json_data.get("objects", [])
            # Find any object without data.line_id -> treat last such rect as lasso
            lasso_box = None
            for o in objs:
                data = o.get("data") or {}
                if o.get("type") == "rect" and ("line_id" not in data):
                    lasso_box = fabric_to_bbox(o)
            if lasso_box is not None:
                st.session_state.last_lasso_box = lasso_box

        # If in transform mode, persist rectangle movements into our model
        if canvas_res.json_data and drawing_mode == "transform":
            # Update all known rect positions by matching line_id in object data
            objs = canvas_res.json_data.get("objects", [])
            changed = 0
            lid2obj = {}
            for o in objs:
                data = o.get("data") or {}
                if o.get("type") == "rect" and ("line_id" in data):
                    lid2obj[int(data["line_id"])] = o
            # Apply back to our page model
            for i, ln in enumerate(pg.lines):
                o = lid2obj.get(ln.id)
                if o is None:
                    continue
                new_box = fabric_to_bbox(o)
                if tuple(map(float, ln.bbox)) != tuple(map(float, new_box)):
                    ln.bbox = new_box
                    changed += 1
            if changed:
                push_history()
                st.caption(f"Updated {changed} box(es).")

        # Click-to-select behavior:
        # If the user clicked within a box area (transform mode), pick the closest line
        if canvas_res.json_data and drawing_mode == "transform":
            # Canvas library doesn’t give direct click coords; we approximate by
            # selecting the smallest-area rect that changed most recently.
            # As a practical heuristic, set selected_line_id to the first object in list
            # whose bbox contains the prior selected point — here, we fall back to centering:
            # So we instead offer a "Select by number" quick input:
            with st.expander("Quick select by number (labels on page image)", expanded=False):
                # --- Go to / jump-to control (safe defaults) ---
                cur_sel = int(st.session_state.get("selected_line_id", -1))
                num_lines = len(pg.lines) if getattr(pg, "lines", None) else 0
                max_n = max(1, num_lines)                     # at least 1
                default_val = 1 if cur_sel < 0 else min(max_n, cur_sel + 1)
                
                num = st.number_input(
                    "Go to #",
                    min_value=1,
                    max_value=max_n,
                    value=default_val,   # always in [1, max_n]
                    step=1,
                )
                # keep session in sync when there are lines
                if num_lines:
                    st.session_state["selected_line_id"] = int(num - 1)
                

        st.caption("Tip: Use **Rect (lasso add)** to draw a missing box; then click **Bind Lasso → Selected line** in sidebar.")

# ------------------------- Footer Save/Export --------------------------------

st.markdown("---")
cols = st.columns(3)
if cols[0].button("Save page boxes to JSON"):
    if not st.session_state.pages:
        st.warning("Nothing to save.")
    else:
        pg = st.session_state.pages[st.session_state.cur_page]
        payload = {
            "page_index": pg.page_index,
            "width": pg.width,
            "height": pg.height,
            "lines": [asdict(ln) for ln in pg.lines],
        }
        st.download_button("Download JSON", data=json.dumps(payload, indent=2), file_name=f"page_{pg.page_index+1}_boxes.json", mime="application/json", key="dljson")

if cols[1].button("Save all pages to JSON"):
    if not st.session_state.pages:
        st.warning("Nothing to save.")
    else:
        allp = []
        for pg in st.session_state.pages:
            allp.append({
                "page_index": pg.page_index,
                "width": pg.width,
                "height": pg.height,
                "lines": [asdict(ln) for ln in pg.lines],
            })
        st.download_button("Download ALL", data=json.dumps({"pages": allp}, indent=2), file_name=f"{(st.session_state.pdf_name or 'document').rsplit('.',1)[0]}_boxes.json", mime="application/json", key="dlall")

if cols[2].button("Reset session"):
    for k in ["pdf_name","pages","cur_page","selected_line_id","history","future","last_lasso_box"]:
        if k in st.session_state: del st.session_state[k]
    st.experimental_rerun()
