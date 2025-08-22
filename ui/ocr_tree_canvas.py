# ui/ocr_tree_canvas.py
# Word→Line→Block tree with interactive canvas:
# - Two-way linking (tree <-> canvas)
# - Move/resize selected box; save with Undo/Redo
# - Show all line boxes; hit-test clicks to select

import io, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw

import streamlit as st
from streamlit_drawable_canvas import st_canvas

try:
    import fitz  # PyMuPDF
except Exception:
    st.error("PyMuPDF not available. Install with: pip install 'pymupdf<1.24'")
    st.stop()

BBox = Tuple[float, float, float, float]  # page space (x0,y0,x1,y1)

@dataclass
class Word:
    bbox: BBox
    text: str

@dataclass
class Line:
    bbox: BBox
    text: str
    words: List[Word]

@dataclass
class Block:
    bbox: BBox
    lines: List[Line]

@dataclass
class PageModel:
    width: int
    height: int
    blocks: List[Block]
    image_png: bytes  # rendered at fixed zoom


# ---------- Geometry helpers ----------

def _render_page_png(doc: "fitz.Document", page_index: int, zoom: float = 2.0) -> bytes:
    pg = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = pg.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _to_pil(png_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

def _union(bbs: List[BBox]) -> BBox:
    x0 = min(b[0] for b in bbs); y0 = min(b[1] for b in bbs)
    x1 = max(b[2] for b in bbs); y1 = max(b[3] for b in bbs)
    return (x0,y0,x1,y1)

def _scale_to_img(bb: BBox, page_wh: Tuple[int,int], img_wh: Tuple[int,int]) -> BBox:
    pw,ph = page_wh; iw,ih = img_wh
    sx,sy = iw/float(pw), ih/float(ph)
    x0,y0,x1,y1 = bb
    return (x0*sx, y0*sy, x1*sx, y1*sy)

def _scale_to_page(bb: BBox, page_wh: Tuple[int,int], img_wh: Tuple[int,int]) -> BBox:
    iw,ih = img_wh; pw,ph = page_wh
    sx,sy = pw/float(iw), ph/float(ih)
    x0,y0,x1,y1 = bb
    return (x0*sx, y0*sy, x1*sx, y1*sy)

def _contains(bb: BBox, x: float, y: float) -> bool:
    return (bb[0] <= x <= bb[2]) and (bb[1] <= y <= bb[3])


# ---------- Page parsing (words -> lines -> blocks) ----------

def _words_for_page(pg: "fitz.Page") -> List[Word]:
    out: List[Word] = []
    for w in pg.get_text("words") or []:
        x0,y0,x1,y1,text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), w[4]
        if text.strip():
            out.append(Word(bbox=(x0,y0,x1,y1), text=text))
    out.sort(key=lambda w: (round(w.bbox[1],1), w.bbox[0]))
    return out

def _cluster_lines(words: List[Word], y_tol: float = 4.0) -> List[List[Word]]:
    lines: List[List[Word]] = []
    for w in words:
        placed = False
        for ln in lines:
            ly0, ly1 = ln[0].bbox[1], ln[0].bbox[3]
            wy0, wy1 = w.bbox[1], w.bbox[3]
            if (wy0 <= ly1 + y_tol) and (wy1 >= ly0 - y_tol):
                ln.append(w); placed=True; break
        if not placed:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda ww: ww.bbox[0])
    return lines

def _cluster_blocks(lines: List[List[Word]], x_gap: float=40.0, y_gap: float=16.0) -> List[List[List[Word]]]:
    blocks: List[List[List[Word]]] = []
    for ln in lines:
        placed=False
        ln_bb = _union([w.bbox for w in ln])
        for blk in blocks:
            last_ln = blk[-1]
            last_bb = _union([w.bbox for w in last_ln])
            same_col = not (ln_bb[2] < last_bb[0]-x_gap or ln_bb[0] > last_bb[2]+x_gap)
            near_y = abs(ln_bb[1]-last_bb[3]) < y_gap or abs(last_bb[1]-ln_bb[3]) < y_gap
            if same_col and near_y:
                blk.append(ln); placed=True; break
        if not placed:
            blocks.append([ln])
    return blocks

def _page_model(pg: "fitz.Page") -> PageModel:
    pw,ph = int(pg.rect.width), int(pg.rect.height)
    words = _words_for_page(pg)
    line_groups = _cluster_lines(words)
    block_groups = _cluster_blocks(line_groups)

    blocks: List[Block] = []
    for g in block_groups:
        line_objs: List[Line] = []
        for ln in g:
            ln_bb = _union([w.bbox for w in ln])
            txt = " ".join(w.text for w in ln)
            line_objs.append(Line(bbox=ln_bb, text=txt, words=[Word(bbox=w.bbox, text=w.text) for w in ln]))
        blk_bb = _union([l.bbox for l in line_objs])
        blocks.append(Block(bbox=blk_bb, lines=line_objs))
    png = _render_page_png(pg.parent, pg.number, zoom=2.0)
    return PageModel(width=pw, height=ph, blocks=blocks, image_png=png)


# ---------- State ----------

st.set_page_config(page_title="LexiGraph – Editable OCR Tree", layout="wide")
st.title("Editable OCR Tree with Two-Way Canvas Linking")

if "pages" not in st.session_state: st.session_state["pages"] = None
if "sel" not in st.session_state:   st.session_state["sel"] = {"page":0, "block":None, "line":None}
# per-line edit history: key=(p,b,l) -> {"stack":[bbox0,bbox1,...], "ptr": int}
if "history" not in st.session_state: st.session_state["history"] = {}

def _push_history(key, bbox):
    h = st.session_state["history"].setdefault(key, {"stack":[], "ptr":-1})
    # truncate redo trail if any
    if h["ptr"] < len(h["stack"])-1:
        h["stack"] = h["stack"][:h["ptr"]+1]
    h["stack"].append(tuple(bbox))
    h["ptr"] += 1

def _can_undo(key):
    h = st.session_state["history"].get(key); 
    return bool(h) and h["ptr"] > 0

def _can_redo(key):
    h = st.session_state["history"].get(key); 
    return bool(h) and h["ptr"] < len(h["stack"])-1

def _current_bbox(key) -> Optional[BBox]:
    h = st.session_state["history"].get(key)
    if not h or h["ptr"] < 0: return None
    return h["stack"][h["ptr"]]

def _undo(key):
    h = st.session_state["history"].get(key)
    if _can_undo(key): h["ptr"] -= 1

def _redo(key):
    h = st.session_state["history"].get(key)
    if _can_redo(key): h["ptr"] += 1


# ---------- Upload / build models ----------

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    try:
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        models = [_page_model(doc.load_page(i)) for i in range(doc.page_count)]
        st.session_state["pages"] = models
        st.session_state["sel"] = {"page":0, "block":None, "line":None}
        st.session_state["history"] = {}
        st.success(f"Loaded {len(models)} page(s).")
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")

pages: Optional[List[PageModel]] = st.session_state["pages"]
if not pages:
    st.info("Upload a PDF to begin."); st.stop()

left, right = st.columns([0.35,0.65])

with left:
    st.subheader("Document Tree")
    p = st.number_input("Page", 0, len(pages)-1, value=int(st.session_state["sel"]["page"]), step=1)
    st.session_state["sel"]["page"] = int(p)
    page = pages[int(p)]
    st.caption(f"Page size: {page.width} × {page.height}")

    blk_opts = [f"Block {i}  bbox={tuple(int(v) for v in b.bbox)}  (lines={len(b.lines)})" for i,b in enumerate(page.blocks)]
    b = st.selectbox("Block", options=[None]+list(range(len(page.blocks))),
                     format_func=lambda i: "—" if i is None else blk_opts[i],
                     index=(0 if st.session_state["sel"]["block"] is None else st.session_state["sel"]["block"]+1))
    st.session_state["sel"]["block"] = b

    l = None
    if b is not None:
        lines = page.blocks[b].lines
        line_opts = [f"{i}: {ln.text[:90]}" for i,ln in enumerate(lines)]
        l = st.selectbox("Line", options=[None]+list(range(len(lines))),
                         format_func=lambda i: "—" if i is None else line_opts[i],
                         index=(0 if st.session_state["sel"]["line"] is None else st.session_state["sel"]["line"]+1))
        st.session_state["sel"]["line"] = l
        if l is not None:
            ln = lines[l]
            st.caption(f"Line bbox (PDF coords): {tuple(round(v,1) for v in ln.bbox)}")
            st.code(ln.text)

    st.divider()
    st.subheader("Edit History (selected)")
    sel_key = None
    if b is not None and l is not None:
        sel_key = (int(p), int(b), int(l))
        st.json(st.session_state["history"].get(sel_key, {"stack":[],"ptr":-1}))
        c1,c2,c3 = st.columns(3)
        with c1:
            if st.button("Undo", disabled=not _can_undo(sel_key)): _undo(sel_key); st.rerun()
        with c2:
            if st.button("Redo", disabled=not _can_redo(sel_key)): _redo(sel_key); st.rerun()
        with c3:
            if st.button("Clear edits", disabled=(sel_key not in st.session_state["history"])):
                st.session_state["history"].pop(sel_key, None); st.rerun()
    else:
        st.caption("Pick a Block + Line to see history.")


with right:
    st.subheader("Page Viewer")
    show_all = st.checkbox("Show ALL line boxes on page", value=True)
    mode = st.radio(
        "Canvas mode",
        ["Select (click)", "Move/Resize (transform)", "Draw (rect)"],
        horizontal=True
    )

    # base image
    base = _to_pil(page.image_png).copy()
    iw, ih = base.size
    page_wh = (page.width, page.height)

    # Build fabric rectangles for all lines (so they are selectable/movable)
    fabric_objects = []
    for bi, blk in enumerate(page.blocks):
        for li, ln in enumerate(blk.lines):
            x0,y0,x1,y1 = _scale_to_img(ln.bbox, page_wh, (iw,ih))
            fabric_objects.append({
                "type": "rect",
                "left": x0, "top": y0, "width": (x1-x0), "height": (y1-y0),
                "fill": "rgba(0,0,0,0.0)",
                "stroke": "#ff0000" if show_all else "#ffaaaa",
                "strokeWidth": 2,
                "name": f"L{bi}.{li}",   # we use name to resolve back to line
                "selectable": True,
                "hasControls": True,
                "hasBorders": True
            })

    # If a selected line has an edited bbox, draw a bold overlay (background image)
    overlays=[]
    if sel_key:
        cur = _current_bbox(sel_key)
        if cur is not None:
            overlays.append((cur, "EDIT"))
    if overlays:
        img2 = base.copy()
        draw = ImageDraw.Draw(img2)
        for bb,_ in overlays:
            sx0,sy0,sx1,sy1 = _scale_to_img(bb, page_wh, (iw,ih))
            draw.rectangle((sx0,sy0,sx1,sy1), outline=(0,200,0), width=3)
        base = img2

    # Draw ALL boxes as visual background if desired (not interactive; the fabric shapes are)
    if show_all:
        img3 = base.copy()
        draw = ImageDraw.Draw(img3)
        for bi, blk in enumerate(page.blocks):
            for li, ln in enumerate(blk.lines):
                sx0,sy0,sx1,sy1 = _scale_to_img(ln.bbox, page_wh, (iw,ih))
                draw.rectangle((sx0,sy0,sx1,sy1), outline=(255,0,0), width=1)
                draw.rectangle((sx0, sy0-14, sx0+28, sy0), fill=(255,0,0))
                draw.text((sx0+3, sy0-12), f"{bi}.{li}", fill=(255,255,255))
        base = img3

    # Canvas mode wiring
    drawing_mode = None
    if mode == "Select (click)":
        drawing_mode = "point"
    elif mode == "Move/Resize (transform)":
        drawing_mode = "transform"
    elif mode == "Draw (rect)":
        drawing_mode = "rect"

    canv = st_canvas(
        background_image=base,
        height=ih,
        width=iw,
        drawing_mode=drawing_mode,
        stroke_color="#00aa00",
        fill_color="rgba(0,170,0,0.18)",
        stroke_width=2,
        initial_drawing={"objects": fabric_objects, "background": None},
        update_streamlit=True,
        key=f"canvas_{int(p)}",
    )
    # --- Two-way: click selects line (robust center + nearest fallback) ---
	def _center(bb: BBox) -> Tuple[float, float]:
	    return ((bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5)
	
	def _nearest_line(px: float, py: float, pg: PageModel) -> Tuple[int, int] | None:
	    best = None
	    best_d2 = float("inf")
	    for bi, blk in enumerate(pg.blocks):
	        for li, ln in enumerate(blk.lines):
	            cx, cy = _center(ln.bbox)
	            d2 = (cx - px) * (cx - px) + (cy - py) * (cy - py)
	            inside = _contains(ln.bbox, px, py)
	            # Prefer containing box; else nearest
	            score = (0.0 if inside else 1.0) * 1e12 + d2
	            if score < best_d2:
	                best_d2 = score
	                best = (bi, li)
	    return best

    # --- Two-way: click selects line ---
    if mode == "Select (click)" and canv.json_data:
        # last object is the click 'point' (small circle); get its center
        objs = canv.json_data.get("objects", [])
        if objs:
            last = objs[-1]
            if last.get("type") in ("circle", "triangle", "rect", "path", "line"):  # 'point' becomes a small circle/tri
                cx = float(last.get("left", 0.0))
                cy = float(last.get("top", 0.0))
                # hit-test against line boxes
                px,py = _scale_to_page((cx,cy,cx,cy), page_wh, (iw,ih))[:2]
                chosen = None
                for bi, blk in enumerate(page.blocks):
                    for li, ln in enumerate(blk.lines):
                        if _contains(ln.bbox, px, py):
                            chosen = (bi,li); break
                    if chosen: break
                if chosen is not None:
                    st.session_state["sel"]["block"] = int(chosen[0])
                    st.session_state["sel"]["line"]  = int(chosen[1])
                    st.rerun()

    # --- Save moved shapes (transform mode) ---
    save_changes = st.button("Save moved/edited boxes", disabled=(mode!="Move/Resize (transform)"))
    if save_changes and canv.json_data:
        changed = 0
        for obj in canv.json_data.get("objects", []):
            if obj.get("type") == "rect" and obj.get("name", "").startswith("L"):
                # map fabric rect -> page bbox
                left = float(obj.get("left",0.0)); top = float(obj.get("top",0.0))
                w = float(obj.get("width",0.0))*float(obj.get("scaleX",1.0))
                h = float(obj.get("height",0.0))*float(obj.get("scaleY",1.0))
                img_bb = (left, top, left+w, top+h)
                page_bb = _scale_to_page(img_bb, page_wh, (iw,ih))
                # identify which line
                name = obj.get("name")
                try:
                    bi, li = name[1:].split(".")
                    key = (int(p), int(bi), int(li))
                    _push_history(key, page_bb)
                    changed += 1
                except Exception:
                    pass
        st.success(f"Saved {changed} box change(s).")

    # --- Draw new rect and bind to selected line (rect mode) ---
    if mode == "Draw (rect)":
        new_rect = None
        if canv.json_data:
            # find the last rect drawn that doesn't have "name"
            for obj in reversed(canv.json_data.get("objects", [])):
                if obj.get("type") == "rect" and not obj.get("name"):
                    left = float(obj.get("left",0.0)); top = float(obj.get("top",0.0))
                    w = float(obj.get("width",0.0))*float(obj.get("scaleX",1.0))
                    h = float(obj.get("height",0.0))*float(obj.get("scaleY",1.0))
                    new_rect = (left,top,left+w,top+h)
                    break
        c1,c2 = st.columns(2)
        with c1:
            bind_ok = st.button("Bind last drawn rectangle to SELECTED line",
                                disabled=(new_rect is None or st.session_state["sel"]["block"] is None or st.session_state["sel"]["line"] is None))
        with c2:
            st.caption("Pick Block+Line on the left, draw a rect, then click Bind.")
        if bind_ok and new_rect:
            key = (int(p), int(st.session_state["sel"]["block"]), int(st.session_state["sel"]["line"]))
            _push_history(key, _scale_to_page(new_rect, page_wh, (iw,ih)))
            st.success(f"Bound new rectangle to {key}")

    # --- Export current page's edited bboxes ---
    export_page = st.button("Export edited bboxes for this page")
    if export_page:
        out = {}
        for (pp, bb, ll), hist in st.session_state["history"].items():
            if pp == int(p) and hist["ptr"] >= 0:
                out[f"{pp}-{bb}-{ll}"] = list(hist["stack"][hist["ptr"]])
        st.download_button("Download page_edits.json",
                           data=json.dumps(out, indent=2).encode("utf-8"),
                           file_name=f"page_{int(p)}_edits.json",
                           mime="application/json")
