import os
from statistics import median
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app.ocr import extract_words
from app.semantics import tag_field, is_money, is_date, product_or_service

# Try PDF raster underlay
PDF_TO_IMAGE = None
try:
    from pdf2image import convert_from_path
    PDF_TO_IMAGE = "pdf2image"
except Exception:
    PDF_TO_IMAGE = None

st.set_page_config(page_title="OCR Tree Canvas", layout="wide")
st.title("OCR Tree Canvas — Viewer & Annotate")

# ---------------- helpers ----------------
def word_bbox(w):
    b = w.bbox
    return b.x0, b.y0, b.x1, b.y1

def bbox_union(boxes: List[Tuple[float,float,float,float]]) -> Optional[Tuple[float,float,float,float]]:
    if not boxes: return None
    xs0 = [b[0] for b in boxes]; ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]; ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))

def point_in_polygon(x, y, poly):
    n = len(poly); inside = False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside

def any_point_in_polygon(bbox, poly):
    x0,y0,x1,y1 = bbox
    pts = [(x0,y0),(x1,y0),(x1,y1),(x0,y1),((x0+x1)/2,(y0+y1)/2)]
    return any(point_in_polygon(px,py,poly) for (px,py) in pts)

def draw_boxes(base_img: Image.Image, boxes, outline=(0,0,0), width=1, fill=None):
    im = base_img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    for b in boxes:
        rect = [b[0], b[1], b[2], b[3]]
        if fill is not None:
            dr.rectangle(rect, outline=outline, width=width, fill=fill)
        else:
            dr.rectangle(rect, outline=outline, width=width)
    return im

def draw_word_labels(base_img: Image.Image, words, font_size=12):
    im = base_img.copy()
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for w in words:
        x0,y0,x1,y1 = word_bbox(w)
        # put label INSIDE the box to guarantee visibility
        dr.text((x0+1, y0+1), w.text, fill=(20,20,20), font=font)
    return im

# -------------- grouping: columns -> lines -> segments -> blocks --------------
def cluster_columns(words, gap_factor=2.2):
    if not words: return []
    ws = sorted(words, key=lambda w: (w.bbox.x0, w.bbox.y0))
    widths = [w.bbox.x1 - w.bbox.x0 for w in ws]
    med_w = median(widths) if widths else 10.0
    threshold = max(med_w * gap_factor, 30.0)
    clusters = [[ws[0]]]
    for w in ws[1:]:
        prev = clusters[-1][-1]
        gap = w.bbox.x0 - prev.bbox.x1
        (clusters if gap <= threshold else clusters.__iadd__([[]]))[-1].append(w)
    for c in clusters:
        c.sort(key=lambda w: (w.bbox.y0, w.bbox.x0))
    return clusters

def group_lines(words_in_col, y_tol_factor=0.7):
    if not words_in_col: return []
    heights = [w.bbox.y1 - w.bbox.y0 for w in words_in_col]
    med_h = median(heights) if heights else 10.0
    y_tol = med_h * y_tol_factor
    lines = []
    for w in words_in_col:
        placed = False
        for line in lines:
            py = line[-1].bbox.y0
            if abs(w.bbox.y0 - py) <= y_tol:
                line.append(w); placed = True; break
        if not placed: lines.append([w])
    for ln in lines: ln.sort(key=lambda w: w.bbox.x0)
    return lines

def split_line_into_segments(line_words, seg_gap_factor=2.0):
    if not line_words: return []
    widths = [w.bbox.x1 - w.bbox.x0 for w in line_words]
    med_w = median(widths) if widths else 10.0
    gap_thr = max(med_w * seg_gap_factor, 30.0)
    segments = [[line_words[0]]]
    for w in line_words[1:]:
        prev = segments[-1][-1]
        gap = w.bbox.x0 - prev.bbox.x1
        (segments if gap <= gap_thr else segments.__iadd__([[]]))[-1].append(w)
    return segments

def group_blocks(segments, v_gap_factor=1.2, x_iou_min=0.2):
    blocks = []
    if not segments: return blocks
    seg_items = []
    for seg in segments:
        x0=min(w.bbox.x0 for w in seg); y0=min(w.bbox.y0 for w in seg)
        x1=max(w.bbox.x1 for w in seg); y1=max(w.bbox.y1 for w in seg)
        seg_items.append({"words": seg, "bbox": (x0,y0,x1,y1)})
    seg_items.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    heights = [s["bbox"][3]-s["bbox"][1] for s in seg_items]
    med_h = median(heights) if heights else 12.0
    v_gap_thr = max(med_h * v_gap_factor, 20.0)
    def x_iou(a,b):
        ax0,_,ax1,_ = a; bx0,_,bx1,_ = b
        inter = max(0, min(ax1,bx1)-max(ax0,bx0))
        union = max(ax1, bx1) - min(ax0,bx0)
        return inter/union if union>0 else 0.0
    for s in seg_items:
        placed=False
        for blk in blocks:
            bx0,by0,bx1,by1 = blk["bbox"]
            sx0,sy0,sx1,sy1 = s["bbox"]
            v_gap = sy0 - by1
            if 0 <= v_gap <= v_gap_thr and x_iou(blk["bbox"], s["bbox"]) >= x_iou_min:
                blk["words"].extend(s["words"])
                blk["bbox"] = (min(bx0,sx0), min(by0,sy0), max(bx1,sx1), max(by1,sy1))
                placed=True; break
        if not placed: blocks.append({"words": list(s["words"]), "bbox": s["bbox"]})
    return blocks

# ---------------- common controls ----------------
with st.sidebar:
    st.header("Segmentation")
    col_gap = st.slider("Column gap factor", 1.5, 4.0, 2.2, 0.1)
    seg_gap = st.slider("Segment gap factor", 1.0, 4.0, 2.2, 0.1)
    v_gap   = st.slider("Vertical merge gap (blocks)", 0.8, 3.0, 1.2, 0.1)
    x_iou_min = st.slider("Min horizontal overlap (blocks)", 0.0, 0.9, 0.2, 0.05)
    st.header("Layers")
    layer_words    = st.checkbox("Word boxes", value=True)
    layer_segments = st.checkbox("Line-segment boxes", value=True)
    layer_blocks   = st.checkbox("Block boxes", value=True)
    layer_columns  = st.checkbox("Columns", value=False)

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
page_index = st.number_input("Page (0-based)", min_value=0, value=0, step=1)

if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

# temp file
import tempfile
fd, path = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
with open(path, "wb") as f: f.write(uploaded.getbuffer())

try:
    words_all, page_size = extract_words(path)
    W, H = int(page_size[0]), int(page_size[1])
    words = [w for w in words_all if getattr(w.bbox, "page", 0) == page_index]

    # group once (shared by both tabs)
    columns = cluster_columns(words, gap_factor=col_gap)
    lines = []; [lines.extend(group_lines(col)) for col in columns]
    segments = []; [segments.extend(split_line_into_segments(ln, seg_gap_factor=seg_gap)) for ln in lines]
    blocks = group_blocks(segments, v_gap_factor=v_gap, x_iou_min=x_iou_min)

    tab_view, tab_annot = st.tabs(["📄 Viewer (read-only)", "✏️ Annotate (lasso)"])

    # -------------------- Viewer (read-only) --------------------
    with tab_view:
        st.caption("Read-only viewer with text overlay. Use Annotate tab to draw.")
        with st.sidebar:
            st.header("Viewer options")
            render_page = st.checkbox("Render PDF page image (needs poppler+pdf2image)", value=bool(PDF_TO_IMAGE))
            show_text   = st.checkbox("Overlay word text", value=True)
            label_size  = st.slider("Text label size", 8, 24, 12)

        # base image
        if render_page and PDF_TO_IMAGE:
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(path, first_page=page_index+1, last_page=page_index+1, dpi=144)
                base_img = pages[0].resize((W,H))
            except Exception:
                base_img = Image.new("RGB", (W,H), (255,255,255))
        else:
            base_img = Image.new("RGB", (W,H), (255,255,255))

        img = base_img.copy()
        dr = ImageDraw.Draw(img, "RGBA")

        if layer_words:
            for w in words:
                x0,y0,x1,y1 = word_bbox(w)
                dr.rectangle([x0,y0,x1,y1], outline=(210,210,210), width=1)

        if layer_segments:
            for seg in segments:
                bx0=min(w.bbox.x0 for w in seg); by0=min(w.bbox.y0 for w in seg)
                bx1=max(w.bbox.x1 for w in seg); by1=max(w.bbox.y1 for w in seg)
                dr.rectangle([bx0,by0,bx1,by1], outline=(0,0,0), width=2)

        if layer_blocks:
            for b in blocks:
                x0,y0,x1,y1 = b["bbox"]
                dr.rectangle([x0,y0,x1,y1], outline=(0,128,255), width=2)

        if layer_columns:
            pal = [(230,57,70,64),(29,53,87,64),(69,123,157,64),(168,218,220,64)]
            for i,col in enumerate(columns):
                xs0=[w.bbox.x0 for w in col]; ys0=[w.bbox.y0 for w in col]
                xs1=[w.bbox.x1 for w in col]; ys1=[w.bbox.y1 for w in col]
                cb=(min(xs0),min(ys0),max(xs1),max(ys1))
                dr.rectangle(cb, fill=pal[i%len(pal)])

        if show_text:
            img = draw_word_labels(img, words, font_size=label_size)

        st.image(img, use_column_width=True)

        # Page text table (segments)
        import pandas as pd
        seg_records = []
        for i, seg in enumerate(segments, 1):
            bx0=min(w.bbox.x0 for w in seg); by0=min(w.bbox.y0 for w in seg)
            bx1=max(w.bbox.x1 for w in seg); by1=max(w.bbox.y1 for w in seg)
            txt=" ".join(w.text for w in seg)
            seg_records.append({"idx": i, "x0": round(bx0,1), "y0": round(by0,1), "x1": round(bx1,1), "y1": round(by1,1), "text": txt})
        st.markdown("### Page text (segments)")
        if seg_records:
            st.dataframe(pd.DataFrame(seg_records), use_container_width=True, hide_index=True)
        else:
            st.info("No segments found on this page.")

    # -------------------- Annotate (lasso) --------------------
    with tab_annot:
        st.caption("Draw a polygon to select content. Use Undo/Clear as needed. Selection box + text shown on the right.")
        from streamlit_drawable_canvas import st_canvas

        # Underlay: always show page if we can (helps precise selection)
        if PDF_TO_IMAGE:
            try:
                pages = convert_from_path(path, first_page=page_index+1, last_page=page_index+1, dpi=144)
                base = pages[0].resize((W,H))
            except Exception:
                base = Image.new("RGB", (W,H), (255,255,255))
        else:
            base = Image.new("RGB", (W,H), (255,255,255))

        # light layers
        base_draw = base.copy()
        d = ImageDraw.Draw(base_draw)
        for w in words:
            x0,y0,x1,y1 = word_bbox(w)
            d.rectangle([x0,y0,x1,y1], outline=(210,210,210), width=1)
        for seg in segments:
            bx0=min(w.bbox.x0 for w in seg); by0=min(w.bbox.y0 for w in seg)
            bx1=max(w.bbox.x1 for w in seg); by1=max(w.bbox.y1 for w in seg)
            d.rectangle([bx0,by0,bx1,by1], outline=(0,0,0), width=1)

        st.markdown("#### Lasso selection (polygon)")
        canvas = st_canvas(
            background_image=base_draw,
            fill_color="rgba(255,0,0,0.15)",
            stroke_color="#FF0000",
            stroke_width=2,
            height=H,
            width=W,
            drawing_mode="polygon",
            key="lasso_canvas_editable"
        )

        # controls
        c1, c2, c3 = st.columns([1,1,3])
        with c1:
            if st.button("Undo last polygon"):
                if canvas and canvas.json_data and canvas.json_data.get("objects"):
                    objs = canvas.json_data["objects"]
                    if objs: objs.pop()
                    st.experimental_rerun()
        with c2:
            if st.button("Clear all"):
                st.session_state["lasso_canvas_editable.json_data"] = None
                st.experimental_rerun()

        # parse polygons robustly
        polys = []
        if canvas.json_data and canvas.json_data.get("objects"):
            for obj in canvas.json_data["objects"]:
                if obj.get("type") != "polygon": continue
                left = obj.get("left", 0); top = obj.get("top", 0)
                pts = []
                # fabric.js can emit "points" (preferred) or "path"
                if "points" in obj and isinstance(obj["points"], list):
                    for p in obj["points"]:
                        pts.append((float(p["x"])+left, float(p["y"])+top))
                elif "path" in obj and isinstance(obj["path"], list):
                    for seg in obj["path"]:
                        if len(seg) >= 3:
                            pts.append((float(seg[-2])+left, float(seg[-1])+top))
                if len(pts) >= 3:
                    polys.append(pts)

        # determine selection from the last polygon
        selected_boxes = []; selected_texts = []; selection_bbox = None
        if polys:
            poly = polys[-1]
            seg_boxes = []
            for seg in segments:
                bx0=min(w.bbox.x0 for w in seg); by0=min(w.bbox.y0 for w in seg)
                bx1=max(w.bbox.x1 for w in seg); by1=max(w.bbox.y1 for w in seg)
                bb = (bx0,by0,bx1,by1)
                if any_point_in_polygon(bb, poly):
                    seg_boxes.append(bb)
                    selected_texts.append(" ".join(w.text for w in seg))
            selection_bbox = bbox_union(seg_boxes)
            selected_boxes = seg_boxes

        # render selection overlay
        sel_img = base_draw
        if selected_boxes:
            sel_img = draw_boxes(sel_img, selected_boxes, outline=(255,0,0,255), width=3, fill=(255,0,0,40))

        colL, colR = st.columns([3,2], gap="large")
        with colL:
            st.image(sel_img, use_column_width=True)

        with colR:
            st.markdown("### Selection details")
            if selection_bbox:
                x0,y0,x1,y1 = selection_bbox
                st.write(f"**Selection bbox**: [x0={x0:.1f}, y0={y0:.1f}, x1={x1:.1f}, y1={y1:.1f}]  (w={(x1-x0):.1f}, h={(y1-y0):.1f})")
            else:
                st.write("_draw a polygon to select_")

            st.markdown("#### Segments in selection")
            if selected_texts:
                for i, txt in enumerate(selected_texts, 1):
                    sem = []
                    if is_money(txt): sem.append("price")
                    okd, _ = is_date(txt); 
                    if okd: sem.append("date")
                    ps = product_or_service(txt)
                    if ps: sem.append(ps)
                    sem_disp = f"  — _{', '.join(sem)}_" if sem else ""
                    st.write(f"{i}. {txt}{sem_disp}")
            else:
                st.write("_none yet_")

            st.markdown("#### Block inspector")
            block_opts = [f"block#{i+1}" for i in range(len(blocks))]
            pick = st.selectbox("Choose a block", options=["(none)"]+block_opts, index=0)
            if pick != "(none)":
                bi = int(pick.split("#")[1]) - 1
                b = blocks[bi]
                bx0,by0,bx1,by1 = b["bbox"]
                text_val = " ".join(w.text for w in sorted(b["words"], key=lambda ww: (ww.bbox.y0, ww.bbox.x0)))
                tagged = tag_field("block", text_val)
                st.write(f"**Block bbox**: [x0={bx0:.1f}, y0={by0:.1f}, x1={bx1:.1f}, y1={by1:.1f}]")
                st.write(f"**Text**: {text_val}")
                st.write(f"**Semantics**: {', '.join(tagged.get('semantics', [])) or '(none)'}")
                if tagged.get("normalized"): st.write(f"**Normalized**: {tagged['normalized']}")

finally:
    try: os.remove(path)
    except: pass
