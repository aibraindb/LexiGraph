# ui/ocr_tree_canvas.py
# -----------------------------------------------------------------------------
# OCR tree + canvas viewer with two-way linking and editable boxes.
# - Upload a PDF -> we index with app.core.ocr_indexer (no PyMuPDF import here)
# - Left pane: hierarchical tree (Page -> Lines -> Words)
# - Right pane: image canvas with boxes; select in tree highlights on canvas
# - Click canvas to select nearest line -> updates tree selection
# - Edit coordinates for selected box; undo/redo stack per page
# -----------------------------------------------------------------------------

import io
import json
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st

# This is your existing indexer – returns an OCRIndex object
from app.core.ocr_indexer import index_pdf_bytes

# Optional interactive canvas (click support). We defensively handle absence.
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False


# ----------------------------- Helpers ----------------------------------------

def _to_dict(x):
    """Recursively convert dataclasses / objects to plain dicts."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, (list, tuple)):
        return [_to_dict(i) for i in x]
    if isinstance(x, dict):
        return {k: _to_dict(v) for k, v in x.items()}
    return x


def _page_to_dict(page_obj) -> Dict[str, Any]:
    """Your OCRIndex.Page -> dict with keys: image (bytes), lines[...], words[...]"""
    d = _to_dict(page_obj)
    # Ensure image is bytes
    if isinstance(d.get("image"), Image.Image):
        buf = io.BytesIO()
        d["image"].save(buf, format="PNG")
        d["image"] = buf.getvalue()
    return d


def _draw_boxes(img_bytes: bytes,
                lines: List[Dict[str, Any]],
                words: List[Dict[str, Any]],
                highlight_line_id: str | None,
                show_lines: bool,
                show_words: bool) -> bytes:
    """Render image with line/word boxes, highlighting selected line."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # Words (light blue)
    if show_words:
        for w in words:
            x, y, w_, h_ = w["bbox"]
            draw.rectangle([x, y, x + w_, y + h_], outline=(30, 144, 255, 220), width=1)

    # Lines (orange, selected = red)
    for ln in lines:
        x, y, w_, h_ = ln["bbox"]
        is_sel = (ln.get("id") == highlight_line_id)
        color = (255, 69, 0, 230) if is_sel else (255, 165, 0, 220)
        width = 3 if is_sel else 2
        draw.rectangle([x, y, x + w_, y + h_], outline=color, width=width)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _nearest_line(click_xy: Tuple[float, float], lines: List[Dict[str, Any]]) -> str | None:
    """Pick nearest line center to a click."""
    if not lines:
        return None
    cx, cy = click_xy
    best = None
    best_d2 = 1e18
    for ln in lines:
        x, y, w_, h_ = ln["bbox"]
        mx, my = x + w_ / 2, y + h_ / 2
        d2 = (mx - cx) ** 2 + (my - cy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best = ln.get("id")
    return best


def _push_undo(page_id: int, state: Dict[str, Any]):
    st.session_state.setdefault("undo_stacks", {})
    st.session_state["undo_stacks"].setdefault(page_id, [])
    # Shallow snapshot lines only
    snapshot = json.dumps(st.session_state["pages"][page_id].get("lines", []))
    st.session_state["undo_stacks"][page_id].append(snapshot)


def _undo(page_id: int):
    stk = st.session_state.get("undo_stacks", {}).get(page_id, [])
    if stk:
        snap = stk.pop()
        try:
            st.session_state["pages"][page_id]["lines"] = json.loads(snap)
        except Exception:
            pass


# ----------------------------- UI --------------------------------------------

st.set_page_config(page_title="LexiGraph • OCR Tree & Canvas", layout="wide")
st.title("📄 OCR Tree + Canvas (HITL)")

# Init session vars
for k, v in {
    "pages": None,             # List[dict] of pages
    "cur_page": 0,             # current page index
    "selected_line_id": None,  # which line is selected (ID string)
}.items():
    st.session_state.setdefault(k, v)

# Sidebar – Upload PDF
with st.sidebar:
    st.header("Upload")
    up = st.file_uploader("PDF document", type=["pdf"])
    if up is not None:
        try:
            ocr_index = index_pdf_bytes(up.read())  # returns OCRIndex
            # Convert to plain dicts
            pages = [_page_to_dict(p) for p in ocr_index.pages]
            st.session_state["pages"] = pages
            st.session_state["cur_page"] = 0
            st.session_state["selected_line_id"] = None
            st.success(f"Indexed: {len(pages)} page(s)")
        except Exception as e:
            st.error(f"OCR failed: {e}")

    if st.session_state["pages"]:
        st.markdown("---")
        st.subheader("Page")
        n_pages = len(st.session_state["pages"])
        st.session_state["cur_page"] = st.slider("Page", 1, n_pages, st.session_state["cur_page"] + 1) - 1

        st.markdown("---")
        st.subheader("Display")
        show_lines = st.checkbox("Show line boxes", value=True)
        show_words = st.checkbox("Show word boxes", value=False)

        st.markdown("---")
        st.subheader("Edit box")
        if st.session_state["selected_line_id"]:
            pid = st.session_state["cur_page"]
            pg = st.session_state["pages"][pid]
            lines = pg.get("lines", [])
            ln = next((l for l in lines if l.get("id") == st.session_state["selected_line_id"]), None)
            if ln:
                st.caption(f"Selected line ID: `{ln.get('id')}`")
                x, y, w_, h_ = ln["bbox"]
                # Simple numeric editors
                new_x = st.number_input("x", value=float(x), step=1.0)
                new_y = st.number_input("y", value=float(y), step=1.0)
                new_w = st.number_input("w", value=float(w_), step=1.0, min_value=1.0)
                new_h = st.number_input("h", value=float(h_), step=1.0, min_value=1.0)

                cols = st.columns(3)
                if cols[0].button("Apply"):
                    _push_undo(pid, st.session_state)
                    ln["bbox"] = [float(new_x), float(new_y), float(new_w), float(new_h)]
                    st.toast("Box updated", icon="✅")
                if cols[1].button("Undo"):
                    _undo(pid)
                if cols[2].button("Clear selection"):
                    st.session_state["selected_line_id"] = None
            else:
                st.info("Selection not found on this page.")
        else:
            st.caption("No line selected yet.")

        st.markdown("---")
        st.subheader("Save / Load")
        if st.button("💾 Save page annotations (JSON)"):
            pid = st.session_state["cur_page"]
            pg = st.session_state["pages"][pid]
            # Save only the structural bits you care about
            payload = {
                "page": pid,
                "lines": pg.get("lines", []),
                "words": pg.get("words", []),
                "width": pg.get("width"),
                "height": pg.get("height"),
            }
            st.download_button("Download JSON", data=json.dumps(payload, indent=2),
                               file_name=f"page_{pid+1}_ann.json", mime="application/json", use_container_width=True)

# Main layout
col_left, col_right = st.columns([0.38, 0.62], gap="large")

# Left: Document Tree
with col_left:
    st.subheader("Document Tree")

    if not st.session_state["pages"]:
        st.info("Upload a PDF to see OCR structure.")
    else:
        pid = st.session_state["cur_page"]
        pg = st.session_state["pages"][pid]
        pgd = pg  # already dict-like

        lines = pgd.get("lines", [])
        words = pgd.get("words", [])
        st.caption(f"Page {pid+1}: {len(lines)} line(s), {len(words)} word(s)")

        # Lines list (flat) with radio for selection
        # Keep it simple and fast (no nested expanders to avoid Streamlit nesting errors)
        options = [l.get("id") for l in lines]
        labels = [l.get("text", "")[:80] for l in lines]
        # Build display labels with small index
        display = [f"[{i+1}] {labels[i]}" for i in range(len(labels))]
        cur = st.session_state["selected_line_id"]
        default_idx = options.index(cur) if cur in options else 0 if options else None

        selected_idx = st.selectbox(
            "Select a line",
            index=default_idx if default_idx is not None else 0,
            options=list(range(len(options))) if options else [],
            format_func=lambda i: display[i],
            key=f"line_select_{pid}",
        )

        if options:
            st.session_state["selected_line_id"] = options[selected_idx]

        # Show a small preview of selected line
        if st.session_state["selected_line_id"]:
            ln = next((l for l in lines if l.get("id") == st.session_state["selected_line_id"]), None)
            if ln:
                st.text_area("Selected text", value=ln.get("text", ""), height=80)

# Right: Canvas
with col_right:
    st.subheader("Page Viewer / Annotator")

    if not st.session_state["pages"]:
        st.info("Upload a PDF to see the page image and boxes.")
    else:
        pid = st.session_state["cur_page"]
        pg = st.session_state["pages"][pid]
        img_bytes = pg.get("image")
        lines = pg.get("lines", [])
        words = pg.get("words", [])

        if img_bytes is None:
            st.error("This page has no image bytes. (Check indexer output.)")
        else:
            highlight_id = st.session_state["selected_line_id"]
            overlay = _draw_boxes(img_bytes, lines, words, highlight_id,
                                  show_lines=True, show_words=False)

            # If the canvas module is present, allow clicks; otherwise static image.
            if HAS_CANVAS:
                # Use the page image as background with our overlay-drawn image
                back_img = Image.open(io.BytesIO(overlay))
                cw, ch = back_img.width, back_img.height
                canvas_res = st_canvas(
                    fill_color="rgba(0, 0, 0, 0)",
                    stroke_width=1,
                    background_color="#FFFFFF",
                    background_image=back_img,
                    update_streamlit=True,
                    height=ch,
                    width=cw,
                    drawing_mode="point",   # click to add small point
                    point_display_radius=2,
                    key=f"canvas_{pid}",
                )
                # If user clicked -> select nearest line
                if canvas_res and canvas_res.json_data:
                    try:
                        js = canvas_res.json_data
                        objs = js.get("objects", [])
                        # find latest point
                        clicks = [o for o in objs if o.get("type") == "circle" and o.get("radius") == 2]
                        if clicks:
                            last = clicks[-1]
                            cx = float(last.get("left", 0)) + float(last.get("radius", 0))
                            cy = float(last.get("top", 0)) + float(last.get("radius", 0))
                            nid = _nearest_line((cx, cy), lines)
                            if nid:
                                st.session_state["selected_line_id"] = nid
                                st.toast("Selected nearest line from canvas click", icon="🖱️")
                    except Exception:
                        pass
            else:
                st.warning("Interactive canvas not available — showing static image (install streamlit-drawable-canvas).")
                st.image(overlay, use_column_width=True)
