import streamlit as st
from streamlit_drawable_canvas import st_canvas
from app.core.ocr_indexer import index_pdf_bytes, page_to_dict

st.set_page_config(layout="wide", page_title="OCR Tree Canvas")

st.title("📑 OCR Tree + Bounding Box Annotator")

# Session state
if "pages" not in st.session_state:
    st.session_state["pages"] = []
if "cur_page" not in st.session_state:
    st.session_state["cur_page"] = 0
if "selected_line" not in st.session_state:
    st.session_state["selected_line"] = None


# --- Upload ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    try:
        st.session_state["pages"] = index_pdf_bytes(pdf_bytes)
        st.success(f"Indexed {len(st.session_state['pages'])} page(s).")
    except Exception as e:
        st.error(f"OCR index failed: {e}")

if not st.session_state["pages"]:
    st.stop()

# --- Sidebar navigation ---
num_pages = len(st.session_state["pages"])
st.sidebar.subheader("Navigation")
st.session_state["cur_page"] = st.sidebar.slider("Page", 1, num_pages, st.session_state["cur_page"] + 1) - 1

pg = page_to_dict(st.session_state["pages"][st.session_state["cur_page"]])
lines = pg.get("lines", [])

# --- Layout: Tree + Canvas ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Document Tree")
    if not lines:
        st.info("No lines extracted on this page.")
    else:
        options = []
        for i, ln in enumerate(lines):
            txt = (ln.get("text", "") or "")[:60].replace("\n", " ")
            options.append(f"[{i+1}] {txt}")

        selected_opt = st.radio("Select line", options, index=0 if st.session_state["selected_line"] is None else st.session_state["selected_line"])
        st.session_state["selected_line"] = options.index(selected_opt)

with col2:
    st.subheader("Page Viewer / Annotator")

    # Build bounding boxes
    rects = []
    for i, ln in enumerate(lines):
        x0, y0, x1, y1 = ln["bbox"]
        rects.append({
            "type": "rect",
            "left": x0,
            "top": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "strokeColor": "red" if i != st.session_state["selected_line"] else "green",
            "fillColor": None,
        })

    canvas_res = st_canvas(
        background_image=None,  # You can feed page image here if available
        height=800,
        width=600,
        drawing_mode="transform",
        initial_drawing={"objects": rects},
        key=f"canvas_{st.session_state['cur_page']}"
    )

    if canvas_res.json_data and "objects" in canvas_res.json_data:
        # Detect clicks: pick the first modified object
        objs = canvas_res.json_data["objects"]
        for i, obj in enumerate(objs):
            if obj.get("strokeColor") == "green":
                st.session_state["selected_line"] = i
                break

# --- Show current selection ---
if st.session_state["selected_line"] is not None:
    sel = lines[st.session_state["selected_line"]]
    st.markdown(f"### Selected Line {st.session_state['selected_line']+1}")
    st.text_area("Text", sel.get("text", ""), height=80)
    st.json(sel, expanded=False)
