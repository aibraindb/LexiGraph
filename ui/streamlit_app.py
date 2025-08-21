# ui/streamlit_app.py
# LexiGraph – single-page Streamlit UI
# - Robust page-image handling for canvas (base64/bytes/path/PIL/np -> PIL)
# - Upload -> Propose FIBO-aligned schema -> Apply extraction -> Evidence (canvas)
# - FIBO Explorer (D3 subgraph) with search + maximize
# - Document Library (light list) – persists to data/runs/
# ------------------------------------------------------------------------------

# --- must be first Streamlit call ---
import streamlit as st
st.set_page_config(page_title="LexiGraph", layout="wide")

# stdlib / 3rd-party
import io, os, json, base64
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image

# optional canvas (nice to have)
try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    HAS_CANVAS = False

# local modules (degrade gracefully if absent)
_missing: List[str] = []
def _safe_import(modpath, name):
    try:
        module = __import__(modpath, fromlist=[name])
        return getattr(module, name)
    except Exception:
        _missing.append(modpath)
        return None

extract_text_blocks = _safe_import("app.core.pdf_text", "extract_text_blocks")
get_page_images     = _safe_import("app.core.pdf_text", "get_page_images")
focused_summary     = _safe_import("app.core.pdf_text", "focused_summary")

build_fibo_index    = _safe_import("app.core.fibo_index", "build_index")
search_scoped       = _safe_import("app.core.fibo_index", "search_scoped")
subgraph_scoped     = _safe_import("app.core.fibo_index", "subgraph_scoped")

build_fibo_vec      = _safe_import("app.core.fibo_vec", "build_fibo_vec")
fibo_search         = _safe_import("app.core.fibo_vec", "search_fibo")

attributes_for_class= _safe_import("app.core.fibo_attrs", "attributes_for_class")

apply_pipeline      = _safe_import("app.core.pipeline", "apply_pipeline")  # optional
value_map_doc       = _safe_import("app.core.value_mapper", "value_map_doc")

# ------------------------------------------------------------------------------
# Helpers: image normalization for canvas (fixes: str has no attribute height)
# ------------------------------------------------------------------------------

def _to_pil(img_like: Any) -> Image.Image:
    """
    Accepts:
      - PIL.Image.Image
      - numpy array (H,W[,C])
      - bytes/bytearray (encoded image)
      - str: base64 (optionally data: URI) or filepath
    Returns PIL.Image in RGBA.
    """
    if isinstance(img_like, Image.Image):
        return img_like.convert("RGBA")

    if isinstance(img_like, np.ndarray):
        arr = img_like
        if arr.ndim == 2:
            return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGBA")
        if arr.ndim == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGBA")

    if isinstance(img_like, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(img_like)).convert("RGBA")
        except Exception:
            pass

    if isinstance(img_like, str):
        s = img_like.strip()
        # data URL?
        if s.startswith("data:image"):
            try:
                b64 = s.split(",", 1)[1]
                raw = base64.b64decode(b64)
                return Image.open(io.BytesIO(raw)).convert("RGBA")
            except Exception:
                pass
        # bare base64?
        try:
            raw = base64.b64decode(s, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGBA")
        except Exception:
            pass
        # path?
        p = Path(s)
        if p.exists():
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                pass

    raise ValueError("Unsupported image input for canvas; cannot convert to PIL.Image")

def _resize_img(pil_img: Image.Image, max_w: int = 1000) -> Image.Image:
    """Keep aspect ratio, cap width at max_w."""
    if not isinstance(pil_img, Image.Image):
        pil_img = _to_pil(pil_img)
    w, h = pil_img.size
    if w <= 0 or h <= 0:
        raise ValueError("Empty image")
    if w <= max_w:
        return pil_img
    scale = max_w / float(w)
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS)

def _pil_to_canvas_rgba(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    return np.array(pil_img)

# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------

RUNS_DIR = Path("data/runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def _save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def _load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _doc_id_from_name(name: str) -> str:
    stem = Path(name).stem
    return stem.replace(" ", "_")[:64]

def _warn_missing_modules():
    if not _missing:
        return
    st.warning(
        "Some modules were not found and related features are disabled:\n\n"
        + "\n".join(f"- `{m}`" for m in sorted(set(_missing)))
    )

# ------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------

with st.sidebar:
    st.header("LexiGraph Controls")
    if build_fibo_index and build_fibo_vec:
        if st.button("Rebuild FIBO index + vectors", use_container_width=True):
            try:
                idx = build_fibo_index(force=True)
                info = build_fibo_vec(force=True)
                st.success(f"FIBO rebuilt: {len(idx.get('classes',[]))} classes, {info.get('n_terms',0)} terms.")
            except Exception as e:
                st.error(f"FIBO rebuild failed: {e}")
    else:
        st.info("FIBO index/vector builders not available in this environment.")

    _warn_missing_modules()

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["1) Upload & Extract", "2) FIBO Explorer (D3)", "3) Document Library"])

# ------------------------------------------------------------------------------
# 1) Upload & Extract
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Upload a PDF, propose a FIBO-aligned schema, and extract values")

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is not None:
        doc_id = _doc_id_from_name(uploaded.name)
        run_dir = RUNS_DIR / doc_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract text + spans + page images
        try:
            bs = uploaded.read()
            if extract_text_blocks is None or get_page_images is None:
                raise RuntimeError("Missing app.core.pdf_text utilities")

            blocks = extract_text_blocks(bs)       # [{page, text, spans:[{text,bbox}]}...]
            page_imgs = get_page_images(bs)        # list of base64/png bytes per page

            # Persist raw artifacts
            full_text = "\n".join([b.get("text","") for b in blocks])
            _save_json(run_dir / "spans.json", blocks)
            (run_dir / "text.txt").write_text(full_text)

            st.success(f"Parsed {len(blocks)} pages for {uploaded.name}")

        except Exception as e:
            st.error(f"PDF parse failed: {e}")
            blocks, page_imgs, full_text = [], [], ""

        # UI: Page preview + evidence canvas
        if page_imgs:
            page_idx = st.number_input("Page", min_value=0, max_value=len(page_imgs)-1, value=0, step=1)
            raw_bg = page_imgs[page_idx]

            try:
                pil_bg = _to_pil(raw_bg)
                pil_bg = _resize_img(pil_bg, 950)
                bg_np  = _pil_to_canvas_rgba(pil_bg)
            except Exception as e:
                bg_np, pil_bg = None, None
                st.error(f"Failed to prepare page image for canvas: {e}")

            st.caption("Draw a rectangle to bind it to a FIBO attribute (after proposing a schema).")

            if HAS_CANVAS and bg_np is not None:
                canvas_res = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_color="#FFFFFF",
                    height=bg_np.shape[0],
                    width=bg_np.shape[1],
                    background_image=pil_bg,  # PIL works well with most versions
                    drawing_mode="rect",
                    key=f"canvas_page_{doc_id}_{page_idx}",
                )
                drawn = canvas_res.json_data if canvas_res is not None else None
            else:
                st.info("Install `streamlit-drawable-canvas` for annotation, showing static page instead.")
                if pil_bg:
                    st.image(pil_bg, use_column_width=False)

        # Step 2: Propose FIBO aligned schema (classes + attributes)
        st.markdown("---")
        st.markdown("### Propose FIBO-aligned schema")
        score_floor = st.slider("Minimum class score (cosine TF‑IDF)", 0.0, 1.0, 0.25, 0.01)
        top_k       = st.slider("Top‑K FIBO classes to consider", 1, 10, 5, 1)

        if st.button("Propose", use_container_width=True, disabled=(fibo_search is None or attributes_for_class is None)):
            try:
                # Basic doc summary to bias search (use first N lines)
                summary = focused_summary(full_text) if focused_summary else full_text[:2000]
                st.session_state["last_summary"] = summary

                # FIBO search
                hits = fibo_search(summary, top_k=top_k)
                hits = [h for h in hits if h.get("score",0.0) >= score_floor]
                st.session_state["fibo_hits"] = hits

                # Merge attributes across selected classes
                attr_map = {}
                for h in hits:
                    cls_uri = h["uri"]
                    attrs = attributes_for_class(cls_uri).get("attributes", [])
                    for row in attrs:
                        prop = row["property"]
                        labels = row.get("labels", []) or [prop.split("/")[-1]]
                        # keep first occurrence
                        if prop not in attr_map:
                            attr_map[prop] = {"property": prop, "labels": labels, "selected": True}

                proposed = list(attr_map.values())
                _save_json(run_dir / "schema.json", {"classes": hits, "attributes": proposed})
                st.session_state["schema_doc"] = str(run_dir / "schema.json")
                st.success(f"Proposed schema with {len(proposed)} attributes from {len(hits)} FIBO classes")

            except Exception as e:
                st.error(f"Proposal failed: {e}")

        # Show/edit proposed schema if exists
        schema_path = st.session_state.get("schema_doc")
        if schema_path and Path(schema_path).exists():
            schema = _load_json(Path(schema_path), {})
            st.markdown("#### Proposed attributes")
            attrs = schema.get("attributes", [])

            if not attrs:
                st.info("No attributes proposed yet. Click **Propose** above.")
            else:
                # Editable table
                df_rows = []
                for r in attrs:
                    df_rows.append({
                        "selected": bool(r.get("selected", True)),
                        "label": (r.get("labels") or [r["property"].split("/")[-1]])[0],
                        "property": r["property"]
                    })
                edited = st.data_editor(df_rows, num_rows="dynamic", use_container_width=True)
                # persist selection back
                by_prop = {r["property"]: r for r in attrs}
                for row in edited:
                    pr = row["property"]
                    if pr in by_prop:
                        by_prop[pr]["selected"] = bool(row["selected"])
                        # keep label in labels[0] for extraction hints
                        lab = row.get("label") or pr.split("/")[-1]
                        labs = by_prop[pr].get("labels") or []
                        if labs:
                            labs[0] = lab
                            by_prop[pr]["labels"] = labs
                        else:
                            by_prop[pr]["labels"] = [lab]
                schema["attributes"] = list(by_prop.values())
                _save_json(Path(schema_path), schema)

                # Apply extraction
                st.markdown("#### Apply extraction")
                if st.button("Run extraction now", use_container_width=True,
                             disabled=(value_map_doc is None and apply_pipeline is None)):
                    try:
                        if apply_pipeline:
                            result = apply_pipeline(full_text, schema)  # your pipeline does everything
                        else:
                            # Fallback: regex/anchors only
                            result = value_map_doc(full_text, schema)
                        _save_json(run_dir / "result.json", result)
                        st.success(f"Extraction complete. Coverage: {result.get('coverage','?')}")
                        # Show a compact view
                        kv = result.get("values", {})
                        st.json(kv)
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")

        # Evidence binding (demo; persists per doc in session)
        st.markdown("---")
        st.markdown("### Evidence (boxes → attributes)")
        if HAS_CANVAS and page_imgs and schema_path and Path(schema_path).exists():
            schema = _load_json(Path(schema_path), {})
            attr_opts = [ ( (r.get("labels") or [r["property"].split("/")[-1]])[0] + "  ·  " + r["property"], r["property"])
                          for r in schema.get("attributes", []) if r.get("selected", True) ]
            if not attr_opts:
                st.info("No selected attributes. Toggle attributes in the table above.")
            else:
                default_attr = attr_opts[0][1]
                bind_prop = st.selectbox("Bind last rectangle to attribute", options=[v for _,v in attr_opts], index=0)
                if st.button("Bind last rectangle", use_container_width=True):
                    doc_key = f"boxes::{doc_id}"
                    store = st.session_state.get(doc_key, [])
                    # append last rect if exists
                    try:
                        if canvas_res and canvas_res.json_data and canvas_res.json_data.get("objects"):
                            rects = [o for o in canvas_res.json_data["objects"] if o.get("type")=="rect"]
                            if rects:
                                last = rects[-1]
                                store.append({
                                    "page": int(page_idx),
                                    "x": float(last.get("left",0)),
                                    "y": float(last.get("top",0)),
                                    "w": float(last.get("width",0)),
                                    "h": float(last.get("height",0)),
                                    "property": bind_prop
                                })
                                st.session_state[doc_key] = store
                                _save_json(run_dir / "evidence.json", store)
                                st.success("Bound last rectangle.")
                        else:
                            st.warning("No rectangle drawn yet.")
                    except Exception as e:
                        st.error(f"Binding failed: {e}")

                # Show existing evidence rows
                evidence = _load_json(run_dir / "evidence.json", [])
                if evidence:
                    st.write(f"Evidence items: {len(evidence)}")
                    st.dataframe(evidence, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# 2) FIBO Explorer (D3)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("FIBO Explorer")
    colA, colB = st.columns([2,1])
    with colA:
        q = st.text_input("Search term (label / altLabel / URI tail)", value="loan")
        hops = st.slider("Neighborhood hops", 1, 4, 2, 1)
        if st.button("Search & Load Subgraph"):
            try:
                hits = search_scoped(q, limit=25) if search_scoped else []
                if not hits:
                    st.warning("No hits in current scope or index not built.")
                else:
                    # pick first hit
                    focus = hits[0]["uri"]
                    subg = subgraph_scoped(focus, hops=hops) if subgraph_scoped else {"nodes": [], "links": []}
                    # load D3 template
                    html_path = Path(__file__).resolve().parent.parent / "components" / "fibo_graph.html"
                    html = html_path.read_text()
                    html = html.replace("window.graphData", "window.graphData = " + json.dumps(subg))
                    html = html.replace("window.apiBase", "window.apiBase = 'http://127.0.0.1:8000'")
                    st.components.v1.html(html, height=650)
            except Exception as e:
                st.error(f"D3 load failed: {e}")
    with colB:
        st.info(
            "Tips:\n"
            "- Enter a broad term like *loan*, *lease*, *guarantee*\n"
            "- Use the **Maximize** button in the toolbar inside the graph\n"
            "- Rebuild the FIBO index/vectors from the sidebar after dropping a new TTL"
        )

# ------------------------------------------------------------------------------
# 3) Document Library (very light)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Indexed Documents")
    rows = []
    for p in RUNS_DIR.glob("*/text.txt"):
        did = p.parent.name
        n_pages = len(_load_json(p.parent / "spans.json", []))
        has_schema = (p.parent / "schema.json").exists()
        has_result = (p.parent / "result.json").exists()
        rows.append({
            "doc_id": did,
            "pages": n_pages,
            "schema": "yes" if has_schema else "no",
            "result": "yes" if has_result else "no"
        })
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No documents yet. Upload a PDF in tab 1.")

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.caption("LexiGraph — FIBO-aligned schema proposal, extraction & evidence. This build fixes canvas image handling.")
