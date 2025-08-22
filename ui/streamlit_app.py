from pathlib import Path
import sys
# add repo root to sys.path so "app.*" imports work no matter the CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import streamlit as st
from pathlib import Path
import json

from app.core.pipeline import process_document_dict
from app.core.fibo_index import search_scoped, subgraph_scoped
from app.core.fibo_attrs import attributes_for_class

st.set_page_config(page_title="LexiGraph Demo", layout="wide")
# --- Sidebar: FIBO Index controls ---
import traceback

with st.sidebar:
    st.subheader("FIBO Index")
    colA, colB = st.columns(2)
    if colA.button("Build Index"):
        try:
            from app.core.fibo_index import build_index
            idx = build_index(force=True)
            st.success(f"Structural index OK\n{idx.get('num_classes',0)} classes")
        except Exception as e:
            st.error("Structural index failed"); st.code(traceback.format_exc())

    if colB.button("Build Vectors"):
        try:
            # same as: python -m app.core.fibo_vec --rebuild
            from app.core.fibo_vec import build_fibo_vec
            info = build_fibo_vec(force=True)
            st.success(f"Vector index OK\n{info.get('n_docs',0)} classes embedded")
        except Exception:
            st.error("Vector index failed"); st.code(traceback.format_exc())

    # Health check
    try:
        from app.core.fibo_index import get_health
        h = get_health()
        st.caption(f"FIBO source: {h.get('source','?')}")
        st.caption(f"Classes: {h.get('num_classes',0)}  •  Edges: {h.get('num_edges',0)}")
    except Exception:
        pass

st.title("📄 LexiGraph – Document → FIBO Mapper")

# --- File Upload ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    pdf_bytes = uploaded.read()
    st.info(f"Processing: {uploaded.name} ...")
    res = process_document_dict(uploaded.name, pdf_bytes,
                                autolink_threshold=0.40,
                                use_ocr=False)

    # Quick JSON debug
    with st.expander("🔍 Raw Extraction JSON"):
        st.json(res)

    # --- Auto-linked FIBO ---
    st.subheader("Step 1. FIBO Auto-Link")
    if res.get("fibo_class_uri"):
        st.success(f"Auto-linked to **{res['fibo_class_uri']}** "
                   f"(score {res['fibo_candidates'][0]['score']:.2f})")
    else:
        st.warning("No confident auto-link. Try manual search below.")

    # Manual override
    query = st.text_input("Manual FIBO Search", value="")
    if st.button("Search"):
        hits = search_scoped(query, limit=15)
        if hits:
            st.write("Top hits:")
            for h in hits:
                st.write(f"- {h['label']} ({h['uri']})")

            # Load subgraph of first hit
            focus = hits[0]["uri"]
            graph = subgraph_scoped(focus, hops=2, include_properties=True)
            html = (Path(__file__).resolve().parents[1] /
                    "components" / "fibo_graph.html").read_text()
            html = html.replace("window.graphData", "window.graphData = " + json.dumps(graph))
            html = html.replace("window.apiBase", "window.apiBase = ''")
            st.components.v1.html(html, height=600)

            attrs = attributes_for_class(focus)
            st.write(f"Attributes for {hits[0]['label']}: {attrs['count']}")
            st.json(attrs["attributes"][:20])
        else:
            st.error("No hits found.")

    # --- Attribute coverage ---
    st.subheader("Step 2. Attribute Coverage")
    cov = res.get("coverage", {})
    st.metric("Coverage", f"{cov.get('have',0)}/{cov.get('want',0)} "
                          f"({100*cov.get('ratio',0):.1f}%)")

    # --- Mapped values ---
    st.subheader("Step 3. Extracted Values")
    if res.get("mapped"):
        st.table(res["mapped"])
    else:
        st.warning("No attribute values were mapped.")

    # --- Warnings ---
    if res.get("warnings"):
        st.error("Warnings:")
        st.write(res["warnings"])
else:
    st.info("Upload a PDF to begin…")
