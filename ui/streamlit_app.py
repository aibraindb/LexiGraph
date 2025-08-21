import sys, os, json, requests
from pathlib import Path
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Must be first
st.set_page_config(page_title="LexiGraph 15.0", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.pdf_text import extract_text_blocks
from app.core.pipeline import propose_schema
from app.core.index_docs import add_document, list_documents, knn, sim_matrix
from app.core.fibo_attrs import attributes_for_class

API = st.sidebar.text_input("API base", "http://127.0.0.1:8000")

with st.sidebar.expander("FIBO Admin", expanded=False):
    if st.button("Rebuild FIBO index + vectors"):
        try:
            r = requests.post(f"{API}/fibo/rebuild", timeout=120)
            st.success(r.json())
        except Exception as e:
            st.error(e)

st.title("LexiGraph — FIBO‑Grounded Extraction + Topology View")

tab1, tab2, tab3 = st.tabs(["Upload & Extract", "FIBO Explorer (D3)", "Document Library"])

# --- Tab 1: Upload & Extract ---
with tab1:
    st.subheader("1) Upload PDF")
    doc = st.file_uploader("PDF", type=["pdf"])
    topk = st.slider("Candidate classes (top‑k)", 1, 12, 5)
    score_floor = st.slider("FIBO score floor", 0.0, 1.0, 0.25, 0.05)

    if doc and st.button("Propose FIBO‑aligned schema"):
        try:
            files = {"file": (doc.name, doc.getvalue(), "application/pdf")}
            data = {"topk": str(topk), "score_floor": str(score_floor)}
            r = requests.post(f"{API}/pipeline/propose", files=files, data=data, timeout=120)
            r.raise_for_status()
            prop = r.json()
            st.session_state["prop"] = prop
            st.session_state["schema"] = prop["schema"]

            # index document (value‑stripped) for topology view
            st.info("Indexing document into embedding space (value‑stripped)...")
            try:
                txt = prop.get("full_text") or ""
                add_document(doc.name, txt)
            except Exception as e:
                st.warning(f"Indexing warning: {e}")

            st.success("Schema proposed.")
        except Exception as e:
            st.error(e)

    prop = st.session_state.get("prop")
    if prop:
        st.subheader("2) Review candidates and attributes")
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("**FIBO Candidates (nearest by content)**")
            st.table([{"label": c.get("label"), "score": round(c.get("score",0.0),3), "uri": c["uri"]} for c in prop.get("candidates", [])])

            attrs = prop["schema"].get("attributes", [])
            editable = [{"include": True,
                        "label": (a.get("labels",[a['property'].split('/')[-1]])[0]),
                        "property": a["property"],
                        "source": a.get("source",{}).get("label")} for a in attrs]
            sel = st.data_editor(editable, hide_index=True,
                                column_config={"include": st.column_config.CheckboxColumn(default=True)},
                                key="attrs_editor")
            eff=[]
            for row in sel:
                if row.get("include"):
                    orig = next((a for a in attrs if a["property"] == row["property"]), None)
                    eff.append({"property": row["property"],
                                "labels": (orig.get("labels", []) if orig else [row["label"]]),
                                "source": orig.get("source", {}) if orig else {}})
            st.session_state["schema"] = {"documentName": prop["schema"]["documentName"],
                                          "fiboCandidates": prop["schema"]["fiboCandidates"],
                                          "attributes": eff}

        with c2:
            st.markdown("**Preview text (first 2000 chars)**")
            st.code((prop.get("full_text") or "")[:2000])

        # Apply schema
        st.subheader("3) Apply schema → extract")
        if st.button("Run extraction now"):
            try:
                files = {"file": (doc.name, doc.getvalue(), "application/pdf")}
                data = {"doc_id": prop["doc_id"], "schema": json.dumps(st.session_state["schema"])}
                r = requests.post(f"{API}/pipeline/apply", files=files, data=data, timeout=120)
                r.raise_for_status()
                res = r.json()
                st.session_state["result"] = res.get("result", {})
                st.session_state["pages"] = res.get("pages", [])
                st.success(f"Extracted {res['coverage']['found']} / {res['coverage']['total']} attributes")
            except Exception as e:
                st.error(e)

        # Evidence
        pages = st.session_state.get("pages") or []
        if pages:
            st.subheader("4) Evidence annotation (draw box and bind to attribute)")
            page_no = st.number_input("Page", 1, len(pages), 1)
            page = pages[page_no-1]
            attrs2 = st.session_state["schema"].get("attributes", [])
            label_map = {a["property"]: (a.get("labels",[a['property'].split('/')[-1]])[0]) for a in attrs2}

            canvas = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=2,
                background_image=page["image_b64"],
                height=680, width=980,
                drawing_mode="rect", key=f"canvas_{page_no}"
            )
            if "bbox_links" not in st.session_state: st.session_state["bbox_links"]=[]
            if canvas.json_data is not None and len(canvas.json_data.get("objects",[]))>0:
                obj = canvas.json_data["objects"][-1]
                x, y, w, h = obj.get("left",0), obj.get("top",0), obj.get("width",0), obj.get("height",0)
                options = [f"{label_map.get(a['property'],'?')} — {a['property']}" for a in attrs2]
                choice = st.selectbox("Bind drawn box to attribute", options) if options else None
                if choice and st.button("Bind box"):
                    prop_uri = choice.split(" — ")[-1]
                    st.session_state["bbox_links"].append({"page": page_no-1, "bbox_canvas":[x,y,x+w,y+h], "property": prop_uri})
                    st.success("Bound (demo persistence in memory).")

        if st.session_state.get("result"):
            st.subheader("Extracted values")
            rows=[]
            for prop, val in st.session_state["result"].items():
                rows.append({"attribute": prop, "value": val.get("value"), "confidence": val.get("confidence")})
            st.table(rows)

# --- Tab 2: FIBO Explorer ---
with tab2:
    st.subheader("Search FIBO and show subgraph")
    q = st.text_input("Search term (e.g., loan, lease, guarantee)", "")
    hops = st.slider("Neighborhood hops", 1, 4, 2)
    if st.button("Search"):
        try:
            r = requests.get(f"{API}/fibo/search", params={"q":q, "limit":25}, timeout=30)
            r.raise_for_status()
            hits = r.json()
            if not hits:
                st.info("No results.")
            for h in hits[:10]:
                with st.expander(f"{h.get('label') or h['uri'].split('/')[-1]}"):
                    st.code(h["uri"])
                    if st.button(f"Load subgraph", key=f"sg_{h['uri']}"):
                        try:
                            r2 = requests.get(f"{API}/fibo/subgraph", params={"focus":h["uri"], "hops": hops}, timeout=30)
                            r2.raise_for_status()
                            subg = r2.json()
                            html = (Path(__file__).resolve().parents[1]/"components"/"fibo_graph.html").read_text()
                            import json as _json
                            html = html.replace("window.graphData", "window.graphData = " + _json.dumps(subg, ensure_ascii=False))
                            html = html.replace("window.apiBase", "window.apiBase = '" + API + "'")
                            st.components.v1.html(html, height=640)
                        except Exception as e:
                            st.error(e)
        except Exception as e:
            st.error(e)

# --- Tab 3: Document Library / Topology ---
with tab3:
    st.subheader("Indexed documents (value‑stripped embedding space)")
    names = list_documents()
    if names:
        st.write({"count": len(names)})
        choice = st.selectbox("Select a document to view nearest neighbors", names)
        if st.button("Show k‑NN (k=8)"):
            try:
                kn = __import__("app.core.index_docs", fromlist=["knn"]).knn
                out = kn(choice, 8)
                st.table([{"neighbor": n, "similarity": round(s,3)} for n,s in out])
            except Exception as e:
                st.error(e)
        if st.button("Show pairwise similarity matrix"):
            import numpy as np
            M = sim_matrix()
            st.write(M)
    else:
        st.info("No documents indexed yet. Upload and click 'Propose schema' to index the doc content.")
