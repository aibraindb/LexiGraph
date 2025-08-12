# ui/streamlit_app.py
# Streamlit UI for DocAI + FIBO with Rule/Hybrid toggle, schema download, and D3 graph context

import os, io, json, requests
import streamlit as st
import pandas as pd
import yaml
from rdflib import Graph, URIRef
from pathlib import Path

# --- Config (no secrets.toml required) ---
API_BASE = os.environ.get("DOCAPI_BASE", "http://127.0.0.1:8000")
FIBO_TTL_PATH = Path("data/fibo_trimmed.ttl")
D3_HTML_PATH = Path("ui/components/d3.html")

st.set_page_config(layout="wide", page_title="DocAI + FIBO")
st.title("📄 Document Intelligence + FIBO")

# Sidebar controls
with st.sidebar:
    st.caption("Classifier mode & learning")
    mode = st.radio("Classifier mode", ["hybrid (default)", "rule"], index=0, help="Hybrid = rules + embeddings")
    add_to_index = st.checkbox("Learn from this doc (add to vector index)", value=True)
    alpha = st.slider("α weight (rules)", 0.0, 1.0, 0.7, 0.05)
    beta  = st.slider("β weight (embeddings)", 0.0, 1.0, 0.3, 0.05)
    st.caption("Server")
    api_input = st.text_input("API base URL", value=API_BASE, help="Your FastAPI URL, e.g., http://127.0.0.1:8000")

API_BASE = api_input.strip() or API_BASE

# ---- Helpers ----------------------------------------------------------------

def call_api_classify_hybrid(file_bytes: bytes, filename: str, mode_key: str, add_flag: bool, alpha: float, beta: float):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    params = {
        "mode": "hybrid" if "hybrid" in mode_key else "rule",
        "add_to_index": str(add_flag).lower(),
        "alpha": alpha,
        "beta": beta,
    }
    r = requests.post(f"{API_BASE}/classify/hybrid", params=params, files=files, timeout=180)
    r.raise_for_status()
    return r.json()

def call_api_schema(variant_id: str):
    r = requests.get(f"{API_BASE}/schema", params={"variant_id": variant_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def call_api_extract(file_bytes: bytes, filename: str, include_rdf: bool,
                     mode: str, alpha: float, beta: float, add_to_index: bool):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    params = {
        "include_rdf": str(include_rdf).lower(),
        "mode": "hybrid" if "hybrid" in mode else "rule",
        "alpha": alpha,
        "beta": beta,
        "add_to_index": str(add_to_index).lower(),
    }
    r = requests.post(f"{API_BASE}/extract", params=params, files=files, timeout=180)
    if r.status_code >= 400:
        try:
            return {"__error__": True, "status": r.status_code, "json": r.json()}
        except Exception:
            return {"__error__": True, "status": r.status_code, "text": r.text}
    return r.json()

def load_fibo_graph():
    g = Graph()
    if FIBO_TTL_PATH.exists() and FIBO_TTL_PATH.stat().st_size > 0:
        g.parse(str(FIBO_TTL_PATH), format="ttl")
        return g
    return None

def build_fibo_subgraph(g: Graph, root_uri: str, hops: int = 1):
    """Return {nodes:[{id,label}], links:[{source,target,label}]} for D3."""
    if g is None:
        root = root_uri.split("/")[-1]
        return {"nodes": [{"id": root_uri, "label": root}], "links": []}

    seen = set(); nodes = []; links = []
    def add_node(uri: str):
        if uri in seen: return
        seen.add(uri)
        nodes.append({"id": uri, "label": uri.split("/")[-1]})

    frontier = {root_uri}
    for _ in range(max(hops, 1)):
        nxt = set()
        for u in list(frontier):
            add_node(u)
            for p, o in g.predicate_objects(subject=URIRef(u)):
                if isinstance(o, URIRef):
                    add_node(str(o))
                    links.append({"source": u, "target": str(o), "label": str(p).split("/")[-1]})
                    nxt.add(str(o))
        frontier = nxt
    return {"nodes": nodes, "links": links}

def render_d3(subgraph: dict, height: int = 600):
    html = D3_HTML_PATH.read_text(encoding="utf-8")
    payload = json.dumps(subgraph)
    # Inject graph JSON into window.GRAPH before D3 code runs
    html_injected = html.replace("/*__GRAPH_DATA__*/", f"window.GRAPH = {payload};")
    st.components.v1.html(html_injected, height=height, scrolling=False)

def df_attr_compare(eff_schema: dict, fields: dict) -> pd.DataFrame:
    # Build rows: [Field, FIBO property, Extracted value, Confidence, Strategy]
    rows = []
    for fname, meta in eff_schema.get("fields", {}).items():
        fibo_prop = meta.get("fibo_property")
        f = fields.get(fname, {})
        rows.append({
            "field": fname,
            "fibo_property": fibo_prop or "",
            "value": f.get("value"),
            "confidence": f.get("confidence"),
            "strategy": f.get("strategy"),
        })
    return pd.DataFrame(rows)

# ---- Flow -------------------------------------------------------------------

st.header("1) Upload PDF")
uploaded = st.file_uploader("Upload one PDF", type=["pdf"])
include_rdf = st.checkbox("Include RDF (FIBO) in /extract response", value=True)

if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

file_bytes = uploaded.read()

with st.spinner("Classifying…"):
    cls = call_api_classify_hybrid(file_bytes, uploaded.name, mode, add_to_index, alpha, beta)

ranked = cls.get("topk", [])
if not ranked:
    st.error("No classification candidates returned.")
    st.stop()

best = ranked[0]
st.success(f"[{cls.get('mode','?').upper()}] → **{best.get('type')}** (variant `{best.get('variant_id')}`)")

colA, colB = st.columns([2,1])
with colA:
    st.caption("Top candidates (rule & hybrid scores)")
    st.json(ranked)
with colB:
    st.caption("Preview")
    st.code(cls.get("preview", ""), language="text")

# Schema
st.header("2) Effective Schema (FIBO aligned)")
variant_id = best.get("variant_id")
if not variant_id:
    st.error("No variant matched; cannot show schema.")
    st.stop()

eff = call_api_schema(variant_id)
schema_json = json.dumps(eff, indent=2)
schema_yaml = yaml.safe_dump(eff, sort_keys=False)

tabs = st.tabs(["Schema (view/download)", "Extraction", "FIBO Graph & Context"])

with tabs[0]:
    st.subheader("Schema JSON")
    st.code(schema_json, language="json")
    st.download_button("⬇️ Download schema.json", schema_json, file_name=f"{variant_id}.schema.json", mime="application/json")
    st.subheader("Schema YAML")
    st.code(schema_yaml, language="yaml")
    st.download_button("⬇️ Download schema.yaml", schema_yaml, file_name=f"{variant_id}.schema.yaml", mime="text/yaml")

# Extract
with tabs[1]:
    st.subheader("Run extraction")
    with st.spinner("Extracting…"):
        ext = call_api_extract(file_bytes, uploaded.name, include_rdf, mode, alpha, beta, add_to_index)
    st.success("Extraction complete.")
    fields = ext.get("fields", {})
    st.json(fields)

    if include_rdf and ext.get("rdf_turtle"):
        st.subheader("RDF Turtle")
        st.code(ext["rdf_turtle"], language="turtle")

    st.subheader("Attributes: FIBO mapping vs Extracted values")
    df = df_attr_compare(eff, fields)
    st.dataframe(df, use_container_width=True)

# FIBO Graph + context
with tabs[2]:
    st.subheader("Ontology Context")
    g = load_fibo_graph()
    fibo_class_suffix = (best.get("type") or "Document").replace("_", " ").title().replace(" ", "")
    root_uri = f"https://spec.edmcouncil.org/fibo/ontology/{fibo_class_suffix}"
    sub = build_fibo_subgraph(g, root_uri, hops=2)

    ctx_col1, ctx_col2 = st.columns([2,1])
    with ctx_col1:
        render_d3(sub, height=520)
    with ctx_col2:
        st.markdown("**Document class (FIBO):**")
        st.code(root_uri, language="text")
        st.markdown("**Properties expected (from schema):**")
        want = [(k, eff["fields"][k].get("fibo_property", "")) for k in eff["fields"]]
        st.table(pd.DataFrame(want, columns=["field", "fibo_property"]))
        extracted = {k for k, v in fields.items() if v.get("value")}
        missing = [k for k in eff["fields"] if k not in extracted]
        st.markdown(f"✅ Extracted: {', '.join(sorted(extracted)) or 'None'}")
        st.markdown(f"❌ Missing: {', '.join(sorted(missing)) or 'None'}")
