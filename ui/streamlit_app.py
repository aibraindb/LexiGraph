import os, io, json, requests
import streamlit as st
import pandas as pd
import yaml
from rdflib import Graph, URIRef
from pathlib import Path

API_BASE = os.environ.get("LEXIGRAPH_API", "http://127.0.0.1:8000")
FIBO_TTL_DEFAULT = Path("data/fibo_trimmed.ttl")
D3_HTML_PATH = Path("ui/components/d3.html")

st.set_page_config(layout="wide", page_title="LexiGraph UI")
st.title("🧠 LexiGraph — Document Intelligence (v3)")

# Sidebar
with st.sidebar:
    st.caption("Classifier mode & learning")
    mode = st.radio("Classifier mode", ["hybrid (default)", "rule"], index=0)
    add_to_index = st.checkbox("Learn from this doc (add to vector index)", value=True)
    alpha = st.slider("α weight (rules)", 0.0, 1.0, 0.7, 0.05)
    beta  = st.slider("β weight (embeddings)", 0.0, 1.0, 0.3, 0.05)
    st.caption("Server")
    api_input = st.text_input("API base URL", value=API_BASE)
    st.markdown("---")
    st.caption("Train on upload")
    try:
        r = requests.get(f"{api_input}/variants/list", timeout=20)
        r.raise_for_status(); _variants = r.json().get("variants", [])
        variant_options = [f"{v['variant_id']} · {v['doc_type']}" for v in _variants]
    except Exception:
        _variants = []; variant_options = []
    sel_var = st.selectbox("Variant", options=variant_options or ["(none found)"])
    sel_variant_id = sel_var.split("·")[0].strip() if "·" in sel_var else None
    train_on_upload = st.checkbox("Train immediately on upload", value=False)
    st.markdown("---")
    with st.expander("🧹 Index admin", expanded=False):
        target_label = st.text_input("Filter by variant_id", value="")
        if st.button("List entries"):
            r = requests.get(f"{api_input}/index/list", params={"label": target_label or None})
            st.json(r.json())
        del_id = st.text_input("Delete entry id")
        if st.button("Delete entry"):
            if del_id:
                r = requests.delete(f"{api_input}/index/delete", params={"id": del_id}); st.json(r.json())
        if st.button("Undo last train()"):
            r = requests.post(f"{api_input}/index/undo_last"); st.json(r.json())
        if st.button("Rebuild index from dataset"):
            r = requests.post(f"{api_input}/index/rebuild"); st.json(r.json())

API_BASE = api_input.strip() or API_BASE

# Helpers
def call_api_classify_hybrid(file_bytes: bytes, filename: str, mode_key: str, add_flag: bool, alpha: float, beta: float):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    params = {"mode": "hybrid" if "hybrid" in mode_key else "rule", "add_to_index": str(add_flag).lower(), "alpha": alpha, "beta": beta}
    r = requests.post(f"{API_BASE}/classify/hybrid", params=params, files=files, timeout=180); r.raise_for_status(); return r.json()

def call_api_schema(variant_id: str):
    r = requests.get(f"{API_BASE}/schema", params={"variant_id": variant_id}, timeout=30); r.raise_for_status(); return r.json()

def call_api_extract(file_bytes: bytes, filename: str, include_rdf: bool, mode: str, alpha: float, beta: float, add_to_index: bool):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    params = {"include_rdf": str(include_rdf).lower(), "mode": "hybrid" if "hybrid" in mode else "rule", "alpha": alpha, "beta": beta, "add_to_index": str(add_to_index).lower()}
    r = requests.post(f"{API_BASE}/extract", params=params, files=files, timeout=180)
    if r.status_code >= 400:
        try: return {"__error__": True, "status": r.status_code, "json": r.json()}
        except Exception: return {"__error__": True, "status": r.status_code, "text": r.text}
    return r.json()

def call_api_list_variants():
    r = requests.get(f"{API_BASE}/variants/list", timeout=20); r.raise_for_status(); return r.json().get("variants", [])

def call_api_train_label(file_bytes: bytes, filename: str, variant_id: str):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    r = requests.post(f"{API_BASE}/train/label", params={"variant_id": variant_id, "persist_copy": True}, files=files, timeout=180)
    if r.status_code >= 400:
        try: return {"__error__": True, "status": r.status_code, "json": r.json()}
        except Exception: return {"__error__": True, "status": r.status_code, "text": r.text}
    return r.json()

def call_variants_suggest(file_bytes, filename):
    files={"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    r = requests.post(f"{API_BASE}/variants/suggest", files=files, timeout=120)
    r.raise_for_status(); return r.json()

def call_variants_create(variant_id, doc_type, identify, extract_fields):
    payload={"variant_id":variant_id,"doc_type":doc_type,"identify":identify,"extract":{"fields":extract_fields}}
    r = requests.post(f"{API_BASE}/variants/create", json=payload, timeout=30)
    if r.status_code>=400:
        try: return {"__error__":True,"status": r.status_code,"json": r.json()}
        except: return {"__error__":True,"status": r.status_code,"text": r.text}
    return r.json()

def call_api_upload(file_bytes: bytes, filename: str, variant_id: str | None, mode: str, alpha: float, beta: float, train_now: bool, persist_for_rules: bool):
    files = {"file": (filename, io.BytesIO(file_bytes), "application/pdf")}
    params = {"variant_id": variant_id, "mode": "hybrid" if "hybrid" in mode else "rule", "alpha": alpha, "beta": beta, "train_now": str(train_now).lower(), "persist_for_rules": str(persist_for_rules).lower()}
    r = requests.post(f"{API_BASE}/upload", params=params, files=files, timeout=180)
    if r.status_code >= 400:
        try: return {"__error__": True, "status": r.status_code, "json": r.json()}
        except Exception: return {"__error__": True, "status": r.status_code, "text": r.text}
    return r.json()

def call_api_dataset_stats():
    r = requests.get(f"{API_BASE}/dataset/stats", timeout=20); r.raise_for_status(); return r.json()

# ---- Flow ----
st.header("1) Upload File")
uploaded = st.file_uploader("Upload one PDF or TXT", type=["pdf","txt"])
include_rdf = st.checkbox("Include RDF (FIBO) in /extract response", value=True)

if not uploaded:
    st.info("Upload a file to begin."); st.stop()

file_bytes = uploaded.read()

# Train on upload
if train_on_upload and sel_variant_id:
    with st.spinner(f"Training with label: {sel_variant_id}…"):
        resp = call_api_train_label(file_bytes, uploaded.name, sel_variant_id)
    if isinstance(resp, dict) and resp.get("__error__"):
        st.error(f"/train/label → HTTP {resp['status']}")
        st.code(json.dumps(resp.get("json", {"text": resp.get("text","")}), indent=2))
        st.stop()
    else:
        st.success(f"Added to index under `{sel_variant_id}`. Embeddings persisted.")
        st.session_state['last_entry_ids'] = resp.get("entry_ids", [])

# New variant wizard
st.markdown("### ➕ Create a new variant from this document")
with st.expander("New variant wizard", expanded=False):
    col1,col2 = st.columns(2)
    with col1:
        new_vid  = st.text_input("variant_id", value="lease_vendorX_v1")
        new_dtype= st.selectbox("doc_type", ["lease_agreement","loan_agreement","invoice","bank_statement","tax_levy_notice","other"])
    if st.button("Suggest anchors & fields from this file"):
        sug = call_variants_suggest(file_bytes, uploaded.name)
        st.session_state["_sug"] = sug
        st.success("Suggested config generated below. Edit if needed and click Create.")
    if "_sug" in st.session_state:
        sug = st.session_state["_sug"]
        identify = st.text_area("identify (JSON)", json.dumps(sug["identify"], indent=2), height=220)
        fields   = st.text_area("extract.fields (JSON)", json.dumps(sug["extract"]["fields"], indent=2), height=280)
        rebuild = st.checkbox("Rebuild index after creating this variant", value=False, help="Recomputes embeddings from dataset/")
        if st.button("Create variant"):
            try:
                res = call_variants_create(new_vid, new_dtype, json.loads(identify), json.loads(fields))
                if isinstance(res, dict) and res.get("__error__"):
                    st.error(f"/variants/create → HTTP {res['status']}"); st.code(json.dumps(res.get("json",res), indent=2))
                else:
                    st.success(f"Variant `{new_vid}` created. Rules hot‑reloaded.")
                    if rebuild:
                        r = requests.post(f"{API_BASE}/index/rebuild"); st.json(r.json())
            except Exception as e:
                st.error(str(e))

# Upload & label one-shot (auto or manual)
st.markdown("### 📥 Upload & Label (one‑shot)")
with st.expander("Auto/Manual label and optional training on upload", expanded=False):
    vlist = call_api_list_variants()
    vopts = ["(auto)"] + [v["variant_id"] for v in vlist]
    chosen_vid = st.selectbox("Label as", options=vopts, index=0, help="Leave '(auto)' to use current classifier")
    train_now_ui = st.checkbox("Train embeddings now", value=True)
    keep_copy = st.checkbox("Keep copy for rule mining (data/labeled/…)", value=True)
    if st.button("Upload → Label → Save → (optional) Train"):
        res = call_api_upload(file_bytes, uploaded.name, None if chosen_vid == "(auto)" else chosen_vid, mode, alpha, beta, train_now_ui, keep_copy)
        if isinstance(res, dict) and res.get("__error__"):
            st.error(f"/upload → HTTP {res['status']}")
            st.code(json.dumps(res.get("json", {"text": res.get("text","")}), indent=2))
        else:
            st.success(f"Labeled as `{res['variant_id']}`. Trained now: {res.get('trained_now')}.")
            if res.get("classified_topk"):
                st.caption("Auto‑classification top‑k"); st.json(res["classified_topk"])
            st.caption("Dataset path"); st.code(res.get("dataset_path","(n/a)"))
            st.session_state['last_entry_ids'] = res.get("entry_ids", [])

# Classify
with st.spinner("Classifying…"):
    cls = call_api_classify_hybrid(file_bytes, uploaded.name, mode, add_to_index, alpha, beta)

ranked = cls.get("topk", [])
if not ranked:
    st.error("No classification candidates returned."); st.stop()

best = ranked[0]
st.success(f"[{cls.get('mode','?').upper()}] → **{best.get('type')}** (variant `{best.get('variant_id')}`)")
c1,c2 = st.columns([2,1])
with c1: st.caption("Top candidates (rule & hybrid scores)"); st.json(ranked)
with c2:
    st.caption("Preview"); st.code(cls.get("preview",""), language="text")
    if cls.get("added_ids"):
        st.caption("Last trained entry IDs")
        st.code(json.dumps(cls["added_ids"], indent=2), language="json")
        if st.button("Undo last train (server)"):
            r = requests.post(f"{API_BASE}/index/undo_last"); st.json(r.json())

# Schema
st.header("2) Effective Schema (FIBO aligned)")
variant_id = best.get("variant_id")
if not variant_id:
    st.error("No variant matched; cannot show schema."); st.stop()
r = requests.get(f"{API_BASE}/schema", params={"variant_id": variant_id}, timeout=30)
if r.status_code>=400:
    st.error("/schema failed"); st.stop()
eff = r.json()
schema_json = json.dumps(eff, indent=2)
schema_yaml = yaml.safe_dump(eff, sort_keys=False)

tabs = st.tabs(["Schema (view/download)", "Extraction", "FIBO Graph & Context"])

with tabs[0]:
    st.subheader("Schema JSON"); st.code(schema_json, language="json")
    st.download_button("⬇️ schema.json", schema_json, file_name=f"{variant_id}.schema.json", mime="application/json")
    st.subheader("Schema YAML"); st.code(schema_yaml, language="yaml")
    st.download_button("⬇️ schema.yaml", schema_yaml, file_name=f"{variant_id}.schema.yaml", mime="text/yaml")

with tabs[1]:
    st.subheader("Run extraction")
    with st.spinner("Extracting…"):
        ext = call_api_extract(file_bytes, uploaded.name, include_rdf, mode, alpha, beta, add_to_index)
    if isinstance(ext, dict) and ext.get("__error__"):
        st.error(f"/extract → HTTP {ext['status']}")
        if "json" in ext: st.code(json.dumps(ext["json"], indent=2), language="json")
        else: st.code(ext.get("text",""), language="text")
        st.stop()
    st.success("Extraction complete.")
    fields = ext.get("fields", {}); st.json(fields)
    if include_rdf and ext.get("rdf_turtle"):
        st.subheader("RDF Turtle"); st.code(ext["rdf_turtle"], language="turtle")
    st.subheader("Attributes: FIBO mapping vs Extracted values")
    rows = []
    for fname, meta in eff.get("fields", {}).items():
        fibo_prop = meta.get("fibo_property"); f = fields.get(fname, {})
        rows.append({"field": fname, "fibo_property": fibo_prop or "", "value": f.get("value"), "confidence": f.get("confidence"), "strategy": f.get("strategy")})
    df = pd.DataFrame(rows); st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.subheader("Ontology Context")
    ttl_up = st.file_uploader("Upload a larger FIBO TTL (optional)", type=["ttl"])
    if ttl_up:
        session_ttl = Path("data/_session_fibo.ttl")
        session_ttl.write_bytes(ttl_up.read())
        st.session_state["_fibo_path"] = str(session_ttl)
        st.success("Loaded session TTL. Graph below uses it.")

    ttl_path = Path(st.session_state.get("_fibo_path", str(FIBO_TTL_DEFAULT)))
    g = Graph()
    try:
        if ttl_path.exists() and ttl_path.stat().st_size > 0:
            g.parse(str(ttl_path), format="ttl")
    except Exception as e:
        st.warning(f"Failed to parse TTL: {e}")
        g = None

    def build_fibo_subgraph(g, root_uri, hops=2):
        if g is None:
            root = root_uri.split("/")[-1]
            return {"nodes":[{"id":root_uri,"label":root}],"links":[]}
        seen=set(); nodes=[]; links=[]
        def add_node(uri):
            if uri in seen: return
            seen.add(uri); nodes.append({"id":uri,"label":uri.split("/")[-1]})
        frontier = {root_uri}
        for _ in range(max(hops,1)):
            nxt=set()
            for u in list(frontier):
                add_node(u)
                for p,o in g.predicate_objects(subject=URIRef(u)):
                    if isinstance(o, URIRef):
                        add_node(str(o)); links.append({"source":u,"target":str(o),"label":str(p).split("/")[-1]})
                        nxt.add(str(o))
            frontier=nxt
        return {"nodes":nodes,"links":links}

    def render_d3(subgraph, height=520):
        html = Path("ui/components/d3.html").read_text(encoding="utf-8")
        payload = json.dumps(subgraph)
        html_injected = html.replace("/*__GRAPH_DATA__*/", f"window.GRAPH = {payload};")
        st.components.v1.html(html_injected, height=height, scrolling=False)

    fibo_class_suffix = (best.get("type") or "Document").replace("_"," ").title().replace(" ","")
    root_uri = f"https://spec.edmcouncil.org/fibo/ontology/{fibo_class_suffix}"
    sub = build_fibo_subgraph(g, root_uri, hops=2)
    a,b = st.columns([2,1])
    with a: render_d3(sub, height=520)
    with b:
        st.markdown("**Document class (FIBO):**"); st.code(root_uri, language="text")
        st.markdown("**Properties expected (from schema):**")
        want = [(k, eff["fields"][k].get("fibo_property","")) for k in eff["fields"]]
        st.table(pd.DataFrame(want, columns=["field","fibo_property"])) 
        extracted = {k for k,v in fields.items() if v.get("value")}
        missing = [k for k in eff["fields"] if k not in extracted]
        st.markdown(f"✅ Extracted: {', '.join(sorted(extracted)) or 'None'}")
        st.markdown(f"❌ Missing: {', '.join(sorted(missing)) or 'None'}")
