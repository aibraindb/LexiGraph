
import streamlit as st, requests, io, json, pandas as pd
from pathlib import Path

st.set_page_config(page_title="LexiGraph Onboard v6", layout="wide")
st.title("LexiGraph — Scoped FIBO + Embeddings")

with st.sidebar:
    st.markdown("### 🔌 API")
    api_base = st.text_input("API base URL", value=st.session_state.get("api_base","http://127.0.0.1:8000"))
    st.session_state["api_base"] = api_base
    st.markdown("---")
    st.markdown("### 🧭 Context Association (CA)")
    case_id = st.text_input("case_id", value=st.session_state.get("case_id","CASE-DEMO-001"))
    product_id = st.text_input("product_id", value=st.session_state.get("product_id","LOAN-FIXED-30Y"))
    customer_id = st.text_input("customer_id", value=st.session_state.get("customer_id","CUST-DEMO-001"))
    if st.button("Associate context"):
        try:
            r = requests.post(f"{api_base}/ca/associate", json={"case_id":case_id,"product_id":product_id,"customer_id":customer_id}, timeout=15)
            r.raise_for_status(); st.session_state["ca"] = r.json(); st.success("CA linked")
        except Exception as e: st.error(f"CA error: {e}")

# Step 0: Namespace scope (editable multiselect)
st.header("0) Choose FIBO scope (namespaces)")

col0a, col0b = st.columns([3,1])
with col0b:
    if st.button("🔄 Reindex FIBO"):
        try:
            r = requests.post(f"{api_base}/fibo/reindex?force=true", timeout=30); r.raise_for_status()
            st.success(f"Reindexed: {r.json().get('num_classes')} classes")
        except Exception as e:
            st.error(e)

# fetch namespaces (after possible reindex)
try:
    r = requests.get(f"{api_base}/fibo/namespaces", timeout=15); r.raise_for_status()
    ns_list = r.json()
except Exception:
    ns_list = []

if ns_list:
    options = [f"{x['ns']} ({x['count']})" for x in ns_list]
    picked = st.multiselect("Namespaces", options=options, default=options[:4])
    active_ns = [p.split(" (",1)[0] for p in picked]
else:
    st.info("No namespaces discovered — using ALL (try Reindex FIBO above if this looks wrong).")
    active_ns = []

if st.button("Apply scope"):
    try:
        r = requests.post(f"{api_base}/fibo/scope", json={"namespaces": active_ns}, timeout=15)
        r.raise_for_status()
        st.success(f"Scope set ({len(active_ns) if active_ns else 'ALL'})")
    except Exception as e:
        st.error(e)

# Step 1: Upload & Embed
st.header("1) Upload & Embed")
uploaded = st.file_uploader("Upload PDF/TXT", type=["pdf","txt"])
if uploaded and st.button("Upload now"):
    try:
        r = requests.post(f"{api_base}/upload", files={"file":(uploaded.name, io.BytesIO(uploaded.read()), "application/octet-stream")}, timeout=120)
        r.raise_for_status(); res = r.json()
        st.session_state["last_doc_id"] = res["doc_id"]
        st.success(f"Uploaded. doc_id = {res['doc_id']}")
        st.subheader("Nearest neighbors"); st.json(res.get("neighbors",[]))
    except requests.HTTPError as e:
        try: st.error(e.response.json())
        except Exception: st.error(str(e))

# Step 2: Label via FIBO
st.header("2) Label via FIBO")
colA, colB = st.columns([3,2], gap="large")

with colA:
    st.subheader("Search FIBO, load subgraph")
    q = st.text_input("Search FIBO (e.g., 'Funding', 'FDS', 'Guarantee')", value="")
    hops = st.slider("Hops", 0, 4, 2, 1)
    if st.button("Search"):
        try:
            r = requests.get(f"{api_base}/fibo/search", params={"q": q, "limit": 25}, timeout=15); r.raise_for_status()
            st.session_state["fibo_hits"] = r.json()
        except Exception as e:
            st.error(e)
    for hit in st.session_state.get("fibo_hits", []):
        lbl = hit["label"] or hit["uri"].split("/")[-1]
        with st.expander(lbl):
            st.code(hit["uri"])
            if st.button(f"Load subgraph for {lbl}", key=f"load_{hit['uri']}"):
                try:
                    r = requests.get(f"{api_base}/fibo/subgraph", params={"focus": hit["uri"], "hops": hops}, timeout=15); r.raise_for_status()
                    subg = r.json()
                    html = (Path(__file__).resolve().parent.parent/"components"/"fibo_graph.html").read_text()
                    import json as _json
                    html = html.replace("window.graphData", "window.graphData = " + _json.dumps(subg, ensure_ascii=False))
                    html = html.replace("window.apiBase", "window.apiBase = '" + api_base + "'")
                    st.components.v1.html(html, height=620)
                    # set selection to focus so dropdown syncs
                    try: requests.post(f"{api_base}/ui/set_selection", params={"uri": hit["uri"]}, timeout=5)
                    except Exception: pass
                except Exception as e:
                    st.error(e)

with colB:
    st.subheader("Class picker & link")
    # dropdown fed by scoped classes
    try:
        r = requests.get(f"{api_base}/fibo/classes", timeout=15); r.raise_for_status()
        classes = r.json()
    except Exception:
        classes = []
    options = [f"{c['label']}|{c['uri']}" for c in classes]
    if "selected_class_uri" not in st.session_state: st.session_state["selected_class_uri"] = None

    # listen to graph clicks
    listen = st.checkbox("Listen to graph clicks", value=True)
    if listen:
        try:
            sel = requests.get(f"{api_base}/ui/get_selection", timeout=5).json().get("uri")
            if sel and sel != st.session_state.get("selected_class_uri"):
                st.session_state["selected_class_uri"] = sel
        except Exception: pass

    # compute index
    index = 0
    sel_uri = st.session_state.get("selected_class_uri")
    if sel_uri and options:
        for i,op in enumerate(options):
            if op.endswith("|"+sel_uri): index = i; break

    chosen = st.selectbox("FIBO class", options, index=index if options else 0)
    try: st.session_state["selected_class_uri"] = chosen.split("|",1)[1]
    except Exception: pass

    doc_id = st.text_input("doc_id", value=st.session_state.get("last_doc_id",""))
    if st.button("Link now", disabled=not (chosen and doc_id)):
        try:
            uri = chosen.split("|",1)[1]
            r = requests.post(f"{api_base}/label/link", json={"doc_id":doc_id, "fibo_class_uri": uri}, timeout=30)
            r.raise_for_status(); res = r.json()
            st.success("Linked ✅"); st.code(res.get("rdf_turtle",""), language="turtle")
            # update CA if token matches
            if "ca" in st.session_state and st.session_state["ca"].get("case_id"):
                label = chosen.split("|",1)[0].lower()
                for token in st.session_state["ca"]["expected"]:
                    if token.replace("_"," ") in label:
                        try:
                            upd = requests.post(f"{api_base}/ca/mark_present", params={"case_id":st.session_state['ca']['case_id'], "doc_type": token}, timeout=10).json()
                            st.session_state["ca"] = upd
                        except Exception: pass
        except Exception as e: st.error(e)

    st.subheader("Suggest class (neighbors)")
    if st.button("Suggest for last doc", disabled=not st.session_state.get("last_doc_id")):
        try:
            r = requests.post(f"{api_base}/suggest/class", params={"doc_id": st.session_state["last_doc_id"]}, timeout=30); r.raise_for_status()
            sug = r.json()
            if sug.get("suggestion"):
                st.info(f"Top suggestion: {sug['suggestion'].split('/')[-1]}")
                for i,op in enumerate(options):
                    if op.endswith("|"+sug["suggestion"]):
                        st.session_state["selected_class_uri"] = sug["suggestion"]; break
            else:
                st.warning("No labeled neighbors yet — add a labeled doc first.")
            st.json(sug)
        except Exception as e: st.error(e)

# Step 3: Documents table & drilldown
st.header("3) Documents")
try:
    r = requests.get(f"{api_base}/docs/list", timeout=10); r.raise_for_status()
    data = r.json()
    if data:
        df = pd.DataFrame(data); st.dataframe(df, use_container_width=True)
        st.markdown("#### Drill‑down")
        doc_pick = st.selectbox("Pick a doc_id to inspect", [d["doc_id"] for d in data])
        if doc_pick:
            try:
                n = requests.get(f"{api_base}/neighbors", params={"doc_id": doc_pick, "topk": 8}, timeout=10).json()
                st.subheader("Nearest neighbors"); st.json(n)
            except Exception: pass
            try:
                t = requests.get(f"{api_base}/rdf/{doc_pick}", timeout=10).json().get("turtle","")
                st.subheader("RDF"); st.code(t, language="turtle")
            except Exception: st.info("No RDF yet (not labeled).")
            st.subheader("FIBO class context vs Extracted fields")
            sel_doc = next((d for d in data if d["doc_id"]==doc_pick), {})
            fibo_uri = sel_doc.get("fibo_class_uri")
            left, right = st.columns(2)
            with left:
                st.markdown("**FIBO Class**")
                if fibo_uri: st.code(fibo_uri)
                else: st.warning("Unlabeled")
            with right:
                st.markdown("**Extracted fields**")
                st.info("No fields extracted in pure-embedding mode (roadmap item).")
    else:
        st.info("No documents yet.")
except Exception: st.info("Docs list unavailable.")
