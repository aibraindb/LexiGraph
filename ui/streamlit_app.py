# ui/streamlit_app.py
import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="LexiGraph", layout="wide")
st.title("LexiGraph — Context + FIBO + Embeddings (Demo)")

# =========================
# Sidebar: API + CA + Email
# =========================
with st.sidebar:
    st.markdown("### 🔌 API")
    api_base = st.text_input(
        "API base URL",
        value=st.session_state.get("api_base", "http://127.0.0.1:8000"),
    )
    st.session_state["api_base"] = api_base

    st.markdown("---")
    st.markdown("### 🧭 Context Association (CA)")
    case_id = st.text_input(
        "case_id", value=st.session_state.get("case_id", "CASE-DEMO-001")
    )
    product_id = st.text_input(
        "product_id", value=st.session_state.get("product_id", "LOAN-FIXED-30Y")
    )
    customer_id = st.text_input(
        "customer_id", value=st.session_state.get("customer_id", "CUST-DEMO-001")
    )
    if st.button("Associate context"):
        try:
            r = requests.post(
                f"{api_base}/ca/associate",
                json={
                    "case_id": case_id,
                    "product_id": product_id,
                    "customer_id": customer_id,
                },
                timeout=15,
            )
            r.raise_for_status()
            st.session_state["ca"] = r.json()
            st.success("CA linked")
        except Exception as e:
            st.error(f"CA error: {e}")

    st.markdown("---")
    st.markdown("### 📬 Email Inbox")
    # Always fetch threads to keep it simple
    try:
        threads = requests.get(f"{api_base}/demo/threads", timeout=10).json()
    except Exception:
        threads = []

    if threads:
        picked = st.selectbox(
            "Select email", [f"{t['id']} — {t['subject']}" for t in threads], key="email_pick"
        )
        th = next((t for t in threads if picked and picked.startswith(t["id"])), None)
        if th:
            # Silent CA bind (no buttons)
            try:
                _ = requests.post(
                    f"{api_base}/ca/associate",
                    json={
                        "case_id": f"CASE-{th['id']}",
                        "product_id": ("LOAN" if "Loan" in th["subject"] else "LEASE"),
                        "customer_id": "CUST-ALPHA",
                    },
                    timeout=10,
                )
                st.session_state["case_id"] = f"CASE-{th['id']}"
                st.session_state["product_id"] = (
                    "LOAN" if "Loan" in th["subject"] else "LEASE"
                )
                st.session_state["customer_id"] = "CUST-ALPHA"
            except Exception:
                pass

            st.caption(th["body"])
            st.markdown("**Attachments**")
            # Two faux attachments; user supplies local file to upload
            for i, att in enumerate(["attachment-1.pdf", "attachment-2.pdf"], start=1):
                st.write(f"📎 {att}")
                up = st.file_uploader(
                    f"Attach local file for {att}",
                    type=["pdf", "txt"],
                    key=f"att_{th['id']}_{i}",
                )
                if up:
                    try:
                        r = requests.post(
                            f"{api_base}/upload",
                            files={
                                "file": (up.name, up.getbuffer(), "application/octet-stream")
                            },
                            timeout=120,
                        )
                        r.raise_for_status()
                        res = r.json()
                        st.success(f"Uploaded as doc_id {res['doc_id']}")
                        st.session_state["last_doc_id"] = res["doc_id"]
                    except Exception as e:
                        st.error(e)
    else:
        st.info("No demo threads exposed by the API.")

# =========================
# 0) FIBO Scope & Search
# =========================
st.header("0) FIBO Scope & Search")

col0a, col0b, col0c = st.columns([2, 2, 1])
with col0c:
    if st.button("🔄 Reindex FIBO"):
        try:
            r = requests.post(f"{api_base}/fibo/reindex?force=true", timeout=60)
            r.raise_for_status()
            st.success(f"Reindexed: {r.json().get('num_classes')} classes")
        except Exception as e:
            st.error(e)

# Namespaces
try:
    ns_list = requests.get(f"{api_base}/fibo/namespaces", timeout=15).json()
except Exception:
    ns_list = []

if ns_list:
    options = [f"{x['ns']} ({x['count']})" for x in ns_list]
    picked_ns = st.multiselect(
        "Namespaces", options=options, default=options[:6], key="ns_picked"
    )
    active_ns = [p.split(" (", 1)[0] for p in picked_ns]
else:
    st.info("No namespaces discovered — using ALL (try Reindex if this looks wrong).")
    active_ns = []

if st.button("Apply scope"):
    try:
        r = requests.post(
            f"{api_base}/fibo/scope", json={"namespaces": active_ns}, timeout=15
        )
        r.raise_for_status()
        st.success(f"Scope set ({len(active_ns) if active_ns else 'ALL'})")
    except Exception as e:
        st.error(e)

# Search
colS1, colS2 = st.columns([3, 1])
with colS1:
    q = st.text_input("Search FIBO (e.g., loan, lease, insurance, passport)", value="")
with colS2:
    hops = st.slider("Hops", 0, 4, 2, 1)

if st.button("Search FIBO"):
    try:
        r = requests.get(
            f"{api_base}/fibo/search",
            params={"q": q, "limit": 25, "fallback_all": True},
            timeout=20,
        )
        r.raise_for_status()
        st.session_state["fibo_hits"] = r.json()
        if not st.session_state["fibo_hits"]:
            st.warning("No classes matched anywhere.")
    except Exception as e:
        st.error(e)

# Results
for i, hit in enumerate(st.session_state.get("fibo_hits", [])):
    lbl = hit["label"] or hit["uri"].split("/")[-1]
    with st.expander(f"{lbl}", expanded=(i == 0)):
        st.code(hit["uri"])
        if st.button(f"Load subgraph for {lbl}", key=f"load_{i}"):
            try:
                r = requests.get(
                    f"{api_base}/fibo/subgraph",
                    params={
                        "focus": hit["uri"],
                        "hops": hops,
                        "include_properties": True,
                    },
                    timeout=30,
                )
                r.raise_for_status()
                subg = r.json()
                html = (
                    Path(__file__).resolve().parent.parent
                    / "components"
                    / "fibo_graph.html"
                ).read_text()
                import json as _json

                html = html.replace(
                    "window.graphData",
                    "window.graphData = " + _json.dumps(subg, ensure_ascii=False),
                )
                html = html.replace(
                    "window.apiBase", "window.apiBase = '" + api_base + "'"
                )
                st.components.v1.html(html, height=640)
                try:
                    requests.post(
                        f"{api_base}/ui/set_selection",
                        params={"uri": hit["uri"]},
                        timeout=5,
                    )
                except Exception:
                    pass
            except Exception as e:
                st.error(e)

# =========================
# 1) Upload & Embed
# =========================
st.header("1) Upload & Embed")
uploaded = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
if uploaded and st.button("Upload now"):
    try:
        r = requests.post(
            f"{api_base}/upload",
            files={"file": (uploaded.name, io.BytesIO(uploaded.read()), "application/octet-stream")},
            timeout=120,
        )
        r.raise_for_status()
        res = r.json()
        st.session_state["last_doc_id"] = res["doc_id"]
        st.success(f"Uploaded. doc_id = {res['doc_id']}")

        # Inline PDF preview
        if Path(uploaded.name).suffix.lower() == ".pdf":
            url = f"{api_base}/docs/raw/{res['doc_id']}"
            st.markdown(
                f'<iframe src="{url}" width="100%" height="480" style="border:1px solid #ddd;border-radius:8px;"></iframe>',
                unsafe_allow_html=True,
            )

        # Extracted text preview
        try:
            all_docs = requests.get(f"{api_base}/docs/list", timeout=10).json()
            meta = next((d for d in all_docs if d["doc_id"] == res["doc_id"]), {})
            txt = (meta.get("text") or "")[:4000]
            with st.expander("Extracted text (first ~4k chars)"):
                st.text(txt if txt else "(empty)")
        except Exception:
            pass

        st.subheader("Nearest neighbors")
        st.json(res.get("neighbors", []))
    except requests.HTTPError as e:
        try:
            st.error(e.response.json())
        except Exception:
            st.error(str(e))

# =========================
# 2) Label via FIBO
# =========================
st.header("2) Label via FIBO")

# Class Tree (optional, if present in your repo)
try:
    tr = requests.get(
        f"{api_base}/fibo/tree", params={"depth": 3, "scope_only": True}, timeout=20
    )
    tr.raise_for_status()
    tree = tr.json()
    html = (
        Path(__file__).resolve().parent.parent / "components" / "class_tree.html"
    ).read_text()
    import json as _json

    html = html.replace(
        "window.treeData", "window.treeData = " + _json.dumps(tree, ensure_ascii=False)
    )
    html = html.replace("window.apiBase", "window.apiBase = '" + api_base + "'")
    st.components.v1.html(html, height=460)
except Exception as e:
    st.info(f"Tree view unavailable: {e}")

# Fallback dropdown
try:
    classes = requests.get(f"{api_base}/fibo/classes", timeout=15).json()
except Exception:
    classes = []

label_to_uri = {(c["label"] or c["uri"].split("/")[-1]): c["uri"] for c in classes}
labels = sorted(label_to_uri.keys())

pre_uri = None
try:
    sel = requests.get(f"{api_base}/ui/get_selection", timeout=5).json().get("uri")
    if sel:
        pre_uri = sel
except Exception:
    pass

pre_label = None
if pre_uri:
    for lbl, u in label_to_uri.items():
        if u == pre_uri:
            pre_label = lbl
            break

chosen_label = st.selectbox(
    "FIBO class (fallback)", labels, index=(labels.index(pre_label) if pre_label in labels else 0) if labels else 0
)
chosen_uri = label_to_uri.get(chosen_label)
st.caption(f"Selected URI: {chosen_uri or '—'}")

doc_id_for_link = st.text_input("doc_id to link", value=st.session_state.get("last_doc_id", ""))

c1, c2, c3 = st.columns(3)
if c1.button("Link now", disabled=not (chosen_uri and doc_id_for_link)):
    try:
        r = requests.post(
            f"{api_base}/label/link",
            json={"doc_id": doc_id_for_link, "fibo_class_uri": chosen_uri},
            timeout=30,
        )
        r.raise_for_status()
        res = r.json()
        st.success("Linked ✅")
        st.code(res.get("rdf_turtle", ""), language="turtle")
    except Exception as e:
        st.error(e)

if c2.button("Load FIBO schema", disabled=not chosen_uri):
    try:
        sch = requests.get(
            f"{api_base}/fibo/schema", params={"class_uri": chosen_uri}, timeout=20
        ).json()
        st.session_state["schema"] = sch
        st.success(f"Loaded schema with {sch.get('count', 0)} properties")
    except Exception as e:
        st.error(e)

if c3.button("Extract (simple) + coverage", disabled=not (doc_id_for_link and chosen_uri)):
    try:
        res = requests.post(
            f"{api_base}/extract/simple",
            params={"doc_id": doc_id_for_link, "class_uri": chosen_uri},
            timeout=30,
        ).json()
        st.session_state["extract_res"] = res
        st.success(f"Coverage: {round(res.get('coverage', 0) * 100, 1)}%")
    except Exception as e:
        st.error(e)

# =========================
# 2.5) FIBO Attributes → Extraction (ontology-driven)
# =========================
st.markdown("### FIBO Attributes → Extraction")
colA, colB, colC = st.columns([1, 1, 1])

# Use current selection if available
try:
    sel_uri = requests.get(f"{api_base}/ui/get_selection", timeout=5).json().get("uri")
except Exception:
    sel_uri = None

class_uri_for_attr = st.text_input(
    "FIBO class URI (auto-filled on graph/tree select)", value=sel_uri or chosen_uri or ""
)
doc_id_for_attr = st.text_input(
    "doc_id for attribute extraction", value=st.session_state.get("last_doc_id", "")
)

if colA.button("Load Attributes", disabled=not class_uri_for_attr):
    try:
        attrs = requests.get(
            f"{api_base}/fibo/attributes",
            params={"class_uri": class_uri_for_attr},
            timeout=30,
        ).json()
        st.session_state["attrs"] = attrs
        st.success(f"Loaded {attrs.get('count', 0)} attributes")
        with st.expander("Attributes (labels & synonyms)"):
            st.json(attrs)
    except Exception as e:
        st.error(e)

if colB.button("Extract by Schema", disabled=not (class_uri_for_attr and doc_id_for_attr)):
    try:
        res = requests.post(
            f"{api_base}/extract/by_schema",
            params={"doc_id": doc_id_for_attr, "class_uri": class_uri_for_attr},
            timeout=60,
        ).json()
        st.session_state["by_schema"] = res
        st.success(f"Coverage: {round(res.get('coverage', 0) * 100, 1)}%")
    except Exception as e:
        st.error(e)

if colC.button("Show Side-by-Side", disabled=("by_schema" not in st.session_state)):
    res = st.session_state.get("by_schema", {})
    fields = res.get("fields", {})
    if not fields:
        st.info("No extracted fields.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**FIBO Properties**")
            st.text("\n".join(list(fields.keys())[:200]))
        with right:
            st.markdown("**Best Values (heuristic)**")
            pretty = {k: v.get("best") for k, v in fields.items()}
            st.json(pretty)

# =========================
# 3) Documents
# =========================
st.header("3) Documents")
try:
    data = requests.get(f"{api_base}/docs/list", timeout=10).json()
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.markdown("#### Drill‑down")
        doc_pick = st.selectbox("Pick a doc_id to inspect", [d["doc_id"] for d in data])
        if doc_pick:
            try:
                n = requests.get(
                    f"{api_base}/neighbors",
                    params={"doc_id": doc_pick, "topk": 8},
                    timeout=10,
                ).json()
                st.subheader("Nearest neighbors")
                st.json(n)
            except Exception:
                pass
            try:
                t = requests.get(f"{api_base}/rdf/{doc_pick}", timeout=10).json().get(
                    "turtle", ""
                )
                st.subheader("RDF")
                st.code(t, language="turtle")
            except Exception:
                st.info("No RDF yet (not labeled).")
            # Auto schema/extraction if labeled
            sel_doc = next((d for d in data if d["doc_id"] == doc_pick), {})
            fibo_uri = sel_doc.get("fibo_class_uri")
            if fibo_uri:
                try:
                    sch = requests.get(
                        f"{api_base}/fibo/schema",
                        params={"class_uri": fibo_uri},
                        timeout=20,
                    ).json()
                    st.markdown("**Schema for labeled class**")
                    st.json(sch)
                    ex = requests.post(
                        f"{api_base}/extract/simple",
                        params={"doc_id": doc_pick, "class_uri": fibo_uri},
                        timeout=30,
                    ).json()
                    st.markdown("**Extraction coverage for this doc**")
                    st.json(ex)
                except Exception:
                    pass
    else:
        st.info("No documents yet.")
except Exception:
    st.info("Docs list unavailable.")
