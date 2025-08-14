import streamlit as st, requests, io, json, pandas as pd

st.set_page_config(page_title="LexiGraph Simple v3", layout="wide")
st.title("LexiGraph — Alpha Path (Embed → Link FIBO)")

with st.sidebar:
    st.markdown("### 🔌 API")
    api_base = st.text_input("API base URL", value=st.session_state.get("api_base","http://127.0.0.1:8000"))
    st.session_state["api_base"] = api_base

    st.markdown("---")
    st.markdown("### 🔎 Search")
    q = st.text_input("Query", "")
    if st.button("Search"):
        try:
            r = requests.post(f"{api_base}/search", json={"query": q, "topk": 8}, timeout=30); r.raise_for_status()
            st.session_state["search_results"] = r.json()
        except Exception as e:
            st.error(e)
    if "search_results" in st.session_state:
        st.json(st.session_state["search_results"])

st.markdown("## 1) Upload → Embed → Neighbors")
up = st.file_uploader("Upload a PDF or TXT", type=["pdf","txt"])
if up and st.button("Upload & Embed"):
    try:
        r = requests.post(f"{api_base}/upload", params={"topk":5},
                          files={"file": (up.name, io.BytesIO(up.read()), "application/octet-stream")},
                          timeout=180)
        r.raise_for_status()
        res = r.json()
        st.success(f"Uploaded: {res['doc_id']} ({res['filename']})")
        st.code(res.get("preview","") or "", language="text")
        st.subheader("Nearest neighbors")
        st.json(res.get("neighbors", []))
        st.session_state["last_doc_id"] = res["doc_id"]
    except requests.HTTPError as e:
        try:
            st.error(e.response.json())
        except Exception:
            st.error(str(e))
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.markdown("## 2) Link to FIBO & (optional) Extract")
doc_id = st.text_input("doc_id", value=st.session_state.get("last_doc_id",""))
cols = st.columns(3)
with cols[0]:
    if st.button("Load classes"):
        try:
            r = requests.get(f"{api_base}/fibo/classes", timeout=20); r.raise_for_status()
            st.session_state["fibo_classes"] = r.json()
            st.success(f"Loaded {len(st.session_state['fibo_classes'])} classes")
        except Exception as e:
            st.error(e)

labels = [c["label"] for c in st.session_state.get("fibo_classes", [])]
if not labels:
    labels = ["Invoice"]
label = st.selectbox("Select FIBO class (label)", options=labels)
extract = st.checkbox("Attempt simple extraction", value=True)

if st.button("Link now"):
    try:
        payload = {"doc_id": doc_id, "class_label": label, "extract": extract}
        r = requests.post(f"{api_base}/fibo/link", json=payload, timeout=60); r.raise_for_status()
        out = r.json()
        st.success("Linked and RDF generated")
        st.json({"class_uri": out.get("class_uri"), "fields": out.get("fields")})
        st.subheader("RDF (Turtle)")
        st.code(out.get("rdf_turtle",""), language="turtle")
    except requests.HTTPError as e:
        try:
            st.error(e.response.json())
        except Exception:
            st.error(str(e))
    except Exception as e:
        st.error(str(e))

st.markdown("---")
st.subheader("📚 Documents")
try:
    rr = requests.get(f"{api_base}/docs/list", timeout=10); rr.raise_for_status()
    data = rr.json()
    if data:
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.info("No documents yet.")
except Exception as e:
    st.info("Docs listing not available yet.")
