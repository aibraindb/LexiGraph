# --- bootstrap package path for Streamlit ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -------------------------------------------
import streamlit as st
from pathlib import Path
import json

from app.core.pdf_text import extract_text_blocks, focused_summary
from app.core.fibo_vec import build_fibo_vec, search_fibo
from app.core.fibo_attrs import get_attributes_for_class
from app.core.value_mapper import map_values_to_attributes
from app.api.ecm_stub import ecm_classify

st.set_page_config(page_title="LexiGraph — Six‑Button ECM + FIBO", layout="wide")
st.title("LexiGraph — Six‑Button ECM + FIBO")

# --- Sidebar: FIBO management ---
with st.sidebar:
    st.header("FIBO / Index")
    fibo_file = st.file_uploader("Load FIBO TTL (optional)", type=["ttl"], key="fibo_up")
    if fibo_file:
        Path("data").mkdir(exist_ok=True, parents=True)
        Path("data/fibo_full.ttl").write_bytes(fibo_file.read())
        st.success("Saved data/fibo_full.ttl")
    if st.button("Rebuild FIBO Vectors"):
        try:
            build_fibo_vec(force=True)
            st.success("FIBO vector index rebuilt.")
        except Exception as e:
            st.error(f"Vector build failed: {e}")
    st.divider()
    st.caption("Tip: Drop your full ontology as data/fibo_full.ttl for best results.")

# Session state init
for k,v in {
    "doc_bytes": None,
    "doc_name": None,
    "extract": None,
    "summary": None,
    "candidates": [],
    "chosen_class": None,
    "attributes": None,
    "mapped": None,
}.items():
    st.session_state.setdefault(k,v)

# --- Upload ---
st.subheader("Upload")
up = st.file_uploader("PDF document", type=["pdf"], key="pdf_up")
if up:
    st.session_state["doc_bytes"] = up.read()
    st.session_state["doc_name"] = up.name
    try:
        st.session_state["extract"] = extract_text_blocks(st.session_state["doc_bytes"])
        st.success(f"Parsed: {up.name}  • Pages: {len(st.session_state['extract']['pages'])}  • KV lines: {len(st.session_state['extract']['colon_lines'])}")
    except Exception as e:
        st.error(f"PDF parse failed: {e}")

cols = st.columns(6)
# 1) Link Now
if cols[0].button("1) Link Now"):
    ext = st.session_state.get("extract")
    if not ext:
        st.warning("Upload a PDF first.")
    else:
        summ = focused_summary(ext)
        st.session_state["summary"] = summ
        # local vector search
        try:
            hits = search_fibo(summ, topk=12)
        except Exception as e:
            hits = []
            st.error(f"FIBO search failed: {e}")
        # optional ECM consult
        consult_ecm = st.checkbox("Consult ECM", value=True, key="consult_ecm")
        if consult_ecm:
            ec = ecm_classify(summ)
            if ec:
                # bump scores for matching labels
                for h in hits:
                    if any(ec["label"] in lb.lower() for lb in h.get("labels", [])):
                        h["score"] = max(h["score"], ec["score"])
        # sort by score and keep top 8
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)[:8]
        st.session_state["candidates"] = hits
        if hits:
            st.session_state["chosen_class"] = hits[0]["uri"]
        st.success("Candidates loaded.")

# 2) Load FIBO Schema
if cols[1].button("2) Load FIBO Schema"):
    uri = st.session_state.get("chosen_class")
    if not uri:
        st.warning("Run 'Link Now' and pick a class first.")
    else:
        try:
            st.session_state["attributes"] = get_attributes_for_class(uri)
            st.success(f"Loaded {st.session_state['attributes']['count']} attributes.")
        except Exception as e:
            st.error(f"Schema load failed: {e}")

# 3) Extract + Coverage
if cols[2].button("3) Extract + Coverage"):
    ext = st.session_state.get("extract")
    attrs = st.session_state.get("attributes") or {"attributes":[],"count":0}
    if not ext or not attrs["attributes"]:
        st.warning("Need a document and loaded schema.")
    else:
        mapped = map_values_to_attributes(ext["full_text"], attrs["attributes"], kv_pairs=ext.get("kv_pairs"), threshold=0.45)
        st.session_state["mapped"] = mapped
        st.success(f"Mapped {len(mapped)}/{attrs['count']} attributes.")

# 4) Load Attributes (detail)
if cols[3].button("4) Load Attributes"):
    uri = st.session_state.get("chosen_class")
    if not uri:
        st.warning("Pick a class first.")
    else:
        try:
            st.session_state["attributes"] = get_attributes_for_class(uri)
            st.success("Attribute metadata refreshed.")
        except Exception as e:
            st.error(f"Load attributes failed: {e}")

# 5) Extract By Schema (guided)
if cols[4].button("5) Extract By Schema"):
    ext = st.session_state.get("extract")
    attrs = st.session_state.get("attributes") or {"attributes":[],"count":0}
    if not ext or not attrs["attributes"]:
        st.warning("Need a document and loaded schema.")
    else:
        # Guided = slightly higher threshold + prefer colon lines
        mapped = map_values_to_attributes(ext["full_text"], attrs["attributes"], kv_pairs=ext.get("kv_pairs"), threshold=0.55)
        st.session_state["mapped"] = mapped
        st.success(f"Guided extraction mapped {len(mapped)}/{attrs['count']} attributes.")

# 6) Show Side By Side
if cols[5].button("6) Show Side By Side"):
    st.session_state["show_sbs"] = True

# --- Candidates & selection ---
cands = st.session_state.get("candidates", [])
if cands:
    st.markdown("### Candidates")
    for i,h in enumerate(cands,1):
        label = (h["labels"][0] if h["labels"] else h["uri"].split("/")[-1])
        st.write(f"{i}. **{label}** — {h['uri']} (score {h['score']:.3f})")
    st.session_state["chosen_class"] = st.selectbox("Chosen FIBO class", [h["uri"] for h in cands], index=0, key="chosen_uri")

# --- Panels ---
pan1, pan2, pan3 = st.columns([1,1,1])
with pan1:
    st.subheader("Document Preview (text)")
    ext = st.session_state.get("extract")
    if ext and ext["pages"]:
        st.code(ext["pages"][0]["text"][:2400])
    if st.session_state.get("summary"):
        with st.expander("Focused Summary"):
            st.code(st.session_state["summary"])

with pan2:
    st.subheader("FIBO Schema")
    attrs = st.session_state.get("attributes")
    if attrs:
        st.caption(f"{attrs['count']} attributes")
        st.json(attrs)
    else:
        st.info("Click '2) Load FIBO Schema'")

with pan3:
    st.subheader("Extraction & Coverage")
    attrs = st.session_state.get("attributes") or {"count":0,"attributes":[]}
    mapped = st.session_state.get("mapped") or {}
    total = max(1, attrs["count"])
    cov = len(mapped)/total*100 if attrs["count"] else 0.0
    st.metric("Coverage", f"{len(mapped)}/{attrs['count']}", f"{cov:.1f}%")
    if mapped:
        st.json(mapped)

# --- Download result JSON ---
if st.session_state.get("extract") and st.session_state.get("chosen_class"):
    result = {
        "document": st.session_state.get("doc_name"),
        "summary": st.session_state.get("summary"),
        "fibo_class": st.session_state.get("chosen_class"),
        "candidates": st.session_state.get("candidates"),
        "attributes_count": (st.session_state.get("attributes") or {}).get("count", 0),
        "mapped_values": st.session_state.get("mapped") or {}
    }
    st.download_button("Download Result JSON", data=json.dumps(result, indent=2), file_name="lexigraph_result.json", mime="application/json")
