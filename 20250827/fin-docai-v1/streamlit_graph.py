
import os, json, subprocess, sys
import streamlit as st
from pathlib import Path
st.set_page_config(page_title="fin-docai-v1 Graph", layout="wide")
st.title("Cross-Document Entity Graph (demo)")
folder = st.text_input("PDF folder", value=str(Path("data/samples/wells")))
outdir = Path("data/labels/entity_graph"); outdir.mkdir(parents=True, exist_ok=True)
if st.button("Build Graph"):
    cmd=[sys.executable, "scripts/build_entity_graph.py", "--folder", folder, "--out", str(outdir)]
    with st.spinner("Building..."): proc=subprocess.run(cmd, capture_output=True, text=True)
    st.code(proc.stdout + "\n" + proc.stderr)
jp = outdir/"entity_graph.json"; hp = outdir/"entity_graph.html"
c1,c2 = st.columns(2)
with c1:
    st.markdown("### Graph JSON")
    st.json(json.load(open(jp))) if jp.exists() else st.info("Click Build Graph")
with c2:
    st.markdown("### Visualization")
    if hp.exists():
        import streamlit.components.v1 as components
        components.html(Path(hp).read_text(encoding="utf-8"), height=720, scrolling=True)
    else: st.info("Install pyvis then rebuild to see interactive graph")
