# ui/embeddings_view.py (new)
import json, numpy as np, streamlit as st
from sklearn.decomposition import PCA

st.title("Embeddings Explorer")
# Expect a JSONL dump with: {"doc_id": "...", "text": "...", "type":"..."}
uploaded = st.file_uploader("Upload embeddings dump (JSONL with text & type)", type=["jsonl","txt"])
if uploaded:
    rows = [json.loads(l) for l in uploaded.read().decode("utf-8").splitlines() if l.strip()]
    labels = [r.get("type","?") for r in rows]
    # same embedding fn as server (hash-based MVP)
    def embed(t):
        rng = np.random.default_rng(abs(hash(t)) % (2**32)); v = rng.normal(size=256)
        v = v / (np.linalg.norm(v) or 1.0); return v
    X = np.stack([embed(r.get("text","")) for r in rows])
    x2 = PCA(2).fit_transform(X)
    import plotly.express as px
    fig = px.scatter(x=x2[:,0], y=x2[:,1], color=labels, hover_data={"label":labels})
    st.plotly_chart(fig, use_container_width=True)
