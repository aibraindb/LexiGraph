# LexiGraph 15.0

**What's new vs 14.x**
- Reintroduced **D3 FIBO viewer** with maximize + tooltips (`components/fibo_graph.html`)
- Added **FIBO index/search/subgraph** endpoints (API)
- Added **Document Library**: embeddings of uploaded docs (value-stripped), pairwise distances, k‑NN, nearest FIBO
- End‑to‑end: Upload → Propose schema → Apply → Annotate evidence → View neighbors

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: put the full FIBO ttl at data/fibo_full.ttl
# then:
uvicorn app.api.main:app --reload --port 8000

# separate terminal:
streamlit run ui/streamlit_app.py
```

Artifacts saved to `data/runs/<doc_id>/`.

## Notes
- Uses PyMuPDF for text + bounding boxes (digital PDFs).
- For scanned PDFs, plug your OCR and fill spans; pipeline remains.
- FIBO vectors: click **Rebuild FIBO vectors** after replacing TTL.
