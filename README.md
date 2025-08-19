# LexiGraph 12.7 (CPU-only OCR, TF-IDF embeddings, FIBO coverage)

Minimal, self-contained demo:
- Digital-text first (PyMuPDF), OCR fallback (EasyOCR CPU)
- TF-IDF embeddings (no sentence-transformers)
- FIBO index + tolerant attribute resolution
- Streamlit single UI

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

For a fuller ontology, drop `data/fibo_full.ttl` and use the sidebar to **Rebuild FIBO index**.
