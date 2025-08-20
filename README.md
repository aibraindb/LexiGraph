# LexiGraph 13.2 — Six‑Button ECM + FIBO MVP

This package gives you a working Streamlit app with the six actions:
1) Link Now
2) Load FIBO Schema
3) Extract + Coverage
4) Load Attributes
5) Extract By Schema
6) Show Side By Side

CPU‑only. No OCR dependency. Uses PyMuPDF for text, TF‑IDF for FIBO + document vectors.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```
Optional: drop your full ontology at `data/fibo_full.ttl` or upload it via sidebar.
