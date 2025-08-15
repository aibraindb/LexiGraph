
# LexiGraph v9 (Demo)
- FIBO indexer (owl:Class + rdfs:Class, multi-format)
- Namespace scope + full-text FIBO search with fallback
- D3 subgraph viewer (maximize, fit, tooltips)
- Class Tree Picker (dblclick to select; hover tooltips)
- Upload preview (PDF iframe) + extracted text preview
- Embeddings: TF-IDF + KNN
- Email Context Simulator + Context Association (CA)
- Schema endpoint + simple extractor + coverage

## Run
### API
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
# http://127.0.0.1:8000/health

### UI
pip install -r requirements-ui.txt
streamlit run ui/streamlit_app.py
# Sidebar → API base: http://127.0.0.1:8000

Place full FIBO at data/fibo_full.ttl (optional). Sample PDFs under data/samples/<Category>/...pdf.
