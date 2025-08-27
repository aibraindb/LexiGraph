
# fin-docai-v1 — Financial Document AI (v1 Prototype)

- Digital PDF words + bounding boxes (pdfplumber)
- Heuristic extractors for **invoice** & **bank statement**
- **Validation** (invoice sum vs total; statement reconciliation)
- **HITL** Streamlit viewer with word boxes + editable fields + audit log
- **Synthetic PDFs** generator
- **Wells Fargo** public PDF downloader (+ hashes)
- **Cross-PDF entity graph** demo (connected components + MST)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# sample PDFs
python scripts/generate_samples.py

# HITL UI
streamlit run streamlit_app.py

# optional Wells Fargo PDFs
python scripts/fetch_public_wf_docs.py

# entity graph demo
python scripts/build_entity_graph.py --folder data/samples/wells
streamlit run streamlit_graph.py
```
