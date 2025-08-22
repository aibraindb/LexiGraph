These modules power the Streamlit UI:

- `pdf_text.py`    — robust text + spans + page images via PyMuPDF (no OCR)
- `fibo_index.py`  — parse FIBO TTL, index classes/edges, search & subgraph
- `fibo_attrs.py`  — collect properties (labels+synonyms) for a FIBO class via rdfs:domain
- `fibo_vec.py`    — TF-IDF vectors over FIBO classes for fuzzy matching
- `index_docs.py`  — simple TF-IDF index for uploaded documents (MVP)

After copying your `data/fibo_full.ttl`, run:

    python -m app.core.fibo_index --rebuild
    python -m app.core.fibo_vec   --rebuild

Then start the Streamlit UI.
