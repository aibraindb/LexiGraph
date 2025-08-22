# Lexi OCR Tree (v2)

A focused, shippable prototype for **OCR → bounding-box lineage → HITL lasso** with
a **mock ECM extraction payload** and **simple TF‑IDF vector stores** for documents
and FIBO classes.

## Features
- Upload a PDF (digital or scanned).
- OCR indexer builds **page → block → line** tree with bounding boxes and text.
  - Uses **PyMuPDF** when available, falls back to **pdfminer + pdf2image**.
- Right pane: **canvas viewer** with all line boxes overlaid. Switch between
  **Select/Move** and **Draw (lasso/rect)**. Two‑way linking with the tree.
- Mock ECM payload (JSON) — click a field to **jump to the best matching line**;
  adjust with lasso; edits tracked; save lineage to JSON.
- Simple **TF‑IDF vector stores** (docs + FIBO) saved as `.npy` and `.joblib`.
- Everything saved to `data/` (OCR index, session edits, vectors).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# (Optional) Install system dep for pdf2image:
# macOS (brew): brew install poppler
# Windows: download poppler binaries and add to PATH

# Run the UI
streamlit run ui/ocr_tree_canvas.py
```

Drop a **PDF** into the uploader. Use the **Document Tree** to navigate pages / blocks / lines.
Toggle “Show ALL line boxes”. Switch canvas mode to **Move/Resize** or **Draw**.

### ECM Mock
- The app loads `ecm_payloads/sample_invoice.json` by default.
- Click any field; we fuzzy find the best OCR line match and jump to it.
- Adjust the box and **Save edit**. The lineage is stored in `data/sessions/<doc_id>.json`.

### Vector Stores (Docs + FIBO)

Build / rebuild from the UI sidebar, or via CLI:

```bash
# Build FIBO vector store (use your fibo_full.ttl placed in data/)
python tools/build_fibo_index.py --ttl data/fibo_full.ttl

# Index a folder of PDFs (text only summaries for now)
python tools/index_docs.py --path samples
```

Vectors saved in `data/vectors/`.

## Mermaid: End‑to‑End (ECM + OCR + HITL)

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst
    participant UI as Lexi UI (Streamlit)
    participant OCR as OCR Indexer (PyMuPDF → fallback)
    participant ECM as ECM Extractor (Prompt Catalog / mock)
    participant Link as AutoLink & Validator
    participant FIBO as FIBO Store (TF‑IDF)
    participant Store as Data Store (files)

    User->>UI: Upload PDF
    UI->>OCR: Build page/block/line index + render images
    OCR-->>UI: OCRIndex JSON (pages, blocks, lines, bboxes)

    par
      UI->>ECM: Send base64 PDF (or doc_id)
      ECM-->>UI: Extraction JSON (field, value, confidence)
    and
      UI->>FIBO: Fuzzy search schemas from title/summary
      FIBO-->>UI: Candidate classes + attributes
    end

    UI->>Link: Match ECM values ↔ OCR lines (text + locality)
    Link-->>UI: Proposed placements (with confidence)

    User->>UI: Inspect, move/resize, or draw (lasso) on canvas
    UI->>Store: Save edit {field_id, page, bbox, method}

    loop Iterate until all fields OK
      User->>UI: Approve / correct per field
      UI->>Store: Upsert value + lineage
    end

    UI-->>User: Result JSON (values + bbox lineage)
    UI->>Store: Persist OCR index, edits, vectors, mappings
```

## Files
- `ui/ocr_tree_canvas.py` – the Streamlit app
- `app/core/ocr_indexer.py` – OCR + PDF parsing
- `app/core/models.py` – dataclasses
- `app/core/vector_store.py` – TF‑IDF saving / loading
- `app/core/fibo_index.py` – minimal FIBO loader + vectorizer
- `app/core/value_mapper.py` – fuzzy field→line mapping
- `ecm_payloads/sample_invoice.json` – mock ECM output
- `tools/build_fibo_index.py` – build FIBO vectors
- `tools/index_docs.py` – index documents
- `tools/nuke_and_build.py` – clean & rebuild helper

## Notes
- If **PyMuPDF** fails to import on macOS → ensure no system `mupdf` conflicts.
  Fallback parser (pdfminer+pdf2image) will still work.
- For **pdf2image**, install poppler (see Quickstart).

Enjoy!
