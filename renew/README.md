# Lexi — OCR Tree & Canvas (HITL)

This drop focuses on **reliable bounding boxes + lassoing + HITL** without Tesseract.

## What’s inside

- `app/core/ocr_indexer.py` — indexes PDFs into pages/lines with bboxes using **pdfplumber** (preferred) and falls back to **PyMuPDF** if available. Produces a clean dict structure the UI consumes.
- `ui/ocr_tree_canvas.py` — Streamlit UI: **Tree ⇄ Canvas two-way linking**, move/resize boxes, save to `data/ocr_cache/page_XXX.json`.
- `scripts/nuke_and_build.py` — reset caches.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install streamlit streamlit-drawable-canvas pdfplumber pillow
# (Optional) If you want PyMuPDF fallback:
# pip install pymupdf
