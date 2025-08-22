# app/core/ocr_indexer.py
import io
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF


@dataclass
class OCRIndex:
    pages: List[Dict[str, Any]]


def _page_image_bytes(page: fitz.Page, scale: float = 2.0) -> bytes:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def index_pdf_bytes(pdf_bytes: bytes) -> OCRIndex:
    """Return a structured OCR index with image bytes + normalized blocks/lines/words."""
    doc = fitz.open("pdf", pdf_bytes)
    pages: List[Dict[str, Any]] = []

    for pnum, page in enumerate(doc):
        # Page image
        img_bytes = _page_image_bytes(page)

        # Rich structure
        raw = page.get_text("rawdict")  # blocks -> lines -> spans
        blocks = []
        lines = []

        for bi, b in enumerate(raw.get("blocks", [])):
            bbox_b = b.get("bbox", [0, 0, 0, 0])
            blocks.append({
                "id": f"p{pnum}_b{bi}",
                "bbox": bbox_b,
            })
            for li, ln in enumerate(b.get("lines", [])):
                # concatenate all span texts of the line
                text = "".join(s.get("text", "") for s in ln.get("spans", []))
                bbox_l = ln.get("bbox", [0, 0, 0, 0])
                lines.append({
                    "id": f"p{pnum}_b{bi}_l{li}",
                    "text": text.strip(),
                    "bbox": bbox_l,
                    "block_index": bi,
                    "line_index": li,
                })

        # Words (PyMuPDF returns tuples)
        words_list = []
        for wi, w in enumerate(page.get_text("words")):
            # (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            x0, y0, x1, y1, word, bno, lno, wno = w
            words_list.append({
                "id": f"p{pnum}_w{wi}",
                "text": word,
                "bbox": [x0, y0, x1, y1],
                "block_index": int(bno),
                "line_index": int(lno),
                "word_index": int(wno),
            })

        pages.append({
            "number": pnum,
            "image": img_bytes,             # ✅ Streamlit can display this
            "size": [float(page.rect.width), float(page.rect.height)],
            "blocks": blocks,               # ✅ normalized dicts
            "lines": lines,                 # ✅ normalized dicts
            "words": words_list,            # ✅ normalized dicts
        })

    return OCRIndex(pages=pages)
