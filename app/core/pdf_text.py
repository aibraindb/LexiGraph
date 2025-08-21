from __future__ import annotations
from typing import Dict, List
import fitz  # PyMuPDF
import base64

def _page_image_b64(page: "fitz.Page", zoom: float = 1.5) -> str:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode("ascii")

def extract_text_blocks(file_bytes: bytes, zoom: float = 1.5) -> Dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    full_text = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        img_b64 = _page_image_b64(page, zoom=zoom)
        d = page.get_text("dict")
        spans_out = []
        text_collect = []
        for b_idx, block in enumerate(d.get("blocks", [])):
            if block.get("type", 0) != 0:
                continue
            for l_idx, line in enumerate(block.get("lines", [])):
                for s_idx, span in enumerate(line.get("spans", [])):
                    bbox = span.get("bbox") or line.get("bbox") or block.get("bbox")
                    txt = (span.get("text") or "").strip()
                    if not txt:
                        continue
                    spans_out.append({
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "text": txt,
                        "block": b_idx,
                        "line": l_idx,
                        "span": s_idx
                    })
                    text_collect.append(txt)
        pages.append({"text": "\n".join(text_collect), "image_b64": img_b64, "spans": spans_out})
        full_text.append("\n".join(text_collect))
    return {"pages": pages, "full_text": "\n\n".join(full_text)}
