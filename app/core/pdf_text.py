from __future__ import annotations
import io, base64
from typing import List, Dict, Any
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. pip install pymupdf") from e

def _page_text_and_spans(page: "fitz.Page") -> Dict[str, Any]:
    blocks = page.get_text("dict")["blocks"]
    lines_out = []
    for b in blocks:
        if "lines" not in b: continue
        for line in b["lines"]:
            txt = "".join([s.get("text","") for s in line.get("spans",[]) ]).strip()
            if not txt: continue
            bbox = line.get("bbox", None)
            spans = []
            for s in line.get("spans", []):
                spans.append({
                    "text": s.get("text",""),
                    "bbox": s.get("bbox", None),
                    "size": s.get("size", None),
                })
            lines_out.append({"text": txt, "bbox": bbox, "spans": spans})
    text_all = "\n".join(ln["text"] for ln in lines_out)
    return {"text": text_all, "lines": lines_out}

def extract_text_blocks(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    try:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            data = _page_text_and_spans(page)
            out.append({"page": pno, "text": data["text"], "spans": data["lines"]})
    finally:
        doc.close()
    return out

def get_page_images(pdf_bytes: bytes, zoom: float=2.0) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images

def get_page_images_as_base64(pdf_bytes: bytes, zoom: float=2.0) -> List[str]:
    out=[]
    for raw in get_page_images(pdf_bytes, zoom=zoom):
        b64 = base64.b64encode(raw).decode("ascii")
        out.append("data:image/png;base64,"+b64)
    return out

def focused_summary(text: str, max_chars: int=2000) -> str:
    if not text: return ""
    lines = text.splitlines()
    head = "\n".join(lines[:60])
    if len(head) > max_chars: return head[:max_chars]
    if len(text) <= max_chars: return text
    return text[:max_chars]
