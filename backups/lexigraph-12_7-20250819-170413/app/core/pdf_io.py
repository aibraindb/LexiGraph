from __future__ import annotations
from typing import Tuple, List, Dict, Any
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

def _render_matrix(page: fitz.Page, dpi: int) -> fitz.Matrix:
    zoom = dpi / 72.0
    return fitz.Matrix(zoom, zoom).preRotate(page.rotation)

def render_page(path: Path, page_no: int = 0, dpi: int = 220) -> Tuple[Image.Image, fitz.Matrix]:
    doc = fitz.open(str(path))
    page = doc[page_no]
    mat = _render_matrix(page, dpi)
    pix = page.get_pixmap(matrix=mat, alpha=False, annots=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img, mat

def page_count(path: Path) -> int:
    doc = fitz.open(str(path))
    n = len(doc)
    doc.close()
    return n

def page_words_pixels(path: Path, page_no: int, mat: fitz.Matrix) -> List[Dict[str, Any]]:
    doc = fitz.open(str(path))
    page = doc[page_no]
    words = page.get_text("words") or []  # (x0,y0,x1,y1, txt, block, line, word_no)
    out = []
    for (x0,y0,x1,y1,txt,*_) in words:
        r = fitz.Rect(x0, y0, x1, y1)
        rp = mat * r  # transform into rendered pixel space
        out.append({"bbox":[float(rp.x0), float(rp.y0), float(rp.x1), float(rp.y1)], "text": txt})
    doc.close()
    return out

def draw_boxes(img: Image.Image, boxes: List[Dict[str, Any]], max_boxes:int=200) -> Image.Image:
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for i, b in enumerate(boxes[:max_boxes]):
        x0,y0,x1,y1 = b["bbox"]
        dr.rectangle([x0,y0,x1,y1], outline=(0,180,255), width=2)
    return im
