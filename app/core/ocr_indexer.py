from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import io
import fitz  # PyMuPDF
from PIL import Image
import base64

@dataclass
class Word:
    text: str
    bbox: List[float]  # [x0,y0,x1,y1]

@dataclass
class Line:
    text: str
    bbox: List[float]
    words: List[Word]

@dataclass
class Block:
    bbox: List[float]
    lines: List[Line]

@dataclass
class PageIndex:
    number: int
    width: int
    height: int
    image_b64: str     # PNG base64
    blocks: List[Block]

def _pix_to_b64(pix: fitz.Pixmap) -> str:
    img_bytes = pix.tobytes("png")
    return base64.b64encode(img_bytes).decode("ascii")

def index_pdf_bytes(pdf_bytes: bytes, max_w: int = 1400) -> Dict[str, Any]:
    """
    Returns: {"pages": [ PageIndex... ], "meta": {...} }
    Uses PyMuPDF text extraction only (no Tesseract).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[PageIndex] = []

    for i, page in enumerate(doc):
        # Render page image (scaled so width <= max_w)
        zoom = min(2.0, max(1.0, max_w / max(page.rect.width, 1)))
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image_b64 = _pix_to_b64(pix)
        scale = zoom  # how coords were scaled for the image

        blocks: List[Block] = []
        for b in page.get_text("blocks"):  # (x0, y0, x1, y1, "text", block_no, block_type)
            x0, y0, x1, y1, *_ = b
            # gather lines+words inside this block via "dict" mode to keep structure
            b_lines: List[Line] = []
            block_rect = fitz.Rect(x0, y0, x1, y1)
            # clip to the block so we only fetch its own lines
            d = page.get_text("dict", clip=block_rect)
            for ld in d.get("blocks", []):
                for sp in ld.get("lines", []):
                    wds: List[Word] = []
                    line_text_parts = []
                    # words in this line
                    for wd in sp.get("spans", []):
                        txt = wd.get("text", "")
                        line_text_parts.append(txt)
                        # PyMuPDF spans have bbox in device space; convert to page rect
                        rx0, ry0, rx1, ry1 = wd["bbox"]
                        wds.append(Word(text=txt, bbox=[rx0, ry0, rx1, ry1]))
                    if not wds:
                        continue
                    l_text = " ".join([t for t in line_text_parts if t])
                    ly0 = min(w.bbox[1] for w in wds); lx0 = min(w.bbox[0] for w in wds)
                    ly1 = max(w.bbox[3] for w in wds); lx1 = max(w.bbox[2] for w in wds)
                    b_lines.append(Line(text=l_text, bbox=[lx0, ly0, lx1, ly1], words=wds))
            blocks.append(Block(bbox=[x0, y0, x1, y1], lines=b_lines))

        pages.append(PageIndex(
            number=i+1,
            width=pix.width,
            height=pix.height,
            image_b64=image_b64,
            blocks=blocks
        ))

    meta = {"num_pages": len(pages)}
    return {"pages": [asdict(p) for p in pages], "meta": meta}
