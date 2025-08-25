from __future__ import annotations
import io, json, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# Prefer pdfplumber (pure Python on top of pdfminer) to avoid native lib drama.
# Fallback to PyMuPDF only if present.
_PLUMBER_OK = False
try:
    import pdfplumber
    _PLUMBER_OK = True
except Exception:
    _PLUMBER_OK = False

_FITZ_OK = False
try:
    import fitz  # PyMuPDF
    _FITZ_OK = True
except Exception:
    _FITZ_OK = False

from PIL import Image, ImageDraw


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Line:
    id: str
    text: str
    bbox: Tuple[float, float, float, float]  # (x0,y0,x1,y1) in PDF points
    conf: float = 1.0
    source: str = "pdf"
    page_index: int = 0

@dataclass
class Page:
    number: int           # 1-based page number
    width: float
    height: float
    image_bytes: Optional[bytes]  # background PNG to show in UI
    lines: List[Line]

@dataclass
class OCRIndex:
    pages: List[Page]


# -----------------------------
# Helpers
# -----------------------------
def _to_png_placeholder(width_pt: float, height_pt: float, dpi: int = 144) -> bytes:
    # A white page placeholder if we cannot rasterize PDF page
    w = int(width_pt * dpi / 72)
    h = int(height_pt * dpi / 72)
    if w <= 0 or h <= 0:
        w, h = 800, 1000
    img = Image.new("RGB", (w, h), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def _plumber_render_png(ppage, dpi: int = 144) -> Optional[bytes]:
    try:
        # pdfplumber’s to_image may require ImageMagick/Wand in some envs.
        # If it fails, we return None and let placeholder kick in.
        im = ppage.to_image(resolution=dpi)
        bio = io.BytesIO()
        im.original.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return None

def _fitz_render_png(fpage, dpi: int = 144) -> Optional[bytes]:
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = fpage.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None

def _normalize_lines(lines: List[Line]) -> List[Dict[str, Any]]:
    out = []
    for ln in lines:
        out.append({
            "id": ln.id,
            "text": ln.text,
            "bbox": ln.bbox,
            "conf": ln.conf,
            "source": ln.source,
            "page_index": ln.page_index
        })
    return out

def _page_to_dict(pg: Page) -> Dict[str, Any]:
    return {
        "number": pg.number,
        "width": pg.width,
        "height": pg.height,
        "image_bytes_b64": None,  # UI will not inline by default
        "lines": _normalize_lines(pg.lines)
    }

# -----------------------------
# Indexers
# -----------------------------
def _index_with_pdfplumber(pdf_bytes: bytes,
                           min_chars_in_line: int = 2,
                           line_merge_tol: int = 3) -> OCRIndex:
    pages: List[Page] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, p in enumerate(pdf.pages):
            width, height = float(p.width), float(p.height)
            # Words: list of dicts with x0, y0, x1, y1, text, top, bottom, baseline, etc.
            try:
                words = p.extract_words(keep_blank_chars=False, use_text_flow=True)
            except Exception:
                # fallback: single block
                words = []

            # group words into lines by 'top' with a tolerance
            lines: List[Line] = []
            if words:
                # sort by top then x0
                words = sorted(words, key=lambda w: (round(w.get("top", 0), 1), w.get("x0", 0)))
                # simple clustering by 'top' ~ line
                current: List[Dict[str, Any]] = []
                last_top = None
                for w in words:
                    t = float(w.get("top", 0))
                    if last_top is None or abs(t - last_top) <= line_merge_tol:
                        current.append(w)
                        last_top = t if last_top is None else (last_top + t) / 2.0
                    else:
                        if current:
                            lines.append(_words_to_line(current, pi))
                        current = [w]
                        last_top = t
                if current:
                    lines.append(_words_to_line(current, pi))

            # Filter tiny lines
            lines = [ln for ln in lines if len(ln.text.strip()) >= min_chars_in_line]

            # Render background
            bg = _plumber_render_png(p) or _to_png_placeholder(width, height)

            pages.append(Page(
                number=pi+1, width=width, height=height,
                image_bytes=bg, lines=lines
            ))
    return OCRIndex(pages=pages)

def _words_to_line(cluster: List[Dict[str, Any]], page_index: int) -> Line:
    x0 = min(float(w["x0"]) for w in cluster)
    y0 = min(float(w["top"]) for w in cluster)
    x1 = max(float(w["x1"]) for w in cluster)
    y1 = max(float(w.get("bottom", w.get("y1", w.get("top", 0)+10))) for w in cluster)
    text = " ".join(w.get("text", "") for w in cluster if w.get("text"))
    lid = f"p{page_index}_l{abs(hash((x0,y0,x1,y1,text)))%1_000_000}"
    return Line(id=lid, text=text, bbox=(x0,y0,x1,y1), page_index=page_index)

def _index_with_fitz(pdf_bytes: bytes,
                     min_chars_in_line: int = 2) -> OCRIndex:
    # Use PyMuPDF text blocks
    pages: List[Page] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pi, fpage in enumerate(doc):
        width, height = float(fpage.rect.width), float(fpage.rect.height)
        # Extract text blocks
        lines: List[Line] = []
        try:
            blocks = fpage.get_text("blocks")  # x0,y0,x1,y1, text, block_no, ...
            for b in blocks:
                x0, y0, x1, y1, txt, *_ = b
                if not isinstance(txt, str):
                    continue
                for raw_line in txt.splitlines():
                    if len(raw_line.strip()) >= min_chars_in_line:
                        lid = f"p{pi}_l{abs(hash((x0,y0,x1,y1,raw_line)))%1_000_000}"
                        lines.append(Line(
                            id=lid, text=raw_line, bbox=(x0,y0,x1,y1),
                            page_index=pi
                        ))
        except Exception:
            pass

        bg = _fitz_render_png(fpage) or _to_png_placeholder(width, height)
        pages.append(Page(number=pi+1, width=width, height=height, image_bytes=bg, lines=lines))
    return OCRIndex(pages=pages)

# -----------------------------
# Public API
# -----------------------------
def index_pdf_bytes(pdf_bytes: bytes,
                    min_chars_in_line: int = 2,
                    line_merge_tol: int = 3) -> OCRIndex:
    """
    Return OCRIndex with pages -> lines (each with bbox).
    Uses pdfplumber by default; falls back to PyMuPDF if available.
    """
    if _PLUMBER_OK:
        return _index_with_pdfplumber(pdf_bytes,
                                      min_chars_in_line=min_chars_in_line,
                                      line_merge_tol=line_merge_tol)
    if _FITZ_OK:
        return _index_with_fitz(pdf_bytes, min_chars_in_line=min_chars_in_line)
    raise RuntimeError("No OCR backend available (pdfplumber / PyMuPDF missing). Install pdfplumber.")

def page_to_dict(pg: Page) -> Dict[str, Any]:
    return _page_to_dict(pg)

