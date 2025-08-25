from __future__ import annotations
import io, math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import pdfplumber
from PIL import Image, ImageDraw

@dataclass
class Line:
    bbox: Tuple[float,float,float,float]   # (x0,y0,x1,y1) in PDF coords
    text: str
    words: List[int] = field(default_factory=list)

@dataclass
class Word:
    bbox: Tuple[float,float,float,float]
    text: str

@dataclass
class PageDict:
    number: int
    size: Tuple[float,float]               # (width,height) in PDF coords
    image_bytes: Optional[bytes]           # PNG bytes if rendered; else None
    lines: List[Line] = field(default_factory=list)
    words: List[Word] = field(default_factory=list)

def _render_page_image(pp: pdfplumber.page.Page, dpi: int = 144) -> Optional[bytes]:
    """
    Try to render a page preview using pdfplumber's rasterizer.
    This path does NOT require PyMuPDF. If rendering fails, return None.
    """
    try:
        # pdfplumber Page has .to_image(resolution=)
        pi = pp.to_image(resolution=dpi)  # returns pdfplumber.display.PageImage
        # PageImage has .original (PIL Image)
        img = pi.original.convert("RGB")
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        # Safe fallback: no background image, tree will still work
        return None

def _group_words_to_lines(words: List[Dict], y_tol: float = 3.0, min_chars: int = 2) -> List[Line]:
    """Group words into text lines by y proximity; returns Line list."""
    if not words:
        return []
    # Normalize: words from pdfplumber.extract_words() have x0,y0,x1,y1,text
    items = []
    for i, w in enumerate(words):
        try:
            x0=float(w["x0"]); y0=float(w["top"]); x1=float(w["x1"]); y1=float(w["bottom"])
            txt=str(w.get("text","") or "")
            items.append((i, x0,y0,x1,y1, txt))
        except Exception:
            continue
    if not items:
        return []
    # Sort by y, then x
    items.sort(key=lambda t: (t[2], t[1]))
    lines: List[Line] = []
    cur: List[Tuple[int,float,float,float,float,str]] = []
    def flush():
        if not cur:
            return
        xs = [c[1] for c in cur]; ys0 = [c[2] for c in cur]; xs1 = [c[3] for c in cur]; ys1 = [c[4] for c in cur]
        text = " ".join([c[5] for c in cur]).strip()
        if len(text) >= min_chars:
            lines.append(Line(
                bbox=(min(xs), min(ys0), max(xs1), max(ys1)),
                text=text
            ))
    # Greedy grouping
    baseline = items[0][2]
    for rec in items:
        _, x0,y0,x1,y1, txt = rec
        if abs(y0 - baseline) <= y_tol:
            cur.append(rec)
        else:
            flush()
            cur = [rec]
            baseline = y0
    flush()
    return lines

def _words_from_plumber(pp: pdfplumber.page.Page) -> List[Word]:
    res = []
    try:
        ws = pp.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=["top","bottom"]
        )
        for w in ws:
            x0=float(w["x0"]); x1=float(w["x1"])
            y0=float(w["top"]); y1=float(w["bottom"])
            txt=str(w.get("text","") or "")
            res.append(Word(bbox=(x0,y0,x1,y1), text=txt))
    except Exception:
        pass
    return res

def index_pdf_bytes(pdf_bytes: bytes, render_dpi: int = 144) -> List[PageDict]:
    """
    Build a structured OCR-like index for a PDF.
    DIGITAL-FIRST: uses pdfplumber to read embedded text and approximate lines.
    If page image rendering fails, we still return lines/words without image_bytes.
    """
    pages: List[PageDict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, pp in enumerate(pdf.pages):
            w,h = pp.width, pp.height
            img_bytes = _render_page_image(pp, dpi=render_dpi)
            # words & lines
            words = _words_from_plumber(pp)
            # for grouping, pdfplumber provides a separate extract_words;
            # we rebuild lines from that set to keep consistent coords
            raw_pw = pp.extract_words(use_text_flow=True, extra_attrs=["top","bottom"])
            lines = _group_words_to_lines(raw_pw, y_tol=3.0, min_chars=2)
            pages.append(PageDict(
                number=i,
                size=(w,h),
                image_bytes=img_bytes,
                lines=lines,
                words=words
            ))
    return pages

# convenience
def pages_as_dicts(pages: List[PageDict]) -> List[Dict]:
    out=[]
    for p in pages:
        out.append({
            "number": p.number,
            "size": p.size,
            "image_bytes": p.image_bytes,
            "lines": [ {"bbox":l.bbox,"text":l.text, "words":l.words} for l in p.lines ],
            "words": [ {"bbox":w.bbox,"text":w.text} for w in p.words ],
        })
    return out
