# app/core/ocr_indexer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
import io, math

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "PyMuPDF (fitz) is required for the indexer. pip install pymupdf==1.23.8"
    ) from e

# --- add these helpers ---

def page_to_dict(pg):
    """
    Accepts either a dict-like page or a Page dataclass/object.
    Returns a dict with keys: blocks, lines, words, image (if present).
    """
    if isinstance(pg, dict):
        return pg

    d = {}
    # tolerate both attribute and key access styles
    for k in ("blocks", "lines", "words", "image"):
        if hasattr(pg, k):
            d[k] = getattr(pg, k)
    return d

def _lines_from_blocks_or_words(pg_dict):
    # Prefer explicit lines if present
    if "lines" in pg_dict and isinstance(pg_dict["lines"], list) and pg_dict["lines"]:
        return pg_dict["lines"]

    # Try to derive lines from blocks->lines
    out = []
    for b in pg_dict.get("blocks", []):
        blines = b.get("lines") if isinstance(b, dict) else None
        if isinstance(blines, list):
            out.extend(blines)

    if out:
        return out

    # Last resort: group words into pseudo-lines by y-center buckets
    words = pg_dict.get("words", [])
    if not words:
        return []

    # bucket by y center with small tolerance
    buckets = []
    tol = 6  # px tolerance; tweak if needed
    for w in words:
        try:
            x0, y0, x1, y1 = w["bbox"]
            y = 0.5 * (y0 + y1)
            placed = False
            for bucket in buckets:
                if abs(bucket["y"] - y) <= tol:
                    bucket["words"].append(w)
                    bucket["y"] = (bucket["y"] * bucket["n"] + y) / (bucket["n"] + 1)
                    bucket["n"] += 1
                    placed = True
                    break
            if not placed:
                buckets.append({"y": y, "n": 1, "words": [w]})
        except Exception:
            continue

    # sort buckets top->bottom; within each, sort words left->right and merge text
    buckets.sort(key=lambda b: b["y"])
    lines = []
    for i, b in enumerate(buckets):
        b["words"].sort(key=lambda w: w["bbox"][0])
        text = " ".join([w.get("text", "").strip() for w in b["words"] if w.get("text")])
        # union bbox
        xs0 = [w["bbox"][0] for w in b["words"]]
        ys0 = [w["bbox"][1] for w in b["words"]]
        xs1 = [w["bbox"][2] for w in b["words"]]
        ys1 = [w["bbox"][3] for w in b["words"]]
        if not xs0:
            continue
        bbox = [min(xs0), min(ys0), max(xs1), max(ys1)]
        lines.append({"id": f"ln_{i+1:04d}", "text": text, "bbox": bbox})

    return lines

# --- replace your existing page_lines with this ---

def page_lines(pg):
    """
    Return a list of line dicts, robust to Page object or plain dict.
    Each line dict: {"id": str, "text": str, "bbox": [x0,y0,x1,y1]}
    """
    pg_dict = page_to_dict(pg)
    return _lines_from_blocks_or_words(pg_dict)

    # --- Normalizers -------------------------------------------------------------
def _to_bbox(b):
    # accept tuples, lists, or objects with x0,y0,x1,y1
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    if hasattr(b, "x0"):
        return [float(b.x0), float(b.y0), float(b.x1), float(b.y1)]
    return None

def _line_to_dict(ln, bi=None, li=None):
    if isinstance(ln, dict):
        d = dict(ln)
        d["bbox"] = _to_bbox(d.get("bbox"))
        d["id"] = d.get("id") or f"{bi}:{li}"
        d["text"] = d.get("text") or ""
        d["words"] = [dict(w, bbox=_to_bbox(w.get("bbox"))) for w in d.get("words", [])]
        return d
    # object style
    return {
        "id": f"{bi}:{li}",
        "text": getattr(ln, "text", "") or "",
        "bbox": _to_bbox(getattr(ln, "bbox", None)),
        "words": [
            {"text": getattr(w, "text", ""), "bbox": _to_bbox(getattr(w, "bbox", None))}
            for w in (getattr(ln, "words", []) or [])
        ],
    }

def _block_to_dict(b, bi=None):
    if isinstance(b, dict):
        return {
            "bbox": _to_bbox(b.get("bbox")),
            "lines": [_line_to_dict(ln, bi=bi, li=i) for i, ln in enumerate(b.get("lines", []))],
        }
    return {
        "bbox": _to_bbox(getattr(b, "bbox", None)),
        "lines": [_line_to_dict(ln, bi=bi, li=i) for i, ln in enumerate(getattr(b, "lines", []) or [])],
    }

def _page_to_dict(pg):
    if isinstance(pg, dict):
        # ensure normalized
        return {
            "num": pg.get("num") or pg.get("number"),
            "size": pg.get("size") or [pg.get("width"), pg.get("height")],
            "image_png": pg.get("image_png"),
            "blocks": [_block_to_dict(b, bi=i) for i, b in enumerate(pg.get("blocks", []))],
        }
    # object style
    size = getattr(pg, "size", None)
    if not size:
        size = [getattr(pg, "width", None), getattr(pg, "height", None)]
    return {
        "num": getattr(pg, "num", getattr(pg, "number", None)),
        "size": size,
        "image_png": getattr(pg, "image_png", None),
        "blocks": [_block_to_dict(b, bi=i) for i, b in enumerate(getattr(pg, "blocks", []) or [])],
    }


# If you also want words at page level:
def page_words(pg_dict):
    for ln in page_lines(pg_dict):
        for w in ln.get("words", []):
            yield {"text": w.get("text", ""), "bbox": w.get("bbox")}

def _safe_index_pdf(pdf_bytes, **ui_kwargs):
    from app.core.ocr_indexer import index_pdf_bytes

    # Read all possible names coming from the UI
    dpi = int(
        ui_kwargs.get("dpi")
        or ui_kwargs.get("render_dpi")
        or 144
    )
    min_line_height = int(
        ui_kwargs.get("min_line_height")
        or ui_kwargs.get("min_line_h")
        or 10
    )
    min_line_width = int(
        ui_kwargs.get("min_line_width")
        or ui_kwargs.get("min_line_w")
        or 40
    )
    min_chars = int(
        ui_kwargs.get("min_chars_in_line")
        or ui_kwargs.get("min_line_chars")
        or ui_kwargs.get("min_chars")
        or 6
    )

    # Try several parameter spellings expected by different indexer versions
    tries = [
        dict(dpi=dpi, min_line_height=min_line_height, min_line_width=min_line_width, min_chars_in_line=min_chars),
        dict(dpi=dpi, min_line_height=min_line_height, min_line_width=min_line_width, min_chars=min_chars),
        dict(dpi=dpi, min_line_h=min_line_height,  min_line_w=min_line_width,  min_chars_in_line=min_chars),
        dict(dpi=dpi, min_line_h=min_line_height,  min_line_w=min_line_width,  min_chars=min_chars),
        dict(dpi=dpi, min_line_height=min_line_height, min_line_width=min_line_width),
        dict(dpi=dpi, min_line_h=min_line_height,  min_line_w=min_line_width),
        dict(dpi=dpi),  # last-resort: let indexer use its own defaults
    ]

    last_err = None
    for kwargs in tries:
        try:
            return index_pdf_bytes(pdf_bytes, **kwargs)
        except TypeError as e:
            last_err = e
            continue
    # If all attempts failed, surface the most recent TypeError
    raise last_err

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1


def _area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _iou(a: Rect, b: Rect) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    u = _area(a) + _area(b) - inter
    return inter / u if u > 0 else 0.0


def _merge_rect(a: Rect, b: Rect) -> Rect:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _nms_merge(rects: List[Rect], iou_thresh: float = 0.35) -> List[Rect]:
    """Greedy merge of overlapping/near-duplicate rectangles."""
    rects = rects[:]
    rects.sort(key=_area, reverse=True)
    out: List[Rect] = []
    while rects:
        r = rects.pop(0)
        merged = r
        keep: List[Rect] = []
        for q in rects:
            if _iou(merged, q) >= iou_thresh:
                merged = _merge_rect(merged, q)
            else:
                keep.append(q)
        out.append(merged)
        rects = keep
    return out


@dataclass
class Line:
    id: str
    text: str
    bbox: Rect  # PDF coordinate space
    conf: float = 1.0


@dataclass
class Block:
    id: str
    bbox: Rect
    lines: List[Line]


@dataclass
class Page:
    number: int  # 1-based
    width: int
    height: int
    image_bytes: bytes  # PNG
    blocks: List[Block]


@dataclass
class OCRIndex:
    pages: List[Page]

    def to_dict(self) -> Dict[str, Any]:
        return {"pages": [asdict(p) for p in self.pages]}


#def _page_png_bytes(pg: "fitz.Page", dpi: int = 144) -> bytes:
#    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
#    pm = pg.get_pixmap(matrix=mat, alpha=False)  # RGB
#    return pm.tobytes("png")

def _page_png_bytes(pg: "fitz.Page", dpi: int = 144) -> bytes:
    """Render a page to PNG. If rendering fails, return a white PNG of page size."""
    try:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pm = pg.get_pixmap(matrix=mat, alpha=False)  # RGB
        return pm.tobytes("png")
    except Exception:
        # Fallback: make a blank white image so the UI never gets None
        import PIL.Image as Image, io
        w, h = int(pg.rect.width * dpi / 72.0), int(pg.rect.height * dpi / 72.0)
        w = max(4, w); h = max(4, h)
        img = Image.new("RGB", (w, h), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _normalize_rect(r) -> Rect:
    # fitz returns rect-like objects; normalize to tuple
    return (float(r[0]), float(r[1]), float(r[2]), float(r[3]))


def _collect_from_rawdict(
    raw: Dict[str, Any],
    min_h: float,
    min_w: float,
    min_chars: int,
    iou_merge: float,
    page_no: int,
) -> List[Block]:
    blocks: List[Block] = []
    bcount = 0
    lcount = 0
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            # type 0 = text block
            continue
        brect = _normalize_rect(b.get("bbox", (0, 0, 0, 0)))
        line_rects: List[Rect] = []
        line_texts: List[str] = []

        for l in b.get("lines", []):
            # concat spans to make a full line
            txt = "".join(sp.get("text", "") for sp in l.get("spans", []))
            if len(txt.strip()) < min_chars:
                continue
            lrect = _normalize_rect(l.get("bbox", (0, 0, 0, 0)))
            w = lrect[2] - lrect[0]
            h = lrect[3] - lrect[1]
            if w < min_w or h < min_h:
                continue
            line_rects.append(lrect)
            line_texts.append(txt)

        # merge near-dupe line rects
        if line_rects:
            merged = _nms_merge(line_rects, iou_merge)
            # rebuild lines using nearest original text (center matching)
            lines: List[Line] = []
            for mr in merged:
                # find original with max IoU to carry text
                best = -1.0
                best_i = -1
                for i, rr in enumerate(line_rects):
                    iou = _iou(mr, rr)
                    if iou > best:
                        best = iou
                        best_i = i
                lcount += 1
                lines.append(Line(id=f"p{page_no}_l{lcount}", text=line_texts[best_i], bbox=mr, conf=1.0))

            bcount += 1
            blocks.append(Block(id=f"p{page_no}_b{bcount}", bbox=brect, lines=lines))

    return blocks


def _collect_from_words(
    words: List[Tuple[float, float, float, float, str, int, int, int]],
    min_h: float,
    min_w: float,
    min_chars: int,
    iou_merge: float,
    page_no: int,
) -> List[Block]:
    """
    Fallback: group words into rough lines by shared line_no, then merge.
    words record: (x0, y0, x1, y1, text, block_no, line_no, word_no)
    """
    by_line: Dict[Tuple[int, int], List[Tuple[Rect, str]]] = {}
    for (x0, y0, x1, y1, s, bno, lno, _wno) in words:
        r = (float(x0), float(y0), float(x1), float(y1))
        if (x1 - x0) < min_w or (y1 - y0) < min_h:
            continue
        by_line.setdefault((bno, lno), []).append((r, s))

    blocks: Dict[int, List[Line]] = {}
    lcount = 0
    for (bno, lno), items in by_line.items():
        items.sort(key=lambda it: it[0][0])
        text = " ".join(s for (_r, s) in items)
        if len(text.strip()) < min_chars:
            continue
        rects = [r for (r, _s) in items]
        mrects = _nms_merge(rects, iou_merge)
        # choose widest merged rect to represent the line
        r = max(mrects, key=_area)
        lcount += 1
        blocks.setdefault(bno, []).append(Line(id=f"p{page_no}_l{lcount}", text=text, bbox=r, conf=1.0))

    out: List[Block] = []
    bcount = 0
    for _bno, lines in sorted(blocks.items()):
        # compute block bbox
        if not lines:
            continue
        x0 = min(l.bbox[0] for l in lines)
        y0 = min(l.bbox[1] for l in lines)
        x1 = max(l.bbox[2] for l in lines)
        y1 = max(l.bbox[3] for l in lines)
        bcount += 1
        out.append(Block(id=f"p{page_no}_b{bcount}", bbox=(x0, y0, x1, y1), lines=lines))
    return out


def index_pdf_bytes(
    pdf_bytes: bytes,
    *,
    dpi: int = 144,
    min_line_height: float = 9.0,
    min_line_width: float = 28.0,
    min_line_chars: int = 6,
    iou_merge: float = 0.35,
) -> OCRIndex:
    """
    Build a light-weight 'OCR' index using PDF text extraction (no external OCR).
    Reduces noisy boxes by:
      - using line/block granularity
      - filtering by size & length
      - merging overlapping rectangles (NMS-style)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[Page] = []

    for i, pg in enumerate(doc):
        pgno = i + 1
        w, h = int(pg.rect.width), int(pg.rect.height)
        png = _page_png_bytes(pg, dpi=dpi)

        raw = pg.get_text("rawdict")
        blocks = _collect_from_rawdict(
            raw,
            min_h=min_line_height,
            min_w=min_line_width,
            min_chars=min_line_chars,
            iou_merge=iou_merge,
            page_no=pgno,
        )

        # If empty (e.g., scanned), fallback to words (may still be empty if no OCR layer)
        if not blocks:
            words = pg.get_text("words")
            blocks = _collect_from_words(
                words,
                min_h=min_line_height,
                min_w=min_line_width,
                min_chars=min_line_chars,
                iou_merge=iou_merge,
                page_no=pgno,
            )

        pages.append(Page(number=pgno, width=w, height=h, image_bytes=png, blocks=blocks))

    return OCRIndex(pages=pages)
