#!/usr/bin/env bash
set -euo pipefail

mkdir -p app/core
touch app/__init__.py app/core/__init__.py

############################################
# app/core/ocr_backends.py
############################################
cat > app/core/ocr_backends.py <<'PY'
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math

# We default to PyMuPDF text only (no network).
# If you install EasyOCR or PaddleOCR later, these will be auto-detected.

def available_backends() -> Dict[str, bool]:
    out = {"pymupdf_text": True, "easyocr": False, "paddle": False}
    try:
        import easyocr  # noqa
        out["easyocr"] = True
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR  # noqa
        out["paddle"] = True
    except Exception:
        pass
    return out

def ocr_easyocr(image_bytes: bytes, lang: str = "en") -> List[Dict[str, Any]]:
    # returns [{"text": str, "bbox": [x0,y0,x1,y1]}]
    import easyocr
    import numpy as np
    import cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    reader = easyocr.Reader([lang], gpu=False)
    res = reader.readtext(img, detail=1)
    out = []
    for bbox, text, conf in res:
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        out.append({"text": text, "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], "conf": float(conf)})
    return out

def ocr_paddle(image_bytes: bytes) -> List[Dict[str, Any]]:
    from paddleocr import PaddleOCR
    import numpy as np, cv2
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, show_log=False)
    result = ocr.ocr(img, cls=True)
    out=[]
    for line in result:
        for box, (txt, score) in line:
            xs=[p[0] for p in box]; ys=[p[1] for p in box]
            out.append({"text": txt, "bbox":[float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))], "conf": float(score)})
    return out

def merge_lines(lines: List[Dict[str,Any]], ocr_lines: List[Dict[str,Any]], iou_thresh: float=0.2) -> List[Dict[str,Any]]:
    # Attach OCR lines that don't heavily overlap with existing lines.
    def iou(a,b):
        ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
        inter_x0=max(ax0,bx0); inter_y0=max(ay0,by0)
        inter_x1=min(ax1,bx1); inter_y1=min(ay1,by1)
        if inter_x1<=inter_x0 or inter_y1<=inter_y0: return 0.0
        inter=(inter_x1-inter_x0)*(inter_y1-inter_y0)
        a_area=(ax1-ax0)*(ay1-ay0); b_area=(bx1-bx0)*(by1-by0)
        return inter/(a_area+b_area-inter+1e-9)
    existing = [l.get("bbox") for l in lines if l.get("bbox")]
    for o in ocr_lines:
        bb = o.get("bbox")
        if not bb: continue
        if max([iou(bb,e) for e in existing] or [0.0]) < iou_thresh:
            lines.append({"text": o.get("text",""), "bbox": bb, "spans":[{"text":o.get("text",""),"bbox":bb,"size":None}], "ocr": True})
    return lines
PY

############################################
# app/core/value_mapper.py
############################################
cat > app/core/value_mapper.py <<'PY'
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

MONEY_RE = re.compile(r"\$?\s?[\d]{1,3}(?:[,][\d]{3})*(?:\.[\d]{2})?")
DATE_RE  = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b")
RATE_RE  = re.compile(r"\b\d{1,2}(?:\.\d+)?\s?%\b")
TERM_RE  = re.compile(r"\b(?:\d+\s?(?:months?|years?|mos?))\b", re.I)
ID_RE    = re.compile(r"\b(?:Acct(?:ount)?\s*#?\s*[:\-]?\s*|No\.?\s*[:\-]?\s*)[A-Za-z0-9\-]{3,}\b", re.I)

def detect_candidates(text: str) -> Dict[str, List[str]]:
    return {
        "money": MONEY_RE.findall(text or ""),
        "date":  DATE_RE.findall(text or ""),
        "rate":  RATE_RE.findall(text or ""),
        "term":  TERM_RE.findall(text or ""),
        "id":    ID_RE.findall(text or ""),
    }

def spans_candidates(spans: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for ln in spans:
        t=ln.get("text","")
        for m in MONEY_RE.finditer(t):
            out.append({"kind":"money","text":m.group(0),"bbox":ln.get("bbox"),"source":t})
        for d in DATE_RE.finditer(t):
            out.append({"kind":"date","text":d.group(0),"bbox":ln.get("bbox"),"source":t})
        for r in RATE_RE.finditer(t):
            out.append({"kind":"rate","text":r.group(0),"bbox":ln.get("bbox"),"source":t})
        for tm in TERM_RE.finditer(t):
            out.append({"kind":"term","text":tm.group(0),"bbox":ln.get("bbox"),"source":t})
        for i in ID_RE.finditer(t):
            out.append({"kind":"id","text":i.group(0),"bbox":ln.get("bbox"),"source":t})
    return out

def make_attr_matcher(attributes: List[Dict[str,Any]]):
    # Build a TF-IDF matcher over attribute label synonyms.
    labels=[]; keys=[]
    for a in attributes:
        labs=a.get("labels") or []
        if not labs: labs=[a["property"].split("/")[-1]]
        labs=[l.lower() for l in labs]
        labels.append(" ".join(labs))
        keys.append(a["property"])
    if not labels:
        vec=TfidfVectorizer(min_df=1); vec.fit(["fallback"])
        return vec, np.zeros((1,1)), ["fallback"]
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X=vec.fit_transform(labels)
    return vec, X, keys

def map_candidates_to_attrs(cands: List[Dict[str,Any]], attributes: List[Dict[str,Any]], threshold: float=0.25) -> List[Dict[str,Any]]:
    vec, X, keys = make_attr_matcher(attributes)
    out=[]
    for c in cands:
        q = vec.transform([c["source"].lower()])
        # cosine sim to attribute label space
        sims = (q @ X.T).toarray().ravel()
        if sims.size==0: continue
        i = int(np.argmax(sims)); score=float(sims[i])
        if score>=threshold:
            out.append({"property": keys[i], "score": score, "kind": c["kind"],
                        "value": c["text"], "bbox": c.get("bbox")})
    # consolidate best value per attribute by score
    best={}
    for r in out:
        k=r["property"]
        if (k not in best) or (r["score"]>best[k]["score"]):
            best[k]=r
    return list(best.values())
PY

############################################
# app/core/pipeline.py
############################################
cat > app/core/pipeline.py <<'PY'
from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from app.core.pdf_text import extract_text_blocks, get_page_images
from app.core.fibo_vec import search_fibo
from app.core.fibo_attrs import attributes_for_class
from app.core.value_mapper import spans_candidates, map_candidates_to_attrs
from app.core.ocr_backends import available_backends, ocr_easyocr, ocr_paddle, merge_lines

@dataclass
class ExtractResult:
    doc_id: str
    fibo_class_uri: Optional[str]
    fibo_candidates: List[Dict[str,Any]]
    attributes: List[Dict[str,Any]]
    mapped: List[Dict[str,Any]]
    coverage: Dict[str, Any]
    pages: int
    warnings: List[str]

def _coverage(attributes: List[Dict[str,Any]], mapped: List[Dict[str,Any]]) -> Dict[str,Any]:
    want=len(attributes)
    have=len({m["property"] for m in mapped})
    return {"have": have, "want": want, "ratio": (have/ max(1,want))}

def process_document(doc_id: str, pdf_bytes: bytes, autolink_threshold: float=0.35, use_ocr: bool=False) -> ExtractResult:
    warnings=[]
    # 1) Parse PDF via PyMuPDF
    pages = extract_text_blocks(pdf_bytes)  # [{"page":0,"text":..., "spans":[...]}]
    # Optional: OCR fallback per page image
    if use_ocr:
        back = available_backends()
        if not (back.get("easyocr") or back.get("paddle")):
            warnings.append("OCR requested but no backend available; using digital text only.")
        else:
            imgs = get_page_images(pdf_bytes, zoom=2.0)
            for pno, raw in enumerate(imgs):
                try:
                    if back.get("easyocr"):
                        ocr_lines = ocr_easyocr(raw)
                    else:
                        ocr_lines = ocr_paddle(raw)
                    pages[pno]["spans"] = merge_lines(pages[pno]["spans"], ocr_lines, iou_thresh=0.2)
                    # also extend text
                    extra = "\n".join([l["text"] for l in ocr_lines])
                    pages[pno]["text"] = (pages[pno]["text"] + "\n" + extra).strip()
                except Exception as e:
                    warnings.append(f"OCR failed on page {pno+1}: {e}")

    # 2) Build a query text for FIBO fuzzy match (first 2k chars)
    full_text = "\n".join(p["text"] for p in pages if p.get("text"))
    seed = full_text[:2000]

    fibo_hits = search_fibo(seed, top_k=8)  # [{"uri","label","score"}]
    top_uri = None
    if fibo_hits and fibo_hits[0]["score"]>=autolink_threshold:
        top_uri = fibo_hits[0]["uri"]

    # 3) Attributes for chosen FIBO class (if any)
    attrs = attributes_for_class(top_uri) if top_uri else {"attributes": [], "count": 0}
    attr_list = attrs.get("attributes", [])

    # 4) Candidate values from spans + schema-aware mapping
    all_spans = []
    for p in pages:
        for ln in p.get("spans", []):
            all_spans.append(ln)
    cands = spans_candidates(all_spans)  # [{kind,value,bbox,source}]
    mapped = map_candidates_to_attrs(cands, attr_list, threshold=0.25)

    # 5) Coverage
    cov = _coverage(attr_list, mapped)

    return ExtractResult(
        doc_id=doc_id,
        fibo_class_uri=top_uri,
        fibo_candidates=fibo_hits,
        attributes=attr_list,
        mapped=mapped,
        coverage=cov,
        pages=len(pages),
        warnings=warnings,
    )

# Convenience: dict output for UI/JSON
def process_document_dict(doc_id: str, pdf_bytes: bytes, autolink_threshold: float=0.35, use_ocr: bool=False) -> Dict[str,Any]:
    return asdict(process_document(doc_id, pdf_bytes, autolink_threshold=autolink_threshold, use_ocr=use_ocr))
PY

echo "✅ Installed: ocr_backends.py, value_mapper.py, pipeline.py"
