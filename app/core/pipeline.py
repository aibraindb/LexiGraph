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
