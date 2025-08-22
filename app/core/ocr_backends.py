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
