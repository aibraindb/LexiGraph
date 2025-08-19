from __future__ import annotations
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

def _easyocr_reader(lang: str = "en"):
    import easyocr
    return easyocr.Reader([lang], gpu=False)  # CPU only

def _paddle_reader(lang: str = "en", det_dir: Optional[str] = None, rec_dir: Optional[str] = None, cls_dir: Optional[str] = None):
    from paddleocr import PaddleOCR
    kwargs = dict(use_angle_cls=True, use_gpu=False, lang=lang)
    if det_dir: kwargs["det_model_dir"] = det_dir
    if rec_dir: kwargs["rec_model_dir"] = rec_dir
    if cls_dir: kwargs["cls_model_dir"] = cls_dir
    return PaddleOCR(**kwargs)

class OCRBackend:
    def __init__(self, lang: str = "en", prefer: str = "easyocr",
                 paddle_det_dir: Optional[str] = None,
                 paddle_rec_dir: Optional[str] = None,
                 paddle_cls_dir: Optional[str] = None):
        self.name: Optional[str] = None
        self.backend = None
        self._init(lang, prefer, paddle_det_dir, paddle_rec_dir, paddle_cls_dir)

    def _init(self, lang: str, prefer: str, det, rec, cls) -> None:
        def try_easy() -> bool:
            try:
                rd = _easyocr_reader(lang)
                self.name = "easyocr"; self.backend = rd; return True
            except Exception: return False
        def try_paddle() -> bool:
            try:
                rd = _paddle_reader(lang, det, rec, cls)
                self.name = "paddleocr"; self.backend = rd; return True
            except Exception: return False
        if (prefer or "").lower() == "paddle":
            if not try_paddle():
                if not try_easy(): raise RuntimeError("No OCR backend available (paddle/easyocr failed).")
        else:
            if not try_easy():
                if not try_paddle(): raise RuntimeError("No OCR backend available (easyocr/paddle failed).")

    def run(self, img: Image.Image) -> List[Dict[str, Any]]:
        if self.name == "easyocr":
            import numpy as np
            arr = np.array(img.convert("RGB"))
            results = self.backend.readtext(arr, detail=1, paragraph=False)
            out: List[Dict[str, Any]] = []
            for (box, txt, conf) in results:
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                out.append({"polygon":[(float(x), float(y)) for x,y in box], "bbox": bbox, "text": str(txt), "conf": float(conf)})
            return out
        if self.name == "paddleocr":
            arr = np.array(img.convert("RGB"))
            result = self.backend.ocr(arr, cls=True)
            out: List[Dict[str, Any]] = []
            if not result: return out
            lines = result[0] or []
            for line in lines:
                if not line or len(line) < 2: continue
                box = line[0]; txt = line[1][0]; conf = float(line[1][1]) if line[1][1] is not None else 0.0
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                out.append({"polygon":[(float(x), float(y)) for x,y in box], "bbox": bbox, "text": str(txt), "conf": conf})
            return out
        return []
