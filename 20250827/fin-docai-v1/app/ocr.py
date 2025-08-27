
import pdfplumber
from typing import List, Tuple
from .schemas import OCRWord, BoundingBox

def extract_pdf_words(path: str) -> Tuple[List[OCRWord], Tuple[int, int]]:
    words: List[OCRWord] = []
    page_size = (612, 792)
    with pdfplumber.open(path) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0: page_size = (float(page.width), float(page.height))
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=True):
                words.append(OCRWord(text=w.get("text",""), bbox=BoundingBox(
                    page=pi, x0=float(w.get("x0",0)), y0=float(w.get("top",0)),
                    x1=float(w.get("x1",0)), y1=float(w.get("bottom",0)))))
    return words, page_size

def is_probably_digital_pdf(path: str) -> bool:
    try:
        with pdfplumber.open(path) as pdf:
            return len(pdf.pages)>0 and len(pdf.pages[0].extract_words())>0
    except Exception:
        return False

def extract_words(path: str) -> Tuple[List[OCRWord], Tuple[int,int]]:
    if is_probably_digital_pdf(path): return extract_pdf_words(path)
    raise RuntimeError("OCR fallback not implemented in v1. Provide a digital PDF or add Tesseract fallback.")
