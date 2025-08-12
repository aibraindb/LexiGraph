from pdfminer.high_level import extract_text
from typing import Optional

def extract_text_from_pdf(path: str) -> Optional[str]:
    try:
        return extract_text(path)
    except Exception:
        return None
