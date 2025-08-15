
from pdfminer.high_level import extract_text
def read_text_from_upload(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt"):
        try: return data.decode("utf-8")
        except Exception: return data.decode("latin-1", errors="ignore")
    if name.endswith(".pdf"):
        with open("/tmp/_lexi_pdf.pdf","wb") as f: f.write(data)
        try: return extract_text("/tmp/_lexi_pdf.pdf") or ""
        except Exception: return ""
    try: return data.decode("utf-8", errors="ignore")
    except Exception: return ""
