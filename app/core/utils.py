from pdfminer.high_level import extract_text
from pathlib import Path

def read_file_as_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    elif name.endswith(".pdf"):
        tmp = Path("/tmp/_lexi_doc.pdf")
        tmp.write_bytes(data)
        try:
            return extract_text(str(tmp)) or ""
        except Exception:
            return ""
    else:
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""
