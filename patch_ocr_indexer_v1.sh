cat > app/core/ocr_indexer.py <<'EOF'
import fitz  # PyMuPDF
import io
from PIL import Image

class OCRIndex:
    def __init__(self, pages):
        self.pages = pages

def index_pdf_bytes(pdf_bytes):
    doc = fitz.open("pdf", pdf_bytes)
    pages = []

    for page in doc:
        # Render page image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # Extract blocks/lines/words
        blocks = page.get_text("blocks")
        lines = page.get_text("lines")
        words = page.get_text("words")

        pages.append({
            "number": page.number,
            "image": buf.getvalue(),    # ✅ image bytes now included
            "blocks": blocks,
            "lines": lines,
            "words": words,
            "size": page.rect,
        })

    return OCRIndex(pages)
EOF
