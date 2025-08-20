from __future__ import annotations
import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple

def extract_text_blocks(pdf_bytes: bytes) -> Dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    colon_lines = []
    headers = []
    kv_pairs = []
    for pno in range(len(doc)):
        p = doc[pno]
        blocks = p.get_text("blocks", flags=fitz.TEXTFLAGS_TEXT)
        blocks.sort(key=lambda b: (b[1], b[0]))
        page_text = []
        for (x0,y0,x1,y1,text,_bno,_type) in blocks:
            if not text: 
                continue
            for ln in text.splitlines():
                t = ln.strip()
                if not t: 
                    continue
                page_text.append(t)
                if ":" in t:
                    k,v = t.split(":",1)
                    k,k2 = k.strip(), re.sub(r"\s+"," ",k.strip())
                    v = v.strip()
                    if k2 and v:
                        kv_pairs.append({"page": pno+1, "bbox": [x0,y0,x1,y1], "key": k2, "val": v})
                        colon_lines.append(t)
                if re.match(r"^[A-Z0-9][A-Z0-9 \-]{6,}$", t):
                    headers.append(t)
        pages.append({"page": pno+1, "text": "\n".join(page_text)})
    full_text = "\n".join(p["text"] for p in pages)
    return {
        "pages": pages, 
        "full_text": full_text, 
        "colon_lines": colon_lines[:1000], 
        "headers": headers[:120],
        "kv_pairs": kv_pairs
    }

ANCHOR_BOOST = [
    "case","account","invoice","lease","loan","levy","garnish","deadline","due",
    "amount","total","balance","remit","employer","lessee","lessor","borrower",
    "guarantor","funding","acceptance","delivery","w-2","401k","statement","guarantee"
]

def focused_summary(extract: Dict, max_chars: int = 2500) -> str:
    parts = []
    if extract["headers"]:
        parts.extend(extract["headers"][:12])
    strong = [ln for ln in extract["colon_lines"] if any(k in ln.lower() for k in ANCHOR_BOOST)]
    parts.extend(strong[:120])
    if extract["pages"]:
        parts.append(extract["pages"][0]["text"][:1500])
    s = "\n".join(parts)
    return s[:max_chars]
