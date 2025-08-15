#!/usr/bin/env bash
set -euo pipefail

# --- New extractor
cat > app/core/attr_extract.py <<'PY'
from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, List, Tuple
from .fibo_index import attributes_for_class

DOCS_DIR = Path("data/docs")

# Simple normalizers
MONEY_RE = re.compile(r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b")
DATE_RE  = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
ID_RE    = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b|\b\d{9,}\b")

def _norm_label(l: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", l.lower()).strip()

def _build_anchor_patterns(labels: List[str]) -> List[re.Pattern]:
    pats = []
    for l in labels:
        ln = _norm_label(l)
        if not ln: continue
        # exact token sequence or loose tokens
        toks = [re.escape(t) for t in ln.split()]
        if not toks: continue
        pats.append(re.compile(r"\\b" + r"\\s*".join(toks) + r"\\b", re.IGNORECASE))
        if len(toks) > 1:
            pats.append(re.compile(r"\\b" + r".{0,30}?".join(toks) + r"\\b", re.IGNORECASE))
    return pats

def _extract_near(text: str, pos: int, window: int = 200) -> Dict[str, List[str]]:
    start = max(0, pos - window//2)
    end   = min(len(text), pos + window//2)
    chunk = text[start:end]
    return {
        "amounts": MONEY_RE.findall(chunk)[:3],
        "dates":   DATE_RE.findall(chunk)[:3],
        "ids":     ID_RE.findall(chunk)[:3],
        "raw":     chunk[:500]
    }

def extract_by_schema(doc_id: str, class_uri: str) -> Dict:
    p = DOCS_DIR / f"{doc_id}.json"
    if not p.exists():
        return {"error":"doc_id_not_found"}
    meta = json.loads(p.read_text())
    text = meta.get("text","")
    attrs = attributes_for_class(class_uri).get("attributes", [])
    results = {}
    covered = 0
    for row in attrs:
        prop = row["property"]
        labels = row.get("labels", [])
        patterns = _build_anchor_patterns(labels)
        hits = []
        for pat in patterns:
            for m in pat.finditer(text):
                ctx = _extract_near(text, m.start())
                # heuristic: prefer single good value if obvious
                val = ctx["amounts"][:1] or ctx["dates"][:1] or ctx["ids"][:1] or []
                hits.append({
                    "match": m.group(0)[:120],
                    "value": val[0] if val else None,
                    "context": ctx["raw"]
                })
                if len(hits) >= 3:
                    break
            if len(hits) >= 3:
                break
        if hits:
            covered += 1
        # return the best guess + all candidates
        results[prop] = {"labels": labels, "best": (hits[0]["value"] if hits and hits[0]["value"] else None), "candidates": hits}

    coverage = (covered / max(1, len(attrs))) if attrs else 0.0
    return {"doc_id": doc_id, "class_uri": class_uri, "coverage": coverage, "fields": results}
PY

# --- Wire endpoint
python3 - <<'PY'
from pathlib import Path
p = Path("app/api/main.py")
s = p.read_text()
if "def extract_by_schema" not in s:
    s = s.replace("from app.core import fibo_index",
                  "from app.core import fibo_index\nfrom app.core.attr_extract import extract_by_schema as _extract_by_schema")
    s += """

# ---- Attribute-driven extraction ----
@app.post("/extract/by_schema")
def api_extract_by_schema(doc_id: str = Query(...), class_uri: str = Query(...)):
    return _extract_by_schema(doc_id, class_uri)
"""
Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: /extract/by_schema is available."
