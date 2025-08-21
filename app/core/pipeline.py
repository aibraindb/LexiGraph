from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json, re

from app.core.pdf_text import extract_text_blocks
from app.core.fibo_vec import ensure_fibo_index, fibo_search
from app.core.fibo_attrs import attributes_for_class
from app.core.value_mapper import map_values

DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def _summarize(full_text: str, max_chars: int = 4000) -> str:
    if not full_text: return ""
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    head = " ".join(lines[:10])
    body = " ".join(lines[10:])
    return (head + "\n" + body)[:max_chars]

def _merge_attrs(cands: List[Dict], limit: int = 200) -> List[Dict]:
    seen=set(); out=[]
    for c in cands:
        attrs = attributes_for_class(c["uri"]).get("attributes", [])
        for a in attrs[:60]:
            if a["property"] in seen: continue
            seen.add(a["property"])
            out.append({"property": a["property"], "labels": a.get("labels", []),
                        "source": {"class": c["uri"], "label": c.get("label")}})
            if len(out)>=limit: break
    return out

def propose_schema(file_bytes: bytes, doc_name: str, topk_classes: int=5, score_floor: float=0.25) -> Dict:
    ex = extract_text_blocks(file_bytes)
    full_text = ex.get("full_text","")
    ensure_fibo_index()
    q = _summarize(full_text)
    hits = fibo_search(q, topk=topk_classes)
    cands = [h for h in hits if h.get("score",0.0)>=score_floor] or hits[:1]
    attrs = _merge_attrs(cands)
    schema = {"documentName": doc_name,
              "fiboCandidates": cands,
              "attributes": [{"property": a["property"], "labels": a.get("labels", []), "source": a.get("source", {})} for a in attrs]}
    doc_id = re.sub(r"[^A-Za-z0-9_.-]+","_",doc_name) or "doc"
    run = RUNS_DIR / doc_id
    run.mkdir(parents=True, exist_ok=True)
    (run/"schema.json").write_text(json.dumps(schema, indent=2))
    (run/"spans.json").write_text(json.dumps(ex, indent=2))
    (run/"text.txt").write_text(full_text)
    return {"doc_id": doc_id, "schema": schema, "candidates": cands, "full_text": full_text[:6000]}

def apply_schema(doc_id: str, file_bytes: bytes, schema: Dict) -> Dict:
    ex = extract_text_blocks(file_bytes)
    full_text = ex.get("full_text","")
    attrs = [{"property": a["property"], "labels": a.get("labels", [])} for a in schema.get("attributes", [])]
    mapped = map_values(full_text, attrs)
    found = sum(1 for v in mapped.values() if v and v.get("value"))
    total = len(attrs)
    run = RUNS_DIR / doc_id
    run.mkdir(parents=True, exist_ok=True)
    (run/"result.json").write_text(json.dumps(mapped, indent=2))
    (run/"spans.json").write_text(json.dumps(ex, indent=2))
    return {"doc_id": doc_id, "result": mapped, "coverage": {"found": int(found), "total": int(total)}, "pages": ex.get("pages", [])}
