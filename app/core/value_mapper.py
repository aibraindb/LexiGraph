from __future__ import annotations
from typing import Dict, List
import re

MONEY = re.compile(r"(?<!\d)(?:USD\s*)?\$?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?")
DATE  = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
PCT   = re.compile(r"\b\d{1,2}(?:\.\d+)?%\b")

def _best(txt: str):
    for rx in (MONEY, DATE, PCT):
        m = rx.search(txt)
        if m:
            return m.group(0)
    return None

def map_values(full_text: str, attributes: List[Dict]) -> Dict[str, Dict]:
    txt = full_text or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    out={}
    for a in attributes:
        prop = a["property"]
        labels = [l.lower() for l in a.get("labels", []) if l]
        found=None
        for i,ln in enumerate(lines):
            lnl=ln.lower()
            if any(l in lnl for l in labels if len(l)>2):
                v = _best(ln) or (i+1<len(lines) and _best(lines[i+1]))
                if v:
                    found={"value":v,"confidence":0.82,"evidence":{"line":i}}
                    break
        if not found:
            v=_best(txt)
            if v:
                found={"value":v,"confidence":0.55,"evidence":{"line":None}}
        out[prop]=found or {"value":None,"confidence":0.0,"evidence":{}}
    return out
