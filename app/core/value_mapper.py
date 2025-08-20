from __future__ import annotations
from typing import Dict, List
import re
from collections import defaultdict

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()

def _sim(a: str, b: str) -> float:
    A = set(_norm(a).split()); B = set(_norm(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def kv_from_text(full_text: str) -> Dict[str, List[str]]:
    out = defaultdict(list)
    for ln in full_text.splitlines():
        if ":" in ln:
            k,v = ln.split(":",1)
            k=k.strip(); v=v.strip()
            if k and v:
                out[k].append(v)
    return out

def map_values_to_attributes(full_text: str, attributes: List[Dict], kv_pairs: List[Dict]|None=None, threshold: float=0.40) -> Dict:
    kv = kv_from_text(full_text)
    # also fold in structured kv_pairs (from blocks)
    if kv_pairs:
        for item in kv_pairs:
            k = item["key"]; v = item["val"]
            if k and v:
                kv[k].append(v)
    results = {}
    for attr in attributes:
        labels = attr.get("labels") or [attr["property"].split("/")[-1]]
        best = None; bestscore = 0.0; best_val = None
        for k,vals in kv.items():
            for lab in labels:
                s = _sim(k, lab)
                if s > bestscore:
                    bestscore = s
                    best = k
                    best_val = vals[0]
        if best and bestscore >= threshold:
            results[attr["property"]] = {"value": best_val, "match_key": best, "score": bestscore}
    return results
