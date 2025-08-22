from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

MONEY_RE = re.compile(r"\$?\s?[\d]{1,3}(?:[,][\d]{3})*(?:\.[\d]{2})?")
DATE_RE  = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\b")
RATE_RE  = re.compile(r"\b\d{1,2}(?:\.\d+)?\s?%\b")
TERM_RE  = re.compile(r"\b(?:\d+\s?(?:months?|years?|mos?))\b", re.I)
ID_RE    = re.compile(r"\b(?:Acct(?:ount)?\s*#?\s*[:\-]?\s*|No\.?\s*[:\-]?\s*)[A-Za-z0-9\-]{3,}\b", re.I)

def detect_candidates(text: str) -> Dict[str, List[str]]:
    return {
        "money": MONEY_RE.findall(text or ""),
        "date":  DATE_RE.findall(text or ""),
        "rate":  RATE_RE.findall(text or ""),
        "term":  TERM_RE.findall(text or ""),
        "id":    ID_RE.findall(text or ""),
    }

def spans_candidates(spans: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for ln in spans:
        t=ln.get("text","")
        for m in MONEY_RE.finditer(t):
            out.append({"kind":"money","text":m.group(0),"bbox":ln.get("bbox"),"source":t})
        for d in DATE_RE.finditer(t):
            out.append({"kind":"date","text":d.group(0),"bbox":ln.get("bbox"),"source":t})
        for r in RATE_RE.finditer(t):
            out.append({"kind":"rate","text":r.group(0),"bbox":ln.get("bbox"),"source":t})
        for tm in TERM_RE.finditer(t):
            out.append({"kind":"term","text":tm.group(0),"bbox":ln.get("bbox"),"source":t})
        for i in ID_RE.finditer(t):
            out.append({"kind":"id","text":i.group(0),"bbox":ln.get("bbox"),"source":t})
    return out

def make_attr_matcher(attributes: List[Dict[str,Any]]):
    # Build a TF-IDF matcher over attribute label synonyms.
    labels=[]; keys=[]
    for a in attributes:
        labs=a.get("labels") or []
        if not labs: labs=[a["property"].split("/")[-1]]
        labs=[l.lower() for l in labs]
        labels.append(" ".join(labs))
        keys.append(a["property"])
    if not labels:
        vec=TfidfVectorizer(min_df=1); vec.fit(["fallback"])
        return vec, np.zeros((1,1)), ["fallback"]
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X=vec.fit_transform(labels)
    return vec, X, keys

def map_candidates_to_attrs(cands: List[Dict[str,Any]], attributes: List[Dict[str,Any]], threshold: float=0.25) -> List[Dict[str,Any]]:
    vec, X, keys = make_attr_matcher(attributes)
    out=[]
    for c in cands:
        q = vec.transform([c["source"].lower()])
        # cosine sim to attribute label space
        sims = (q @ X.T).toarray().ravel()
        if sims.size==0: continue
        i = int(np.argmax(sims)); score=float(sims[i])
        if score>=threshold:
            out.append({"property": keys[i], "score": score, "kind": c["kind"],
                        "value": c["text"], "bbox": c.get("bbox")})
    # consolidate best value per attribute by score
    best={}
    for r in out:
        k=r["property"]
        if (k not in best) or (r["score"]>best[k]["score"]):
            best[k]=r
    return list(best.values())
