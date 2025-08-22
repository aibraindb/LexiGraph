from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
DOC_DIR   = DATA/"doc_index"
DOC_JSON  = DOC_DIR/"docs.json"
VEC_NPY   = DOC_DIR/"X.npy"

_state = {"ids": [], "texts": [], "X": None, "vec": None}
DOC_DIR.mkdir(parents=True, exist_ok=True)

def _save():
    DOC_JSON.write_text(json.dumps({"ids": _state["ids"]}))
    if _state["X"] is not None:
        np.save(VEC_NPY, _state["X"].toarray())

def _load():
    if DOC_JSON.exists():
        try:
            meta=json.loads(DOC_JSON.read_text())
            _state["ids"]=meta.get("ids",[])
        except Exception:
            _state["ids"]=[]
    if VEC_NPY.exists():
        arr=np.load(VEC_NPY, allow_pickle=False)
        from scipy import sparse
        _state["X"]=sparse.csr_matrix(arr)
    if _state["vec"] is None:
        _state["vec"]=TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)

def add_document(doc_id: str, text: str):
    _load()
    _state["ids"].append(doc_id)
    _state["texts"].append(text or "")
    X = _state["vec"].fit_transform(_state["texts"])  # refit corpus each time (small MVP)
    _state["X"]=X
    _save()

def nearest_to(doc_id: str, top_k: int=5) -> List[Dict[str,Any]]:
    _load()
    if not _state["ids"] or _state["X"] is None: return []
    if doc_id not in _state["ids"]: return []
    idx=_state["ids"].index(doc_id)
    q=_state["X"][idx]
    sims=cosine_similarity(q, _state["X"]).ravel()
    order=np.argsort(-sims)[:top_k+1]
    out=[]
    for i in order:
        if i==idx: continue
        out.append({"doc_id": _state["ids"][i], "score": float(sims[i])})
    return out

def nearest_text(query: str, top_k: int=5) -> List[Dict[str,Any]]:
    _load()
    if not _state["ids"] or _state["X"] is None: return []
    q = _state["vec"].transform([query or ""])
    sims=cosine_similarity(q, _state["X"]).ravel()
    order=np.argsort(-sims)[:top_k]
    return [{"doc_id": _state["ids"][i], "score": float(sims[i])} for i in order]
