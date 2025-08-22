from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
INDEX_JSON = DATA/"fibo_index.json"
VEC_JSON   = DATA/"fibo_vec.json"

def _load_index() -> Dict[str, Any]:
    if not INDEX_JSON.exists():
        raise FileNotFoundError("No FIBO index. Run build_fibo_index first or use sidebar button.")
    return json.loads(INDEX_JSON.read_text())

def build_fibo_vec(force: bool=False) -> Dict[str, Any]:
    if VEC_JSON.exists() and not force:
        try: return json.loads(VEC_JSON.read_text())
        except Exception: pass
    idx=_load_index()
    classes=idx.get("classes",[])
    texts=[]
    uris=[]
    for c in classes:
        t = (c.get("label","") + " " + c.get("search_text","") + " " + c.get("uri","")).strip()
        texts.append(t)
        uris.append(c["uri"])
    if not texts:
        out={"n_classes":0,"n_terms":0}
        VEC_JSON.write_text(json.dumps(out)); return out
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)
    X = vec.fit_transform(texts)
    out={"uris":uris, "vocabulary":vec.vocabulary_, "idf":vec.idf_.tolist()}
    VEC_JSON.write_text(json.dumps(out))
    return {"n_classes":len(uris), "n_terms":len(vec.vocabulary_)}

def _load_vec() -> tuple[TfidfVectorizer, np.ndarray, List[str]]:
    info=json.loads(VEC_JSON.read_text())
    vocab=info.get("vocabulary",{})
    idf=np.array(info.get("idf",[]), dtype=np.float64)
    uris=info.get("uris",[])
    vec=TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=1.0)
    # inject vocab & idf
    vec.vocabulary_=vocab
    vec.fixed_vocabulary_=True
    vec._idf_diag = None
    def _set_idf(v):
        from scipy import sparse
        vec.idf_ = idf
        vec._idf_diag = sparse.spdiags(idf, diags=0, m=len(idf), n=len(idf))
        return v
    # monkey patch a flag we toggle after first transform
    vec._lexi_idf_ready = False
    vec._lexi_set_idf = _set_idf
    return vec, idf, uris

def search_fibo(query_text: str, top_k: int=5) -> List[Dict[str,Any]]:
    idx=_load_index()
    classes=idx.get("classes",[])
    if not VEC_JSON.exists():
        build_fibo_vec(force=True)
    vec, idf, uris = _load_vec()
    docs_texts=[]
    for c in classes:
        t = (c.get("label","") + " " + c.get("search_text","") + " " + c.get("uri","")).strip()
        docs_texts.append(t)
    X_docs = vec.fit_transform(docs_texts)  # safe even with fixed vocabulary
    if not getattr(vec, "_lexi_idf_ready", False):
        vec._lexi_set_idf(True)
        vec._lexi_idf_ready=True
    q = vec.transform([query_text or ""])
    sims = cosine_similarity(q, X_docs).ravel()
    order = np.argsort(-sims)[:max(1, top_k)]
    out=[]
    for i in order:
        out.append({"uri": classes[i]["uri"], "label": classes[i]["label"], "score": float(sims[i])})
    return out
