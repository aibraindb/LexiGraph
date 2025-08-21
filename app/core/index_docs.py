from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np, re, json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/vector_index")
DATA.mkdir(parents=True, exist_ok=True)

STATE = DATA / "state.joblib"
DOCS_JSON = DATA / "docs.json"

def _strip_values(text: str) -> str:
    if not text: return ""
    # remove money, dates, numbers for structure/keys emphasis
    text = re.sub(r"(?<!\d)(?:USD\s*)?\$?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?", " ", text)
    text = re.sub(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", " ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " ", text)
    text = re.sub(r"[_/\\\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def _load_state():
    if STATE.exists():
        return joblib.load(STATE)
    return {"names": [], "texts": [], "vec": None, "X": None}

def add_document(name: str, raw_text: str):
    st = _load_state()
    norm = _strip_values(raw_text)
    st["names"].append(name)
    st["texts"].append(norm)
    # rebuild vectorizer on corpus
    vec = TfidfVectorizer(min_df=1, max_df=1.0, token_pattern=r"(?u)\b\w[\w\-/\.]+\b")
    X = vec.fit_transform(st["texts"])
    st["vec"] = vec
    st["X"] = X
    joblib.dump(st, STATE)
    Path(DOCS_JSON).write_text(json.dumps({"names": st["names"]}))
    return {"count": len(st["names"]), "features": X.shape[1]}

def list_documents() -> List[str]:
    st = _load_state()
    return st["names"]

def knn(name: str, k: int = 5) -> List[Tuple[str, float]]:
    st = _load_state()
    if name not in st["names"]: return []
    i = st["names"].index(name)
    vec, X = st["vec"], st["X"]
    sims = cosine_similarity(X, X[i]).ravel()
    order = sims.argsort()[::-1]
    out=[]
    for j in order:
        if j == i: continue
        out.append((st["names"][j], float(sims[j])))
        if len(out)>=k: break
    return out

def sim_matrix() -> np.ndarray:
    st = _load_state()
    if st["X"] is None: return np.zeros((0,0))
    return cosine_similarity(st["X"])
