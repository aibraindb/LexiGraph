from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format
import joblib, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data")
TTL_FULL = DATA / "fibo_full.ttl"
TTL_TRIM = DATA / "fibo_trimmed.ttl"
MODEL    = DATA / "fibo_vec.joblib"

def _ttl_path() -> Path:
    return TTL_FULL if TTL_FULL.exists() else TTL_TRIM

def _tokenize_label(s: str) -> str:
    s = s or ""
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    s = s.replace("/", " ").replace("_", " ")
    return s.lower()

def _collect_labels(g: Graph) -> List[Dict]:
    out = []
    class_types = {OWL.Class, RDFS.Class}
    seen = set()
    for ctype in class_types:
        for s,_,_ in g.triples((None, RDFS.type, ctype)):
            if s in seen: continue
            seen.add(s)
            uri = str(s)
            labels = set()
            for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
                for _s,_p,o in g.triples((s,p,None)):
                    try: labels.add(str(o))
                    except: pass
            local = uri.split("/")[-1]
            if local: labels.add(local)
            text = " ".join(sorted(labels))
            out.append({"uri": uri, "label": text or local})
    return out

def rebuild_fibo_index() -> Dict:
    g = Graph()
    fmt = guess_format(str(_ttl_path())) or "turtle"
    g.parse(_ttl_path(), format=fmt)
    rows = _collect_labels(g)
    corpus = [_tokenize_label(r["label"]) for r in rows]
    vec = TfidfVectorizer(min_df=1, max_df=1.0, token_pattern=r"(?u)\b\w[\w\-/\.]+\b")
    X = vec.fit_transform(corpus)
    joblib.dump({"rows": rows, "vec": vec, "X": X}, MODEL)
    return {"classes": len(rows), "features": X.shape[1]}

def ensure_fibo_index():
    if not MODEL.exists():
        rebuild_fibo_index()

def fibo_search(query: str, topk: int = 5) -> List[Dict]:
    ensure_fibo_index()
    blob = joblib.load(MODEL)
    rows, vec, X = blob["rows"], blob["vec"], blob["X"]
    qv = vec.transform([_tokenize_label(query or "")])
    sims = cosine_similarity(X, qv).ravel()
    idx = sims.argsort()[::-1][:max(1, topk)]
    return [{"uri": rows[i]["uri"], "label": rows[i]["label"], "score": float(sims[i])} for i in idx]
