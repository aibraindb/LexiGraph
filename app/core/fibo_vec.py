from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json, re
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import SKOS
from rdflib.util import guess_format

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

FALLBACK_TTL = Path("data/fibo_trimmed.ttl")
FULL_TTL     = Path("data/fibo_full.ttl")
INDEX_JSON   = Path("data/fibo_vec_index.json")
MATRIX_NPZ   = Path("data/fibo_vec_matrix.npz")

def _ttl_path() -> Path:
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL

def _labels_for(g: Graph, u: URIRef) -> List[str]:
    out: List[str] = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u,p,None)):
            try: out.append(str(o))
            except: pass
    tail = str(u).split("/")[-1]
    if tail:
        tail_spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", tail)
        out.extend([tail, tail_spaced])
    # de-dup
    seen=set(); dedup=[]
    for t in out:
        if t and t not in seen:
            seen.add(t); dedup.append(t)
    return dedup

def build_fibo_vec(force: bool=False) -> Dict:
    from scipy import sparse
    if INDEX_JSON.exists() and MATRIX_NPZ.exists() and not force:
        try: return json.loads(INDEX_JSON.read_text())
        except: pass
    path = _ttl_path()
    g = Graph()
    fmt = guess_format(str(path)) or "turtle"
    tried = [fmt] + [f for f in ["turtle","xml","n3","nt","trig","trix","json-ld"] if f!=fmt]
    for f in tried:
        try: g.parse(path, format=f); break
        except: continue

    classes: List[Dict] = []
    seen = set()
    for u,_,_ in g.triples((None, None, None)):
        if isinstance(u, URIRef) and str(u).startswith("http"):
            lbs = _labels_for(g, u)
            if lbs:
                su = str(u)
                if su in seen: continue
                seen.add(su)
                classes.append({"uri": su, "labels": lbs})

    docs = [" ".join(c["labels"]).lower() for c in classes]
    if not docs:
        raise RuntimeError("No labeled resources found in FIBO TTL: %s" % path)

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.98)
    X = vectorizer.fit_transform(docs)

    idx = {
        "source": str(path),
        "classes": classes,
        "vocab": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "shape": [int(X.shape[0]), int(X.shape[1])]
    }
    INDEX_JSON.write_text(json.dumps(idx))
    sparse.save_npz(MATRIX_NPZ, X)
    return idx

def _load_vec():
    from scipy import sparse
    idx = json.loads(INDEX_JSON.read_text())
    X = sparse.load_npz(MATRIX_NPZ)
    vect = TfidfVectorizer()
    vect.vocabulary_ = {k:int(v) for k,v in idx["vocab"].items()}
    import numpy as np
    vect.idf_ = np.array(idx["idf"])
    vect._tfidf._idf_diag = None
    return idx, vect, X

def search_fibo(query: str, topk: int=10) -> List[Dict]:
    if not INDEX_JSON.exists() or not MATRIX_NPZ.exists():
        build_fibo_vec(force=True)
    idx, vect, X = _load_vec()
    q = vect.transform([query.lower()])
    sims = cosine_similarity(q, X).flatten()
    order = sims.argsort()[::-1][:topk]
    out = []
    for i in order:
        c = idx["classes"][int(i)]
        out.append({"uri": c["uri"], "labels": c["labels"], "score": float(sims[int(i)])})
    return out
