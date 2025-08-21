# app/core/fibo_vec.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re, json
import joblib
from rdflib import Graph, RDFS,RDF, URIRef
from rdflib.namespace import SKOS, OWL
from rdflib.util import guess_format
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

FALLBACK_TTL = Path("data/fibo_trimmed.ttl")
FULL_TTL     = Path("data/fibo_full.ttl")
MODEL_PATH   = Path("data/fibo_vec.joblib")
INDEX_PATH   = Path("data/fibo_index_meta.json")

PARSER_TRY_ORDER = ["turtle","xml","n3","nt","trig","trix","json-ld"]
CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

_state: Dict[str, object] = {
    "vectorizer": None,   # TfidfVectorizer
    "X": None,            # sparse matrix
    "docs": [],           # list[str] corpus texts
    "uris": [],           # list[str] aligned URIs
    "labels": [],         # list[str] aligned labels
    "source": None        # str path used
}

def _ttl_path() -> Path:
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL

def _parse_graph(path: Path) -> Graph:
    g = Graph()
    fmt = guess_format(str(path)) or "turtle"
    tried = []
    for f in [fmt] + [x for x in PARSER_TRY_ORDER if x != fmt]:
        try:
            g.parse(path, format=f)
            return g
        except Exception:
            tried.append(f)
    raise RuntimeError(f"Failed to parse {path} (tried {tried})")

def _label_or_tail(g: Graph, u: URIRef) -> str:
    lbl = g.value(u, RDFS.label) or g.value(u, SKOS.prefLabel)
    if lbl:
        try: return str(lbl)
        except Exception: pass
    return str(u).split("/")[-1]

def _tokenize_for_search(g: Graph, u: URIRef) -> str:
    txts = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,val in g.triples((u, p, None)):
            try:
                s = str(val).strip()
                if s: txts.append(s)
            except Exception:
                pass
    tail = str(u).split("/")[-1]
    if tail:
        txts.append(tail)
        txts.append(CAMEL_RE.sub(" ", tail))
    # de-dupe, lower
    seen=set(); out=[]
    for t in txts:
        t=t.lower()
        if t and t not in seen:
            seen.add(t); out.append(t)
    return " ".join(out)

def _collect_classes(g: Graph) -> List[Tuple[str,str,str]]:
    out=[]
    seen=set()
    for ctype in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            #                               ^^^ was RDFS.type
            su=str(s)
            if su in seen: continue
            seen.add(su)
            lab=_label_or_tail(g, s)
            txt=_tokenize_for_search(g, s)
            out.append((su, lab, txt))

    # also include any resource that has a label/prefLabel even if not typed
    for s,_,_ in g.triples((None, RDFS.label, None)):
        su=str(s)
        if su in seen:
            continue
        seen.add(su)
        lab=_label_or_tail(g, s)
        txt=_tokenize_for_search(g, s)
        out.append((su, lab, txt))
    return [t for t in out if t[2].strip()]



def _persist():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "vectorizer": _state["vectorizer"],
        "X": _state["X"],
        "uris": _state["uris"],
        "labels": _state["labels"],
        "source": _state["source"]
    }, MODEL_PATH)
    INDEX_PATH.write_text(json.dumps({
        "count": len(_state["uris"]),
        "source": _state["source"]
    }))

def _load_persisted() -> bool:
    if not MODEL_PATH.exists():
        return False
    try:
        blob = joblib.load(MODEL_PATH)
        _state["vectorizer"] = blob["vectorizer"]
        _state["X"] = blob["X"]
        _state["uris"] = blob["uris"]
        _state["labels"] = blob["labels"]
        _state["source"] = blob.get("source")
        # sanity: fitted?
        if not hasattr(_state["vectorizer"], "idf_"):
            return False
        return True
    except Exception:
        return False

def _is_fitted() -> bool:
    vec = _state.get("vectorizer")
    X   = _state.get("X")
    return (vec is not None) and hasattr(vec, "idf_") and (X is not None)

def build_fibo_vec(force: bool=False) -> Dict:
    """
    Build (or rebuild) TF-IDF index from local TTL.
    """
    if _is_fitted() and not force:
        return {"status":"ok","source":_state["source"],"count":len(_state["uris"])}
    if not force and _load_persisted():
        return {"status":"ok","source":_state["source"],"count":len(_state["uris"])}

    path = _ttl_path()
    if not path.exists():
        raise FileNotFoundError(f"No TTL found. Expected {FULL_TTL} or {FALLBACK_TTL}")

    g = _parse_graph(path)
    triples = _collect_classes(g)
    if not triples:
        raise RuntimeError("FIBO corpus is empty; cannot fit vectorizer")

    uris, labels, docs = zip(*triples)
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_df=0.9)
    X = vectorizer.fit_transform(docs)

    _state.update({
        "vectorizer": vectorizer,
        "X": X,
        "docs": list(docs),
        "uris": list(uris),
        "labels": list(labels),
        "source": str(path)
    })
    _persist()
    return {"status":"ok","source":str(path),"count":len(uris)}

def search_fibo(query: str, topk: int=10) -> List[Dict]:
    """
    Safe search: auto-builds on first use if needed.
    """
    if not query or not query.strip():
        return []
    if not _is_fitted():
        # Try to load persisted; if still not fitted, build
        if not _load_persisted():
            build_fibo_vec(force=False)

    vec = _state["vectorizer"]; X = _state["X"]
    qv  = vec.transform([query])
    # simple cosine (rows are L2-normalized by TfidfVectorizer)
    scores = (X @ qv.T).toarray().ravel()
    if scores.size == 0:
        return []

    idx = np.argsort(scores)[::-1][:max(1, topk)]
    out=[]
    for i in idx:
        out.append({
            "uri": _state["uris"][i],
            "label": _state["labels"][i],
            "score": float(scores[i])
        })
    return out

# --- CLI helper -------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild TF-IDF index from TTL")
    args = ap.parse_args()
    info = build_fibo_vec(force=args.rebuild)
    print(json.dumps(info, indent=2))
