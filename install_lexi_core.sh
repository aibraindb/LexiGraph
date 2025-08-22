#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
mkdir -p app/core data components
touch app/__init__.py
touch app/core/__init__.py

############################################
# app/core/pdf_text.py
############################################
cat > app/core/pdf_text.py <<'PY'
from __future__ import annotations
import io, base64
from typing import List, Dict, Any
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. pip install pymupdf") from e

def _page_text_and_spans(page: "fitz.Page") -> Dict[str, Any]:
    blocks = page.get_text("dict")["blocks"]
    lines_out = []
    for b in blocks:
        if "lines" not in b: continue
        for line in b["lines"]:
            txt = "".join([s.get("text","") for s in line.get("spans",[]) ]).strip()
            if not txt: continue
            bbox = line.get("bbox", None)
            spans = []
            for s in line.get("spans", []):
                spans.append({
                    "text": s.get("text",""),
                    "bbox": s.get("bbox", None),
                    "size": s.get("size", None),
                })
            lines_out.append({"text": txt, "bbox": bbox, "spans": spans})
    text_all = "\n".join(ln["text"] for ln in lines_out)
    return {"text": text_all, "lines": lines_out}

def extract_text_blocks(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    try:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            data = _page_text_and_spans(page)
            out.append({"page": pno, "text": data["text"], "spans": data["lines"]})
    finally:
        doc.close()
    return out

def get_page_images(pdf_bytes: bytes, zoom: float=2.0) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    try:
        for pno in range(len(doc)):
            page = doc.load_page(pno)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images

def get_page_images_as_base64(pdf_bytes: bytes, zoom: float=2.0) -> List[str]:
    out=[]
    for raw in get_page_images(pdf_bytes, zoom=zoom):
        b64 = base64.b64encode(raw).decode("ascii")
        out.append("data:image/png;base64,"+b64)
    return out

def focused_summary(text: str, max_chars: int=2000) -> str:
    if not text: return ""
    lines = text.splitlines()
    head = "\n".join(lines[:60])
    if len(head) > max_chars: return head[:max_chars]
    if len(text) <= max_chars: return text
    return text[:max_chars]
PY

############################################
# app/core/fibo_index.py
############################################
cat > app/core/fibo_index.py <<'PY'
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json, re
from rdflib import Graph, URIRef, RDF, RDFS
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

DATA = Path("data")
FULL_TTL   = DATA/"fibo_full.ttl"
SMALL_TTL  = DATA/"fibo.ttl"
INDEX_JSON = DATA/"fibo_index.json"

PARSER_ORDER = ["turtle","xml","n3","nt","trig","trix","json-ld"]
CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def _ttl_path() -> Path:
    if FULL_TTL.exists(): return FULL_TTL
    if SMALL_TTL.exists(): return SMALL_TTL
    raise FileNotFoundError("No FIBO TTL found. Place fibo_full.ttl or fibo.ttl in data/")

def _parse_graph(path: Path) -> Graph:
    g = Graph()
    fmt = guess_format(str(path)) or "turtle"
    tried=[]
    for f in [fmt]+[x for x in PARSER_ORDER if x!=fmt]:
        try:
            g.parse(path, format=f)
            return g
        except Exception:
            tried.append(f)
    raise RuntimeError(f"Failed to parse {path}; tried: {tried}")

def _label(g: Graph, u: URIRef) -> str:
    v = g.value(u, RDFS.label) or g.value(u, SKOS.prefLabel)
    return str(v) if v else str(u).split("/")[-1]

def _search_text(g: Graph, u: URIRef) -> str:
    parts=[]
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u,p,None)):
            try: parts.append(str(o))
            except: pass
    tail=str(u).split("/")[-1]
    if tail:
        parts.append(tail)
        parts.append(CAMEL_RE.sub(" ", tail))
    # dedupe, lower
    seen=[]
    for t in parts:
        t=t.strip().lower()
        if t and t not in seen: seen.append(t)
    return " ".join(seen)

def _ns_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    return m.group(1) if m else uri.rsplit("/",1)[0]+"/"

def build_index(force: bool=False) -> Dict[str, Any]:
    if INDEX_JSON.exists() and not force:
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    path = _ttl_path()
    g = _parse_graph(path)

    classes=[]
    edges=[]
    prop_edges=[]
    ns_counts={}
    seen=set()
    for ctype in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            su=str(s)
            if su in seen: continue
            seen.add(su)
            lbl=_label(g,s)
            ns=_ns_of(su)
            st=_search_text(g,s)
            ns_counts[ns]=ns_counts.get(ns,0)+1
            classes.append({"uri":su,"label":lbl,"ns":ns,"search_text":st})

    for s,o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])

    for p,_,_ in g.triples((None, RDF.type, OWL.ObjectProperty)):
        dom = g.value(p, RDFS.domain)
        rng = g.value(p, RDFS.range)
        if dom and rng:
            prop_edges.append([str(dom), str(rng), str(p)])

    idx={"source":str(path),
         "classes":classes,
         "edges":edges,
         "prop_edges":prop_edges,
         "namespaces":[{"ns":ns,"count":ns_counts[ns]} for ns in sorted(ns_counts.keys())],
         "active_ns":[]}
    INDEX_JSON.parent.mkdir(parents=True,exist_ok=True)
    INDEX_JSON.write_text(json.dumps(idx))
    return idx

def _load() -> Dict[str, Any]:
    if INDEX_JSON.exists():
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    return build_index(force=True)

def search_scoped(q: str, limit: int=25, fallback_all: bool=True) -> List[Dict[str,str]]:
    q=(q or "").strip().lower()
    if not q: return []
    idx=_load()
    def _pool(all_classes):  # rank by substring match quality
        hits=[]
        for c in all_classes:
            st=c.get("search_text") or (c.get("label","").lower()+" "+c["uri"].split("/")[-1].lower())
            if q in st:
                hits.append({"uri":c["uri"],"label":c["label"],"ns":c["ns"]})
                if len(hits)>=limit: break
        return hits
    # If you later add active namespaces, filter here; for now just all
    hits=_pool(idx.get("classes",[]))
    return hits

def subgraph_scoped(focus_uri: str, hops: int=2, include_props: bool=True) -> Dict[str, Any]:
    idx=_load()
    und={}
    for su,ou in idx.get("edges",[]):
        und.setdefault(su,set()).add(ou)
        und.setdefault(ou,set()).add(su)
    if include_props:
        for dom,rng,_ in idx.get("prop_edges",[]):
            und.setdefault(dom,set()).add(rng)
            und.setdefault(rng,set()).add(dom)
    seen={focus_uri}
    frontier={focus_uri}
    for _ in range(max(0,hops)):
        nxt=set()
        for u in list(frontier):
            for v in und.get(u,[]):
                if v not in seen: nxt.add(v)
        frontier=nxt; seen|=frontier
    lookup={c["uri"]:c for c in idx.get("classes",[])}
    nodes=[{"id":u,"label": lookup.get(u,{}).get("label", u.split("/")[-1])} for u in seen]
    links=[]
    for su,ou in idx.get("edges",[]):
        if su in seen and ou in seen:
            links.append({"source":su,"target":ou,"kind":"subClassOf"})
    if include_props:
        for dom,rng,prop in idx.get("prop_edges",[]):
            if dom in seen and rng in seen:
                links.append({"source":dom,"target":rng,"kind":"property","label":prop})
    return {"nodes":nodes,"links":links}

if __name__=="__main__":
    import argparse, json as _json
    ap=argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args=ap.parse_args()
    idx=build_index(force=args.rebuild)
    print(_json.dumps({"classes":len(idx.get("classes",[])), "edges":len(idx.get("edges",[]))}))
PY

############################################
# app/core/fibo_attrs.py
############################################
cat > app/core/fibo_attrs.py <<'PY'
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from rdflib import Graph, URIRef, RDFS
from rdflib.namespace import SKOS
from rdflib.util import guess_format

DATA = Path("data")
TTL = (DATA/"fibo_full.ttl") if (DATA/"fibo_full.ttl").exists() else (DATA/"fibo.ttl")

def attributes_for_class(class_uri: str) -> Dict[str, Any]:
    if not TTL.exists():
        return {"attributes": [], "count": 0}
    g = Graph()
    g.parse(TTL, format=guess_format(str(TTL)) or "turtle")
    # climb superclasses
    supers=set([URIRef(class_uri)])
    frontier={URIRef(class_uri)}
    while frontier:
        nxt=set()
        for s in list(frontier):
            for _,_,o in g.triples((s, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier=nxt
    # properties with domain in supers
    props=set()
    for p,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in supers: props.add(p)
    rows=[]
    for p in props:
        labels=set()
        for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
            for _,_,val in g.triples((p, pred, None)):
                try: labels.add(str(val))
                except: pass
        local=str(p).split("/")[-1]
        if local: labels.add(local)
        rows.append({"property": str(p), "labels": sorted(l for l in labels if l)})
    rows.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": rows, "count": len(rows)}
PY

############################################
# app/core/fibo_vec.py
############################################
cat > app/core/fibo_vec.py <<'PY'
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
    vec._tfidf._idf_diag = None
    # lazy-fit _tfidf with provided idf
    def _set_idf(v):
        from scipy import sparse
        vec._tfidf.idf_ = idf
        vec._tfidf._idf_diag = sparse.spdiags(idf, diags=0, m=len(idf), n=len(idf))
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
PY

############################################
# app/core/index_docs.py
############################################
cat > app/core/index_docs.py <<'PY'
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
PY

############################################
# Optional tiny README note (core)
############################################
cat > app/core/README_CORE.md <<'MD'
These modules power the Streamlit UI:

- `pdf_text.py`    — robust text + spans + page images via PyMuPDF (no OCR)
- `fibo_index.py`  — parse FIBO TTL, index classes/edges, search & subgraph
- `fibo_attrs.py`  — collect properties (labels+synonyms) for a FIBO class via rdfs:domain
- `fibo_vec.py`    — TF-IDF vectors over FIBO classes for fuzzy matching
- `index_docs.py`  — simple TF-IDF index for uploaded documents (MVP)

After copying your `data/fibo_full.ttl`, run:

    python -m app.core.fibo_index --rebuild
    python -m app.core.fibo_vec   --rebuild

Then start the Streamlit UI.
MD

echo "✅ Installed core modules into app/core/"
