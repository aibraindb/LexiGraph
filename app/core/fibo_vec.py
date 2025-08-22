# app/core/fibo_vec.py
from pathlib import Path
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json, re

DATA = Path("data")
TTL = DATA / "fibo_full.ttl"
IDX = DATA / "fibo_vec.json"

def _label(g, u):
    return (g.value(u, RDFS.label) or g.value(u, SKOS.prefLabel) or str(u).split("/")[-1])

def _collect_classes(g: Graph):
    classes = set()
    for ct in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDFS.type, ct)):
            classes.add(s)
    return list(classes)

def _text_for(g: Graph, u: URIRef):
    parts = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u, p, None)):
            parts.append(str(o))
    tail = str(u).split("/")[-1]
    if tail: parts += [tail, re.sub(r'(?<!^)(?=[A-Z])', ' ', tail)]
    return " ".join(dict.fromkeys([p for p in parts if p]))

def build_fibo_vec(force: bool=False):
    if IDX.exists() and not force:
        return json.loads(IDX.read_text())
    g = Graph()
    g.parse(str(TTL), format="turtle")
    nodes = []
    texts = []
    for u in _collect_classes(g):
        nodes.append(str(u))
        texts.append(_text_for(g, u).lower())
    # safe params for small corpora
    vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    info = {
        "nodes": nodes,
        "vocab": vec.vocabulary_,
        "idf": vec.idf_.tolist(),
        "ngram_range": vec.ngram_range,
        "stop_words": None
    }
    IDX.write_text(json.dumps(info))
    return info

def _load_index():
    if not IDX.exists():
        raise FileNotFoundError("No FIBO index. Run build_fibo_index first or use sidebar button.")
    return json.loads(IDX.read_text())

def _vectorize_texts(vec, texts):
    # rebuild vectorizer from saved state
    v = TfidfVectorizer(vocabulary=vec["vocab"],
                        ngram_range=tuple(vec["ngram_range"]),
                        lowercase=True)
    # set idf
    v._tfidf._idf_diag = None  # guard
    v.idf_ = vec["idf"]
    # transform
    X = v.transform(texts) if hasattr(v, "transform") and hasattr(v, "vocabulary_") else v.fit_transform(texts)
    return v, X

def search_fibo(query: str, top_k=10):
    if not query: return []
    vec = _load_index()
    v = TfidfVectorizer(vocabulary=vec["vocab"],
                        ngram_range=tuple(vec["ngram_range"]),
                        lowercase=True)
    # attach idf
    try:
        v.idf_ = vec["idf"]
    except Exception:
        pass
    qX = v.transform([query.lower()]) if hasattr(v, "transform") else v.fit_transform([query.lower()])
    # compute candidates
    Vall = TfidfVectorizer(vocabulary=vec["vocab"], ngram_range=tuple(vec["ngram_range"]), lowercase=True)
    Vall.idf_ = vec["idf"]
    # dummy 1-hot to infer ndim:
    _ = Vall.fit_transform(["x"])
    # We’ll just cosine against query by projecting each token that exists
    # Simpler: treat query terms as features and rank by feature overlap on nodes:
    nodes = vec["nodes"]
    # Cheap score: number of overlapping features in vocab with query
    q_terms = set([t for t in query.lower().split() if t in vec["vocab"]])
    scores = []
    for n in nodes:
        # proxy: overlap with node token (last path segment)
        tail = n.split("/")[-1].lower()
        tokens = set(re.findall(r"[a-z0-9]+", tail))
        scores.append(len(q_terms & tokens))
    order = sorted(range(len(nodes)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"uri": nodes[i], "label": nodes[i].split("/")[-1], "score": float(scores[i])} for i in order]

# CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    info = build_fibo_vec(force=args.rebuild)
    print(f"FIBO index built: {len(info['nodes'])} nodes, vocab {len(info['vocab'])}")
