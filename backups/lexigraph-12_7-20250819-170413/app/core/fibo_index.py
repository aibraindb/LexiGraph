from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json, re
from rdflib import Graph, RDF, RDFS, URIRef, BNode
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

FALLBACK_TTL = Path("data/fibo_trimmed.ttl")
FULL_TTL     = Path("data/fibo_full.ttl")
INDEX_JSON   = Path("data/fibo_index.json")

PARSER_TRY_ORDER = ["turtle", "xml", "n3", "nt", "trig", "trix", "json-ld"]
CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def _ttl_path():
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
    raise RuntimeError(f"Failed to parse {path} with formats tried: {tried}")

def _label_or_tail(g: Graph, uri: URIRef) -> str:
    lbl = g.value(uri, RDFS.label) or g.value(uri, SKOS.prefLabel)
    return str(lbl) if lbl else str(uri).split("/")[-1]

def _tokenize_for_search(g: Graph, uri: URIRef) -> str:
    txts = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _, _, val in g.triples((uri, p, None)):
            try: txts.append(str(val))
            except Exception: pass
    tail = str(uri).split("/")[-1]
    if tail:
        txts.append(tail)
        txts.append(CAMEL_RE.sub(" ", tail))
    return " ".join(dict.fromkeys([t.lower() for t in txts if t]))

def _namespace_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    if m: return m.group(1)
    return uri.rsplit("/", 1)[0] + "/"

def build_index(force: bool=False) -> Dict:
    if INDEX_JSON.exists() and not force:
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    path = _ttl_path()
    g = _parse_graph(path)
    classes, edges, ns_counts, prop_edges = [], [], {}, []
    class_types = {OWL.Class, RDFS.Class}
    seen = set()
    for ctype in class_types:
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen: continue
            seen.add(su)
            lbl = _label_or_tail(g, s)
            ns  = _namespace_of(su)
            search_text = _tokenize_for_search(g, s)
            ns_counts[ns] = ns_counts.get(ns, 0) + 1
            classes.append({"uri": su, "label": lbl, "ns": ns, "search_text": search_text})
    for s,o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])
    for prop,_t,_ in g.triples((None, RDF.type, OWL.ObjectProperty)):
        dom = g.value(prop, RDFS.domain); rng = g.value(prop, RDFS.range)
        if dom and rng: prop_edges.append([str(dom), str(rng), str(prop)])

    idx = {"source": str(path), "classes": classes, "edges": edges,
           "prop_edges": prop_edges,
           "namespaces": [{"ns": ns, "count": ns_counts[ns]} for ns in sorted(ns_counts.keys())],
           "active_ns": []}
    INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(json.dumps(idx))
    return idx

def _load() -> Dict:
    if INDEX_JSON.exists():
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    return build_index(force=True)

def get_health() -> Dict:
    idx = _load()
    return {"source": idx.get("source"),
            "num_classes": len(idx.get("classes", [])),
            "num_edges": len(idx.get("edges", [])),
            "num_namespaces": len(idx.get("namespaces", [])),
            "active_ns": idx.get("active_ns", [])}

def search_scoped(q: str, limit: int = 25, fallback_all: bool = True) -> List[Dict]:
    q = (q or "").strip().lower()
    if not q: return []
    idx = _load(); act = idx.get("active_ns", [])
    def _in_scope(ns): return (not act) or (ns in act)
    def _hits(pool):
        res = []
        for c in pool:
            st = c.get("search_text") or (c.get("label","").lower()+" "+c["uri"].split("/")[-1].lower())
            if q in st:
                res.append({"uri": c["uri"], "label": c["label"], "ns": c["ns"]})
                if len(res) >= limit: break
        return res
    scoped = [c for c in idx.get("classes", []) if _in_scope(c["ns"])]
    hits = _hits(scoped)
    if not hits and fallback_all: hits = _hits(idx.get("classes", []))
    return hits

# tolerant attributes
def _parse_ttl(path): 
    g = Graph(); g.parse(path, format=guess_format(str(path)) or "turtle"); return g

def _superclasses(g: Graph, cls: URIRef) -> set:
    supers = {cls}; frontier = {cls}
    while frontier:
        nxt = set()
        for s in list(frontier):
            for _,_,o in g.triples((s, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
                if isinstance(o, BNode):
                    supers.add(o)
        frontier = nxt
    return supers

def _labels(g: Graph, node: URIRef) -> list[str]:
    vals = set()
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,v in g.triples((node, p, None)):
            try: vals.add(str(v))
            except: pass
    tail = str(node).split("/")[-1]
    if tail: vals.add(tail)
    return sorted(vals)

def attributes_for_class_tolerant(ttl_path: str|Path, class_uri: str) -> dict:
    path = Path(ttl_path)
    g = _parse_ttl(path)
    C = URIRef(class_uri)
    supers = _superclasses(g, C)
    props = set()
    for p,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in supers: props.add(p)
    for s,_,o in g.triples((C, RDFS.subClassOf, None)):
        stack = [o]; seen=set()
        while stack:
            node = stack.pop()
            if node in seen: continue
            seen.add(node)
            if isinstance(node, BNode):
                if (node, RDF.type, OWL.Restriction) in g:
                    for _,_,p in g.triples((node, OWL.onProperty, None)):
                        props.add(p)
                for _,_,child in g.triples((node, RDFS.subClassOf, None)):
                    stack.append(child)
    rows = []
    for p in props:
        rows.append({"property": str(p), "labels": _labels(g, p)})
    rows.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": rows, "count": len(rows)}
