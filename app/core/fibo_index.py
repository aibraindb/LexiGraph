from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format
import json, re

DATA = Path("data")
FALLBACK_TTL = DATA / "fibo_trimmed.ttl"
FULL_TTL = DATA / "fibo_full.ttl"
INDEX_JSON = DATA / "fibo_index.json"

PARSER_TRY = ["turtle","xml","n3","nt","trig","trix","json-ld"]
CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

def _ttl_path() -> Path:
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL

def _parse_graph() -> Graph:
    g = Graph()
    p = _ttl_path()
    fmt = guess_format(str(p)) or "turtle"
    tried = []
    for f in [fmt] + [x for x in PARSER_TRY if x != fmt]:
        try:
            g.parse(p, format=f)
            return g
        except Exception:
            tried.append(f)
    raise RuntimeError(f"Failed to parse {_ttl_path()} (tried {tried})")

def _label(g: Graph, u: URIRef) -> str:
    for p in (RDFS.label, SKOS.prefLabel):
        v = g.value(u, p)
        if v: return str(v)
    return str(u).split("/")[-1]

def _tokens(g: Graph, u: URIRef) -> str:
    txts = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u,p,None)):
            try: txts.append(str(o))
            except: pass
    tail = str(u).split("/")[-1]
    if tail:
        txts.append(tail)
        txts.append(CAMEL_RE.sub(" ", tail))
    return " ".join(dict.fromkeys([t.lower() for t in txts if t]))

def _ns_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    return m.group(1) if m else uri.rsplit("/",1)[0] + "/"

def build_index(force: bool=False) -> Dict:
    if INDEX_JSON.exists() and not force:
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    g = _parse_graph()
    classes, edges, ns_counts, props = [], [], {}, []
    seen = set()
    for ctype in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen: continue
            seen.add(su)
            lbl = _label(g, s)
            ns  = _ns_of(su)
            st  = _tokens(g, s)
            ns_counts[ns] = ns_counts.get(ns,0)+1
            classes.append({"uri": su, "label": lbl, "ns": ns, "search_text": st})
    for s,o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])
    for prop,_t,_ in g.triples((None, RDF.type, OWL.ObjectProperty)):
        dom = g.value(prop, RDFS.domain)
        rng = g.value(prop, RDFS.range)
        if dom and rng:
            props.append([str(dom), str(rng), str(prop)])
    idx = {"source": str(_ttl_path()), "classes": classes, "edges": edges,
           "prop_edges": props,
           "namespaces": [{"ns": ns, "count": ns_counts[ns]} for ns in sorted(ns_counts.keys())],
           "active_ns": []}
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

def get_namespaces() -> List[Dict]:
    return _load().get("namespaces", [])

def set_scope(namespaces: List[str]) -> Dict:
    idx = _load()
    valid = {n["ns"] for n in idx.get("namespaces", [])}
    active = [ns for ns in (namespaces or []) if ns in valid]
    idx["active_ns"] = active
    INDEX_JSON.write_text(json.dumps(idx))
    return {"active_ns": active, "total_ns": len(valid)}

def _in_scope(ns: str, active: List[str]) -> bool:
    return (not active) or (ns in active)

def search_scoped(q: str, limit: int = 25, fallback_all: bool=True) -> List[Dict]:
    q = (q or "").strip().lower()
    if not q: return []
    idx = _load()
    act = idx.get("active_ns", [])
    def _hits(pool):
        out = []
        for c in pool:
            st = c.get("search_text") or ((c.get("label","") + " " + c["uri"].split("/")[-1]).lower())
            if q in st:
                out.append({"uri": c["uri"], "label": c["label"], "ns": c["ns"]})
                if len(out)>=limit: break
        return out
    scoped = [c for c in idx.get("classes", []) if _in_scope(c["ns"], act)]
    hits = _hits(scoped)
    if not hits and fallback_all:
        hits = _hits(idx.get("classes", []))
    return hits

def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool=True) -> Dict:
    idx = _load()
    act = idx.get("active_ns", [])
    def in_scope(u: str) -> bool:
        if not act: return True
        for c in idx.get("classes", []):
            if c["uri"] == u:
                return c["ns"] in act
        return True
    und = {}
    for su,ou in idx.get("edges", []):
        if in_scope(su) and in_scope(ou):
            und.setdefault(su,set()).add(ou)
            und.setdefault(ou,set()).add(su)
    if include_properties:
        for dom,rng,_p in idx.get("prop_edges", []):
            if in_scope(dom) and in_scope(rng):
                und.setdefault(dom,set()).add(rng)
                und.setdefault(rng,set()).add(dom)
    seen = {focus_uri}; frontier = {focus_uri}
    for _ in range(max(0,hops)):
        nxt=set()
        for u in list(frontier):
            for v in und.get(u, []):
                if v not in seen: nxt.add(v)
        frontier = nxt; seen |= frontier
    # fallback to all namespaces if too small
    if len(seen)<8 and act:
        und={}
        for su,ou in idx.get("edges", []):
            und.setdefault(su,set()).add(ou)
            und.setdefault(ou,set()).add(su)
        if include_properties:
            for dom,rng,_p in idx.get("prop_edges", []):
                und.setdefault(dom,set()).add(rng)
                und.setdefault(rng,set()).add(dom)
        seen={focus_uri}; frontier={focus_uri}
        for _ in range(max(0,hops)):
            nxt=set()
            for u in list(frontier):
                for v in und.get(u, []):
                    if v not in seen: nxt.add(v)
            frontier=nxt; seen|=frontier
    lookup = {c["uri"]: c for c in idx.get("classes", [])}
    nodes = [{"id": u, "label": lookup.get(u,{}).get("label") or u.split("/")[-1]} for u in seen]
    links=[]
    for su,ou in idx.get("edges", []):
        if su in seen and ou in seen:
            links.append({"source": su, "target": ou, "kind":"subClassOf"})
    if include_properties:
        for dom,rng,prop in idx.get("prop_edges", []):
            if dom in seen and rng in seen:
                links.append({"source": dom, "target": rng, "kind":"property", "label": prop})
    return {"nodes": nodes, "links": links}
