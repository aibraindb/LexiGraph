from __future__ import annotations
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import SKOS
from rdflib.util import guess_format
from pathlib import Path
from typing import Dict, List

FALLBACK_TTL = Path("data/fibo_trimmed.ttl")
FULL_TTL     = Path("data/fibo_full.ttl")

def _ttl_path() -> Path:
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL

def _labels_for(g: Graph, u: URIRef) -> List[str]:
    out = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u,p,None)):
            try: out.append(str(o))
            except: pass
    tail = str(u).split("/")[-1]
    if tail: out.append(tail)
    seen=set(); dedup=[]
    for t in out:
        if t and t not in seen:
            seen.add(t); dedup.append(t)
    return dedup

def get_attributes_for_class(class_uri: str) -> Dict:
    path = _ttl_path()
    g = Graph(); g.parse(path, format=guess_format(str(path)) or "turtle")
    cls = URIRef(class_uri)
    supers = {cls}
    frontier = {cls}
    while frontier:
        nxt=set()
        for s in list(frontier):
            for _,_,o in g.triples((s, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    props = set()
    # properties constrained by rdfs:domain in class supertree
    for p,_dom,_ in g.triples((None, RDFS.domain, None)):
        dom = g.value(p, RDFS.domain)
        if dom in supers:
            props.add(p)
    out=[]
    for p in props:
        out.append({"property": str(p), "labels": _labels_for(g, p)})
    out.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"count": len(out), "attributes": out}
