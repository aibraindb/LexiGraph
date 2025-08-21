from __future__ import annotations
from typing import Dict, List
from pathlib import Path
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

DATA = Path("data")
TTL_FULL = DATA / "fibo_full.ttl"
TTL_TRIM = DATA / "fibo_trimmed.ttl"

def _ttl_path() -> Path:
    return TTL_FULL if TTL_FULL.exists() else TTL_TRIM

def _g() -> Graph:
    g = Graph()
    g.parse(_ttl_path(), format=guess_format(str(_ttl_path())) or "turtle")
    return g

def _super(g: Graph, c: URIRef) -> set:
    S = {c}; Q = {c}
    while Q:
        nxt=set()
        for s in list(Q):
            for _,_,o in g.triples((s, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in S:
                    S.add(o); nxt.add(o)
        Q = nxt
    return S

def attributes_for_class(class_uri: str) -> Dict:
    g = _g()
    c = URIRef(class_uri)
    doms = _super(g, c)
    props=set()
    for p,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in doms:
            props.add(p)
    out=[]
    for p in props:
        labels=set()
        for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
            for _,_,v in g.triples((p,pred,None)):
                try: labels.add(str(v))
                except: pass
        local = str(p).split("/")[-1]
        if local: labels.add(local)
        out.append({"property": str(p), "labels": sorted(labels)})
    out.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": out, "count": len(out)}
