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
