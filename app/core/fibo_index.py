from __future__ import annotations
import os, json
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import SKOS, RDF, OWL

from .vector_store import build_from_texts

def build_fibo_texts(ttl_path: str) -> list[str]:
    g = Graph()
    g.parse(ttl_path)
    texts = []
    for s,_,_ in g.triples((None, RDF.type, OWL.Class)):
        label = g.value(s, RDFS.label) or g.value(s, SKOS.prefLabel) or URIRef(s).split('/')[-1]
        comment = g.value(s, RDFS.comment) or ""
        t = f"{label}\n{comment}"
        texts.append(str(t))
    return texts

def build_fibo_vectors(ttl_path: str, name: str="fibo"):
    texts = build_fibo_texts(ttl_path)
    if not texts:
        return {"status":"no_classes_found"}
    return build_from_texts(name, texts)
