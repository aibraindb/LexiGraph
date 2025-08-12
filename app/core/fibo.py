from typing import Dict, Any
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF

FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
EX = Namespace("http://example.org/docs/")

def to_rdf(doc_id:str, fibo_class_curie:str, fields:Dict[str,Any])->str:
    g = Graph(); g.bind("fibo", FIBO); g.bind("ex", EX)
    node = EX[doc_id]
    cls = fibo_class_curie.split(":")[-1]
    g.add((node, RDF.type, URIRef(FIBO[cls])))
    for k,v in fields.items():
        if not isinstance(v, dict): continue
        val = v.get('value'); prop = v.get('fibo_property')
        if not val or not prop: continue
        prop_name = prop.split(":")[-1]
        g.add((node, URIRef(FIBO[prop_name]), Literal(str(val))))
    return g.serialize(format="turtle")
