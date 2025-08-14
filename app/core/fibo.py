from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS
from pathlib import Path

FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")
EX = Namespace("http://example.org/")
RDFNS = RDF
RDFSNS = RDFS

TTL_PATH = Path("data/fibo.ttl")

def load_fibo() -> Graph:
    g = Graph()
    g.parse(str(TTL_PATH), format="turtle")
    return g

def list_doc_classes():
    g = load_fibo()
    out = []
    for s in g.subjects(RDFNS.type, RDFSNS.Class):
        label = g.label(s)
        if label:
            out.append({"uri": str(s), "label": str(label)})
    # de-dup and sort by label
    seen = set()
    uniq = []
    for item in out:
        key = (item["uri"], item["label"])
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    uniq.sort(key=lambda x: x["label"].lower())
    return uniq

def make_doc_rdf(doc_id: str, class_uri: str, fields: dict | None = None) -> Graph:
    g = Graph()
    subj = URIRef(f"http://example.org/doc/{doc_id}")
    g.bind("fibo", FIBO)
    g.bind("ex", EX)
    g.add((subj, RDF.type, URIRef(class_uri)))
    if fields:
        for pred_qname, value in fields.items():
            if value is None or value == "":
                continue
            if ":" in pred_qname:
                prefix, local = pred_qname.split(":",1)
                if prefix == "fibo":
                    pred = URIRef(str(FIBO) + local)
                elif prefix == "ex":
                    pred = URIRef(str(EX) + local)
                else:
                    pred = URIRef(pred_qname)
            else:
                pred = URIRef(pred_qname)
            g.add((subj, pred, Literal(value)))
    return g

def save_doc_rdf(doc_id: str, graph: Graph):
    out_dir = Path("data/rdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{doc_id}.ttl"
    p.write_text(graph.serialize(format="turtle"))
    # append to all.ttl
    allp = out_dir / "all.ttl"
    with open(allp, "a") as f:
        f.write("\n")
        f.write(graph.serialize(format="turtle"))
    return str(p)
