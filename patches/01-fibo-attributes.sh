#!/usr/bin/env bash
set -euo pipefail

# --- Ensure folder
mkdir -p patches

# --- Update/augment app/core/fibo_index.py
python3 - <<'PY'
from pathlib import Path
p = Path("app/core/fibo_index.py")
s = p.read_text()

# If schema_for_class already exists, keep it; add attributes_for_class that includes synonyms.
if "def schema_for_class(class_uri:" not in s:
    # Add a minimal schema_for_class using rdflib (safe to add even if not used elsewhere)
    s += """

from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import SKOS
from rdflib.util import guess_format

def schema_for_class(class_uri: str) -> dict:
    path = _ttl_path()
    g = Graph(); g.parse(path, format=guess_format(str(path)) or "turtle")
    supers = set([URIRef(class_uri)])
    frontier = {URIRef(class_uri)}
    while frontier:
        nxt = set()
        for scls in list(frontier):
            for _,_,o in g.triples((scls, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    props = set()
    for p,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in supers:
            props.add(p)
    out = []
    for piri in props:
        lbl = g.value(piri, RDFS.label) or g.value(piri, SKOS.prefLabel) or piri.split("/")[-1]
        out.append({"property": str(piri), "label": str(lbl)})
    out.sort(key=lambda x: x["label"].lower())
    return {"properties": out, "count": len(out)}
"""

if "def attributes_for_class(" not in s:
    s += """

def attributes_for_class(class_uri: str) -> dict:
    \"\"\"Return attributes with synonyms for matching (labels, altLabels, localName).\"\"\"
    path = _ttl_path()
    g = Graph(); g.parse(path, format=guess_format(str(path)) or "turtle")
    # superclasses
    supers = set([URIRef(class_uri)])
    frontier = {URIRef(class_uri)}
    while frontier:
        nxt = set()
        for scls in list(frontier):
            for _,_,o in g.triples((scls, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    # properties
    props = set()
    for piri,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in supers:
            props.add(piri)

    rows = []
    for piri in props:
        labels = set()
        for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
            for _,_,val in g.triples((piri, pred, None)):
                try:
                    labels.add(str(val))
                except Exception:
                    pass
        local = str(piri).split("/")[-1]
        if local:
            labels.add(local)
        rows.append({
            "property": str(piri),
            "labels": sorted({l for l in labels if l})
        })
    rows.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": rows, "count": len(rows)}
"""
p.write_text(s)
print("Patched", p)
PY

# --- Expose new endpoint in app/api/main.py
python3 - <<'PY'
from pathlib import Path
p = Path("app/api/main.py")
s = p.read_text()

if "def fibo_schema(" not in s:
    # add schema endpoint near other fibo endpoints
    s = s.replace("@app.post(\"/fibo/nodeinfo_bulk\")", "@app.post(\"/fibo/nodeinfo_bulk\")\n"
                    "\n@app.get(\"/fibo/schema\")\n"
                    "def fibo_schema(class_uri: str):\n"
                    "    return fibo_index.schema_for_class(class_uri)\n")

if "def fibo_attributes(" not in s:
    s = s.replace("@app.get(\"/fibo/schema\")", "@app.get(\"/fibo/schema\")\n"
                    "def fibo_schema(class_uri: str):\n"
                    "    return fibo_index.schema_for_class(class_uri)\n"
                    "\n@app.get(\"/fibo/attributes\")\n"
                    "def fibo_attributes(class_uri: str):\n"
                    "    return fibo_index.attributes_for_class(class_uri)\n")

Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: /fibo/schema and /fibo/attributes are available."
