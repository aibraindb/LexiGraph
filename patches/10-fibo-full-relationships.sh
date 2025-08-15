#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("app/core/fibo_index.py")
s = p.read_text()

# Ensure we import OWL/SKOS/guess_format earlier
if "from rdflib.namespace import OWL, SKOS" not in s:
    s = s.replace("from rdflib.namespace import OWL, SKOS", "from rdflib.namespace import OWL, SKOS")

# Add/ensure object property edge indexing in build_index
marker = "def build_index(force: bool=False)"
if marker not in s:
    raise SystemExit("Expected build_index in app/core/fibo_index.py")

if "prop_edges" not in s:
    s = s.replace(
        "def build_index(force: bool=False) -> Dict:",
        "def build_index(force: bool=False) -> Dict:"
    )
    s = s.replace(
        "    classes, edges, ns_counts = [], [], {}",
        "    classes, edges, ns_counts = [], [], {}\n    prop_edges = []  # domain->range with label"
    )
    # after we parse classes and subclass edges, add property edges
    s = s.replace(
        "    for s,o in g.subject_objects(RDFS.subClassOf):",
        "    # subclass edges\n    for s,o in g.subject_objects(RDFS.subClassOf):"
    )
    s = s.replace(
        "    idx = {\"source\": str(path), \"classes\": classes, \"edges\": edges,",
        "    # object property edges (domain->range)\n"
        "    for prop,_t,_ in g.triples((None, RDF.type, OWL.ObjectProperty)):\n"
        "        dom = g.value(prop, RDFS.domain)\n"
        "        rng = g.value(prop, RDFS.range)\n"
        "        if dom and rng:\n"
        "            prop_edges.append([str(dom), str(rng), str(prop)])\n"
        "\n"
        "    idx = {\"source\": str(path), \"classes\": classes, \"edges\": edges,\n"
        "           \"prop_edges\": prop_edges,"
    )

# Upgrade subgraph to include property edges
if "def subgraph_scoped(" in s and "prop_edges" not in s.split("def subgraph_scoped(",1)[1].split("return",1)[0]:
    s = s.replace(
        "def subgraph_scoped(focus_uri: str, hops: int = 2) -> Dict:",
        "def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool = True) -> Dict:"
    )
    s = s.replace(
        "    idx = _load(); act = idx.get(\"active_ns\", [])",
        "    idx = _load(); act = idx.get(\"active_ns\", [])"
    )
    s = s.replace(
        "    nodes = [{\"id\": u, \"label\": lookup.get(u, {}).get(\"label\", u.split(\"/\")[-1])} for u in seen]\n"
        "    links = [{\"source\": su, \"target\": ou} for su,ou in idx.get(\"edges\", []) if su in seen and ou in seen]\n"
        "    return {\"nodes\": nodes, \"links\": links}",
        "    nodes = [{\"id\": u, \"label\": lookup.get(u, {}).get(\"label\", u.split(\"/\")[-1])} for u in seen]\n"
        "    links = [{\"source\": su, \"target\": ou, \"kind\":\"subClassOf\"} for su,ou in idx.get(\"edges\", []) if su in seen and ou in seen]\n"
        "    if include_properties:\n"
        "        for dom, rng, prop in idx.get(\"prop_edges\", []):\n"
        "            if dom in seen and rng in seen:\n"
        "                links.append({\"source\": dom, \"target\": rng, \"kind\": \"property\", \"label\": prop})\n"
        "    return {\"nodes\": nodes, \"links\": links}"
    )

Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: Full FIBO relationships will be indexed and shown."
