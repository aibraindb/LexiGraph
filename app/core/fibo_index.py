from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL
from rdflib.util import guess_format
from pathlib import Path
from typing import List, Dict
import json, re

FALLBACK_TTL = Path("data/fibo.ttl")
FULL_TTL     = Path("data/fibo_full.ttl")
INDEX_JSON   = Path("data/fibo_index.json")

PARSER_TRY_ORDER = ["turtle", "xml", "n3", "nt", "trig", "trix", "json-ld"]

def _ttl_path():
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL

def _parse_graph(path: Path) -> Graph:
    g = Graph()
    # try to detect then fall back through formats
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
    lbl = g.value(uri, RDFS.label)
    return str(lbl) if lbl else str(uri).split("/")[-1]

def _namespace_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    if m:
        return m.group(1)
    return uri.rsplit("/", 1)[0] + "/"

def build_index(force: bool=False) -> Dict:
    if INDEX_JSON.exists() and not force:
        try:
            return json.loads(INDEX_JSON.read_text())
        except Exception:
            pass

    path = _ttl_path()
    g = _parse_graph(path)

    classes, edges, ns_counts = [], [], {}

    # include owl:Class and rdfs:Class
    class_types = {OWL.Class, RDFS.Class}
    seen_nodes = set()
    for ctype in class_types:
        for s, _, _ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen_nodes:
                continue
            seen_nodes.add(su)
            lbl = _label_or_tail(g, s)
            ns  = _namespace_of(su)
            ns_counts[ns] = ns_counts.get(ns, 0) + 1
            classes.append({"uri": su, "label": lbl, "ns": ns})

    # rdfs:subClassOf edges
    for s, o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])

    idx = {
        "source": str(path),
        "classes": classes,
        "edges": edges,
        "namespaces": [{"ns": ns, "count": ns_counts[ns]} for ns in sorted(ns_counts.keys())],
        "active_ns": []
    }
    INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)
    INDEX_JSON.write_text(json.dumps(idx))
    return idx

def _load() -> Dict:
    if INDEX_JSON.exists():
        try:
            return json.loads(INDEX_JSON.read_text())
        except Exception:
            pass
    return build_index(force=True)

def get_health() -> Dict:
    idx = _load()
    return {
        "source": idx.get("source"),
        "num_classes": len(idx.get("classes", [])),
        "num_edges": len(idx.get("edges", [])),
        "num_namespaces": len(idx.get("namespaces", [])),
        "active_ns": idx.get("active_ns", [])
    }

def get_namespaces() -> List[Dict]:
    idx = _load()
    return idx.get("namespaces", [])

def set_scope(namespaces: List[str]) -> Dict:
    idx = _load()
    valid = {n["ns"] for n in idx.get("namespaces", [])}
    active = [ns for ns in (namespaces or []) if ns in valid]
    idx["active_ns"] = active
    INDEX_JSON.write_text(json.dumps(idx))
    return {"active_ns": active, "total_ns": len(valid)}

def _in_scope(ns: str, active: List[str]) -> bool:
    return (not active) or (ns in active)

def get_scoped_classes() -> List[Dict]:
    idx = _load()
    act = idx.get("active_ns", [])
    return [c for c in idx.get("classes", []) if _in_scope(c["ns"], act)]

def search_scoped(q: str, limit: int = 25) -> List[Dict]:
    q = (q or "").strip().lower()
    if not q:
        return []
    cls = get_scoped_classes()
    hits = []
    for c in cls:
        tail = c["uri"].split("/")[-1].lower()
        if q in (c["label"] or "").lower() or q in tail:
            hits.append({"uri": c["uri"], "label": c["label"], "ns": c["ns"]})
            if len(hits) >= limit:
                break
    return hits

def subgraph_scoped(focus_uri: str, hops: int = 2) -> Dict:
    idx = _load()
    act = idx.get("active_ns", [])
    uris_in_scope = {c["uri"] for c in idx.get("classes", []) if _in_scope(c["ns"], act)}
    uris_in_scope.add(focus_uri)  # always include focus

    children, parents = {}, {}
    for su, ou in idx.get("edges", []):
        if su in uris_in_scope and ou in uris_in_scope:
            children.setdefault(ou, set()).add(su)
            parents.setdefault(su, set()).add(ou)

    seen = {focus_uri}
    frontier = {focus_uri}
    for _ in range(max(0, hops)):
        nxt = set()
        for u in list(frontier):
            for p in parents.get(u, []):
                if p not in seen: nxt.add(p)
            for ch in children.get(u, []):
                if ch not in seen: nxt.add(ch)
        frontier = nxt - seen
        seen |= frontier

    node_lookup = {c["uri"]: c for c in idx.get("classes", [])}
    nodes = [{"id": u, "label": node_lookup.get(u, {}).get("label", u.split("/")[-1])} for u in seen]
    links = [{"source": su, "target": ou} for su, ou in idx.get("edges", []) if su in seen and ou in seen]
    return {"nodes": nodes, "links": links}
