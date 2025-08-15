# app/core/fibo_index.py
from pathlib import Path
from typing import List, Dict
import json, re

from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

FALLBACK_TTL = Path("data/fibo.ttl")         # small demo TTL
FULL_TTL     = Path("data/fibo_full.ttl")    # full FIBO, if present
INDEX_JSON   = Path("data/fibo_index.json")

PARSER_TRY_ORDER = ["turtle", "xml", "n3", "nt", "trig", "trix", "json-ld"]
CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')


def _ttl_path() -> Path:
    return FULL_TTL if FULL_TTL.exists() else FALLBACK_TTL


def _is_uri(u: str) -> bool:
    return isinstance(u, str) and (u.startswith("http://") or u.startswith("https://"))


def _parse_graph(path: Path) -> Graph:
    g = Graph()
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
    lbl = g.value(uri, RDFS.label) or g.value(uri, SKOS.prefLabel)
    return str(lbl) if lbl else str(uri).split("/")[-1]


def _tokenize_for_search(g: Graph, uri: URIRef) -> str:
    txts = []
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _, _, val in g.triples((uri, p, None)):
            try:
                txts.append(str(val))
            except Exception:
                pass
    tail = str(uri).split("/")[-1]
    if tail:
        txts.append(tail)
        txts.append(CAMEL_RE.sub(" ", tail))
    # unique, lowercased
    return " ".join(dict.fromkeys([t.lower() for t in txts if t]))


def _namespace_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    if m:
        return m.group(1)
    return uri.rsplit("/", 1)[0] + "/"


def build_index(force: bool = False) -> Dict:
    if INDEX_JSON.exists() and not force:
        try:
            return json.loads(INDEX_JSON.read_text())
        except Exception:
            pass

    path = _ttl_path()
    g = _parse_graph(path)

    classes: List[Dict] = []
    edges: List[List[str]] = []
    prop_edges: List[List[str]] = []
    ns_counts: Dict[str, int] = {}

    # Collect classes from both owl:Class and rdfs:Class
    seen = set()
    for ctype in (OWL.Class, RDFS.Class):
        for s, _, _ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen:
                continue
            seen.add(su)
            lbl = _label_or_tail(g, s)
            ns = _namespace_of(su)
            ns_counts[ns] = ns_counts.get(ns, 0) + 1
            classes.append({
                "uri": su,
                "label": lbl,
                "ns": ns,
                "search_text": _tokenize_for_search(g, s),
            })

    # subclass edges (URI-only)
    for s, o in g.subject_objects(RDFS.subClassOf):
        su, ou = str(s), str(o)
        if _is_uri(su) and _is_uri(ou):
            edges.append([su, ou])

    # object property edges (domain->range), keep only if both ends are URIs
    for prop, _t, _ in g.triples((None, RDF.type, OWL.ObjectProperty)):
        dom = g.value(prop, RDFS.domain)
        rng = g.value(prop, RDFS.range)
        if dom and rng:
            du, ru = str(dom), str(rng)
            if _is_uri(du) and _is_uri(ru):
                prop_edges.append([du, ru, str(prop)])

    idx = {
        "source": str(path),
        "classes": classes,
        "edges": edges,
        "prop_edges": prop_edges,
        "namespaces": [{"ns": ns, "count": ns_counts[ns]} for ns in sorted(ns_counts.keys())],
        "active_ns": [],  # scoped list of namespaces
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
        "active_ns": idx.get("active_ns", []),
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


def search_scoped(q: str, limit: int = 25, fallback_all: bool = True) -> List[Dict]:
    q = (q or "").strip().lower()
    if not q:
        return []
    idx = _load()
    act = idx.get("active_ns", [])

    def _hits(pool):
        res = []
        for c in pool:
            st = c.get("search_text") or (
                (c.get("label", "").lower() + " " + c["uri"].split("/")[-1].lower())
            )
            if q in st:
                res.append({"uri": c["uri"], "label": c["label"], "ns": c["ns"]})
                if len(res) >= limit:
                    break
        return res

    scoped = [c for c in idx.get("classes", []) if _in_scope(c["ns"], act)]
    hits = _hits(scoped)
    if not hits and fallback_all:
        hits = _hits(idx.get("classes", []))
    return hits


def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool = False) -> Dict:
    """
    v6-like default: subclass-only (include_properties=False).
    Always ignore blank/restriction nodes by requiring URI-only nodes in the neighborhood.
    """
    idx = _load()
    act = idx.get("active_ns", [])

    def in_scope(u: str) -> bool:
        if not act:
            return True
        for c in idx.get("classes", []):
            if c["uri"] == u:
                return c["ns"] in act
        return True

    # build undirected adjacency over filtered edges
    undirected: Dict[str, set] = {}
    for su, ou in idx.get("edges", []):
        if _is_uri(su) and _is_uri(ou) and in_scope(su) and in_scope(ou):
            undirected.setdefault(su, set()).add(ou)
            undirected.setdefault(ou, set()).add(su)

    if include_properties:
        for dom, rng, _prop in idx.get("prop_edges", []):
            if _is_uri(dom) and _is_uri(rng) and in_scope(dom) and in_scope(rng):
                undirected.setdefault(dom, set()).add(rng)
                undirected.setdefault(rng, set()).add(dom)

    # BFS neighborhood
    seen = {focus_uri}
    frontier = {focus_uri}
    for _ in range(max(0, hops)):
        nxt = set()
        for u in list(frontier):
            for v in undirected.get(u, []):
                if v not in seen:
                    nxt.add(v)
        frontier = nxt
        seen |= frontier

    # Fallback: if too small and scope active, retry without scope limits
    if len(seen) < 10 and act:
        undirected = {}
        for su, ou in idx.get("edges", []):
            if _is_uri(su) and _is_uri(ou):
                undirected.setdefault(su, set()).add(ou)
                undirected.setdefault(ou, set()).add(su)
        if include_properties:
            for dom, rng, _prop in idx.get("prop_edges", []):
                if _is_uri(dom) and _is_uri(rng):
                    undirected.setdefault(dom, set()).add(rng)
                    undirected.setdefault(rng, set()).add(dom)
        seen = {focus_uri}
        frontier = {focus_uri}
        for _ in range(max(0, hops)):
            nxt = set()
            for u in list(frontier):
                for v in undirected.get(u, []):
                    if v not in seen:
                        nxt.add(v)
            frontier = nxt
            seen |= frontier

    # nodes: readable + full labels; only URIs
    lookup = {c["uri"]: c for c in idx.get("classes", [])}
    nodes = []
    for u in seen:
        if not _is_uri(u):
            continue
        full = lookup.get(u, {}).get("label") or u.split("/")[-1]
        short = full if len(full) <= 36 else full[:33] + "…"
        nodes.append({"id": u, "label": short, "full": full})

    # links: keep kind; URI-only
    links = []
    for su, ou in idx.get("edges", []):
        if su in seen and ou in seen and _is_uri(su) and _is_uri(ou):
            links.append({"source": su, "target": ou, "kind": "subClassOf"})
    if include_properties:
        for dom, rng, prop in idx.get("prop_edges", []):
            if dom in seen and rng in seen and _is_uri(dom) and _is_uri(rng):
                links.append({"source": dom, "target": rng, "kind": "property", "label": prop})

    return {"nodes": nodes, "links": links}


def tree_from(focus_uri: str | None, depth: int = 3, scope_only: bool = True) -> Dict:
    idx = _load()
    act = idx.get("active_ns", [])
    def in_scope(ns): return (not act) or (ns in act)
    children = {}
    class_info = {c["uri"]: c for c in idx.get("classes", [])}
    for su, ou in idx.get("edges", []):
        su_ns = class_info.get(su, {}).get("ns", "")
        ou_ns = class_info.get(ou, {}).get("ns", "")
        if scope_only and (not in_scope(su_ns) or not in_scope(ou_ns)):
            continue
        children.setdefault(ou, set()).add(su)
    roots = [focus_uri] if focus_uri else []
    if not roots:
        roots = roots[:5] or [next(iter(class_info.keys()))]
    def build(u, d):
        lab = class_info.get(u, {}).get("label") or u.split("/")[-1]
        node = {"uri": u, "label": lab, "children": []}
        if d <= 0: return node
        for ch in sorted(children.get(u, []),
                         key=lambda x: (class_info.get(x, {}).get("label") or x.split("/")[-1]).lower()):
            node["children"].append(build(ch, d - 1))
        return node
    return {"roots": [build(r, depth) for r in roots]}


def nodeinfo_bulk(uris: list[str]) -> Dict[str, Dict]:
    idx = _load()
    info = {c["uri"]: c for c in idx.get("classes", [])}
    parents = {}
    children = {}
    for su, ou in idx.get("edges", []):
        children.setdefault(ou, set()).add(su)
        parents.setdefault(su, set()).add(ou)
    out = {}
    for u in uris:
        c = info.get(u) or {}
        out[u] = {
            "label": c.get("label") or u.split("/")[-1],
            "ns": c.get("ns"),
            "parents": list(parents.get(u, []))[:8],
            "children": list(children.get(u, []))[:8],
        }
    return out


def schema_for_class(class_uri: str) -> Dict:
    g = Graph(); path = _ttl_path()
    g.parse(path, format=guess_format(str(path)) or "turtle")
    supers = set([URIRef(class_uri)])
    frontier = {URIRef(class_uri)}
    while frontier:
        nxt = set()
        for s in list(frontier):
            for _, _, o in g.triples((s, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    props = set()
    for p, _, dom in g.triples((None, RDFS.domain, None)):
        if dom in supers:
            props.add(p)
    out = []
    for p in props:
        lbl = g.value(p, RDFS.label) or g.value(p, SKOS.prefLabel) or p.split("/")[-1]
        out.append({"property": str(p), "label": str(lbl)})
    out.sort(key=lambda x: x["label"].lower())
    return {"properties": out, "count": len(out)}


def attributes_for_class(class_uri: str) -> Dict:
    """Return attributes with synonyms for matching (labels, altLabels, localName)."""
    path = _ttl_path()
    g = Graph(); g.parse(path, format=guess_format(str(path)) or "turtle")
    supers = set([URIRef(class_uri)])
    frontier = {URIRef(class_uri)}
    while frontier:
        nxt = set()
        for scls in list(frontier):
            for _, _, o in g.triples((scls, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    props = set()
    for piri, _, dom in g.triples((None, RDFS.domain, None)):
        if dom in supers:
            props.add(piri)
    rows = []
    for piri in props:
        labels = set()
        for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
            for _, _, val in g.triples((piri, pred, None)):
                try:
                    labels.add(str(val))
                except Exception:
                    pass
        local = str(piri).split("/")[-1]
        if local:
            labels.add(local)
        rows.append({"property": str(piri), "labels": sorted({l for l in labels if l})})
    rows.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": rows, "count": len(rows)}
