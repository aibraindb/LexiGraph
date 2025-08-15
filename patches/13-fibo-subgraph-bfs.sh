#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path, re
p = Path("app/core/fibo_index.py")
s = p.read_text()

# Upgrade subgraph_scoped to traverse subclass + property edges in both directions,
# and fallback to ALL namespaces if result is too small (< 10 nodes).
head = "def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool = True) -> Dict:"
if head not in s:
    # rename existing signature if needed
    s = s.replace("def subgraph_scoped(focus_uri: str, hops: int = 2) -> Dict:",
                  "def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool = True) -> Dict:")

block_start = s.find("def subgraph_scoped(")
block_end = s.find("\ndef ", block_start+1)
if block_end == -1: block_end = len(s)

impl = '''
def subgraph_scoped(focus_uri: str, hops: int = 2, include_properties: bool = True) -> Dict:
    idx = _load()
    act = idx.get("active_ns", [])

    def in_scope(u: str) -> bool:
        if not act: return True
        # look up class ns
        # when class not indexed, allow
        for c in idx.get("classes", []):
            if c["uri"] == u:
                return c["ns"] in act
        return True

    # Build adjacency over BOTH subclass edges and property edges
    undirected = {}
    for su, ou in idx.get("edges", []):
        if in_scope(su) and in_scope(ou):
            undirected.setdefault(su, set()).add(ou)
            undirected.setdefault(ou, set()).add(su)
    if include_properties:
        for dom, rng, _prop in idx.get("prop_edges", []):
            if in_scope(dom) and in_scope(rng):
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

    # Fallback to ALL namespaces if graph is too small
    if len(seen) < 10 and act:
        # temporarily ignore scope
        undirected = {}
        for su, ou in idx.get("edges", []):
            undirected.setdefault(su, set()).add(ou)
            undirected.setdefault(ou, set()).add(su)
        if include_properties:
            for dom, rng, _prop in idx.get("prop_edges", []):
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

    # Build nodes/links (preserve kind/labels on property edges)
    lookup = {c["uri"]: c for c in idx.get("classes", [])}
    nodes = [{"id": u, "label": lookup.get(u, {}).get("label", u.split("/")[-1])} for u in seen]

    links = []
    for su, ou in idx.get("edges", []):
        if su in seen and ou in seen:
            links.append({"source": su, "target": ou, "kind": "subClassOf"})
    if include_properties:
        for dom, rng, prop in idx.get("prop_edges", []):
            if dom in seen and rng in seen:
                links.append({"source": dom, "target": rng, "kind": "property", "label": prop})
    return {"nodes": nodes, "links": links}
'''
s = s[:block_start] + impl + s[block_end:]
Path(p).write_text(s)
print("Patched:", p)
PY
