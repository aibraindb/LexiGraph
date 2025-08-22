from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json, re
from rdflib import Graph, URIRef, RDF, RDFS
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

DATA = Path("data")
FULL_TTL   = DATA/"fibo_full.ttl"
SMALL_TTL  = DATA/"fibo.ttl"
INDEX_JSON = DATA/"fibo_index.json"

PARSER_ORDER = ["turtle","xml","n3","nt","trig","trix","json-ld"]
CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def _ttl_path() -> Path:
    if FULL_TTL.exists(): return FULL_TTL
    if SMALL_TTL.exists(): return SMALL_TTL
    raise FileNotFoundError("No FIBO TTL found. Place fibo_full.ttl or fibo.ttl in data/")

def _parse_graph(path: Path) -> Graph:
    g = Graph()
    fmt = guess_format(str(path)) or "turtle"
    tried=[]
    for f in [fmt]+[x for x in PARSER_ORDER if x!=fmt]:
        try:
            g.parse(path, format=f)
            return g
        except Exception:
            tried.append(f)
    raise RuntimeError(f"Failed to parse {path}; tried: {tried}")

def _label(g: Graph, u: URIRef) -> str:
    v = g.value(u, RDFS.label) or g.value(u, SKOS.prefLabel)
    return str(v) if v else str(u).split("/")[-1]

def _search_text(g: Graph, u: URIRef) -> str:
    parts=[]
    for p in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _,_,o in g.triples((u,p,None)):
            try: parts.append(str(o))
            except: pass
    tail=str(u).split("/")[-1]
    if tail:
        parts.append(tail)
        parts.append(CAMEL_RE.sub(" ", tail))
    # dedupe, lower
    seen=[]
    for t in parts:
        t=t.strip().lower()
        if t and t not in seen: seen.append(t)
    return " ".join(seen)

def _ns_of(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    return m.group(1) if m else uri.rsplit("/",1)[0]+"/"

def build_index(force: bool=False) -> Dict[str, Any]:
    if INDEX_JSON.exists() and not force:
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    path = _ttl_path()
    g = _parse_graph(path)

    classes=[]
    edges=[]
    prop_edges=[]
    ns_counts={}
    seen=set()
    for ctype in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            su=str(s)
            if su in seen: continue
            seen.add(su)
            lbl=_label(g,s)
            ns=_ns_of(su)
            st=_search_text(g,s)
            ns_counts[ns]=ns_counts.get(ns,0)+1
            classes.append({"uri":su,"label":lbl,"ns":ns,"search_text":st})

    for s,o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])

    for p,_,_ in g.triples((None, RDF.type, OWL.ObjectProperty)):
        dom = g.value(p, RDFS.domain)
        rng = g.value(p, RDFS.range)
        if dom and rng:
            prop_edges.append([str(dom), str(rng), str(p)])

    idx={"source":str(path),
         "classes":classes,
         "edges":edges,
         "prop_edges":prop_edges,
         "namespaces":[{"ns":ns,"count":ns_counts[ns]} for ns in sorted(ns_counts.keys())],
         "active_ns":[]}
    INDEX_JSON.parent.mkdir(parents=True,exist_ok=True)
    INDEX_JSON.write_text(json.dumps(idx))
    return idx

def _load() -> Dict[str, Any]:
    if INDEX_JSON.exists():
        try: return json.loads(INDEX_JSON.read_text())
        except Exception: pass
    return build_index(force=True)

def search_scoped(q: str, limit: int=25, fallback_all: bool=True) -> List[Dict[str,str]]:
    q=(q or "").strip().lower()
    if not q: return []
    idx=_load()
    def _pool(all_classes):  # rank by substring match quality
        hits=[]
        for c in all_classes:
            st=c.get("search_text") or (c.get("label","").lower()+" "+c["uri"].split("/")[-1].lower())
            if q in st:
                hits.append({"uri":c["uri"],"label":c["label"],"ns":c["ns"]})
                if len(hits)>=limit: break
        return hits
    # If you later add active namespaces, filter here; for now just all
    hits=_pool(idx.get("classes",[]))
    return hits

def subgraph_scoped(focus_uri: str, hops: int=2, include_props: bool=True) -> Dict[str, Any]:
    idx=_load()
    und={}
    for su,ou in idx.get("edges",[]):
        und.setdefault(su,set()).add(ou)
        und.setdefault(ou,set()).add(su)
    if include_props:
        for dom,rng,_ in idx.get("prop_edges",[]):
            und.setdefault(dom,set()).add(rng)
            und.setdefault(rng,set()).add(dom)
    seen={focus_uri}
    frontier={focus_uri}
    for _ in range(max(0,hops)):
        nxt=set()
        for u in list(frontier):
            for v in und.get(u,[]):
                if v not in seen: nxt.add(v)
        frontier=nxt; seen|=frontier
    lookup={c["uri"]:c for c in idx.get("classes",[])}
    nodes=[{"id":u,"label": lookup.get(u,{}).get("label", u.split("/")[-1])} for u in seen]
    links=[]
    for su,ou in idx.get("edges",[]):
        if su in seen and ou in seen:
            links.append({"source":su,"target":ou,"kind":"subClassOf"})
    if include_props:
        for dom,rng,prop in idx.get("prop_edges",[]):
            if dom in seen and rng in seen:
                links.append({"source":dom,"target":rng,"kind":"property","label":prop})
    return {"nodes":nodes,"links":links}

if __name__=="__main__":
    import argparse, json as _json
    ap=argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args=ap.parse_args()
    idx=build_index(force=args.rebuild)
    print(_json.dumps({"classes":len(idx.get("classes",[])), "edges":len(idx.get("edges",[]))}))
