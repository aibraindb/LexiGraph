#!/usr/bin/env python3
"""
Build or augment data/fibo_full.ttl from a local directory of FIBO modules.
- Merges all TTL/OWL/RDF/JSON-LD files recursively
- Deduplicates triples
- Optional namespace include/exclude regex filters
- Can augment an existing data/fibo_full.ttl
- Emits stats usable by /health

Usage examples:

  # Merge everything under ~/git/fibo into data/fibo_full.ttl
  python scripts/fibo_augment.py --src ~/git/fibo --out data/fibo_full.ttl

  # Augment existing fibo_full.ttl with a new folder
  python scripts/fibo_augment.py --src ~/fibo-addons --out data/fibo_full.ttl --augment

  # Only keep FIBO ontology IRIs (recommended)
  python scripts/fibo_augment.py --src ~/git/fibo --out data/fibo_full.ttl \
      --include 'spec\\.edmcouncil\\.org/fibo/ontology'
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import Iterable, Tuple, Optional, Set

from rdflib import Graph, ConjunctiveGraph, URIRef, RDF, RDFS
from rdflib.namespace import OWL, SKOS, Namespace
from rdflib.util import guess_format

PARSER_TRY = ["turtle", "xml", "n3", "nt", "trig", "trix", "json-ld"]
FIBO = Namespace("https://spec.edmcouncil.org/fibo/ontology/")

def iter_source_files(root: Path) -> Iterable[Path]:
    exts = {".ttl", ".rdf", ".owl", ".n3", ".nt", ".trig", ".trix", ".jsonld", ".json"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def parse_any(g: Graph, path: Path) -> int:
    fmt = guess_format(str(path)) or "turtle"
    tried = []
    for f in [fmt] + [x for x in PARSER_TRY if x != fmt]:
        try:
            before = len(g)
            g.parse(path, format=f)
            return len(g) - before
        except Exception:
            tried.append(f)
    raise RuntimeError(f"Failed to parse {path} with tried formats={tried}")

def filter_graph(
    g: Graph,
    include: Optional[re.Pattern] = None,
    exclude: Optional[re.Pattern] = None
) -> Graph:
    if include is None and exclude is None:
        return g
    out = Graph()
    # carry prefixes
    for p, ns in g.namespaces():
        try: out.bind(p, ns)
        except Exception: pass
    for s,p,o in g:
        keep = True
        def m(x):
            return isinstance(x, URIRef) and str(x)
        if include:
            keep = any(include.search(v) for v in (m(s), m(p), m(o)) if v)
        if keep and exclude:
            if any(exclude.search(v) for v in (m(s), m(p), m(o)) if v):
                keep = False
        if keep:
            out.add((s,p,o))
    return out

def add_prefixes(g: Graph):
    # Safe common prefixes; rdflib will keep existing where possible
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("fibo", FIBO)

def collect_stats(g: Graph) -> dict:
    cls = set(s for s,_,_ in g.triples((None, RDF.type, OWL.Class))) | \
          set(s for s,_,_ in g.triples((None, RDF.type, RDFS.Class)))
    obj_props = set(s for s,_,_ in g.triples((None, RDF.type, OWL.ObjectProperty)))
    dt_props  = set(s for s,_,_ in g.triples((None, RDF.type, OWL.DatatypeProperty)))
    nspaces: Set[str] = set()
    for s,p,o in g:
        for x in (s,p,o):
            if isinstance(x, URIRef):
                # heuristic namespace split
                nspaces.add(str(x).rsplit("/", 1)[0] + "/")
    return {
        "triples": len(g),
        "classes": len(cls),
        "object_properties": len(obj_props),
        "datatype_properties": len(dt_props),
        "namespaces": len(nspaces)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Folder containing FIBO modules (ttl/owl/rdf/jsonld)")
    ap.add_argument("--out", type=str, default="data/fibo_full.ttl", help="Output Turtle")
    ap.add_argument("--augment", action="store_true", help="Augment existing OUT file (merge + dedupe)")
    ap.add_argument("--include", type=str, default=r"spec\.edmcouncil\.org/fibo/ontology", help="Regex include filter")
    ap.add_argument("--exclude", type=str, default=None, help="Regex exclude filter")
    ap.add_argument("--dryrun", action="store_true", help="Parse and report only, do not write OUT")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser()

    if not src.exists():
        print(f"[ERROR] Source folder not found: {src}", file=sys.stderr)
        sys.exit(2)

    include_re = re.compile(args.include) if args.include else None
    exclude_re = re.compile(args.exclude) if args.exclude else None

    g = Graph()
    add_prefixes(g)

    if args.augment and out.exists():
        try:
            print(f"[INFO] Loading existing OUT for augmentation: {out}")
            parse_any(g, out)
        except Exception as e:
            print(f"[WARN] Could not parse existing OUT ({e}); will rebuild from SRC only")

    files = list(iter_source_files(src))
    if not files:
        print(f"[WARN] No RDF files found under {src}")
    total_added = 0
    for i, f in enumerate(files, 1):
        try:
            added = parse_any(g, f)
            total_added += max(0, added)
            if i % 25 == 0:
                print(f"  … parsed {i}/{len(files)} files, graph size={len(g)}")
        except Exception as e:
            print(f"[WARN] Failed to parse {f}: {e}")

    if include_re or exclude_re:
        print("[INFO] Applying namespace filters")
        g = filter_graph(g, include_re, exclude_re)

    add_prefixes(g)  # ensure common prefixes bound
    stats = collect_stats(g)
    print(f"[STATS] Triples={stats['triples']}  Classes={stats['classes']}  "
          f"ObjProps={stats['object_properties']}  DtProps={stats['datatype_properties']}  "
          f"Namespaces={stats['namespaces']}")

    if args.dryrun:
        print("[DRYRUN] Not writing output.")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    ttl = g.serialize(format="turtle")
    out.write_text(ttl)
    print(f"[OK] Wrote {out} ({len(ttl)} bytes)")

if __name__ == "__main__":
    main()
