# ui/streamlit_app.py
# Run:  streamlit run ui/streamlit_app.py

import os, io, re, json, tempfile
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import rdflib
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PyMuPDF for PDF rendering & text/word boxes
import fitz  # PyMuPDF


# ------------- Streamlit page setup (must be first call) -------------
st.set_page_config(page_title="LexiGraph", layout="wide")


# ------------- Sidebar controls -------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Rendering & OCR")
    dpi = st.slider("Render DPI", 120, 300, 200, 10)
    use_ocr = st.checkbox("Enable OCR fallback (Paddle/EasyOCR if present)", value=False)
    max_boxes_to_draw = st.slider("Max boxes to draw per page", 100, 5000, 1500, 100)

    st.subheader("Linking")
    autolink_thr = st.slider("Auto‑link min similarity", 0.50, 0.99, 0.80, 0.01)
    subgraph_hops = st.slider("Subgraph hops (for D3 view)", 1, 4, 2, 1)

    st.subheader("Ontology")
    ttl_upload = st.file_uploader("Upload FIBO TTL (overwrites data/fibo_full.ttl)", type=["ttl"])
    if ttl_upload is not None:
        Path("data").mkdir(parents=True, exist_ok=True)
        out = Path("data/fibo_full.ttl")
        out.write_bytes(ttl_upload.read())
        st.success(f"Saved {out}. Click 'Rerun' from Streamlit menu.")


# ------------- Robust FIBO loader -------------
@st.cache_resource
def load_fibo_graph() -> Tuple[Graph, str]:
    """
    Load FIBO from data/fibo_full.ttl (preferred), else data/fibo_trimmed.ttl, else fibo.ttl.
    Tries multiple formats and sanitizes leading junk if needed.
    Returns (graph, status_str).
    """
    from rdflib.util import guess_format

    def _try_parse(path: str) -> Tuple[Graph, str]:
        g = Graph()
        fmt_hint = guess_format(path) or "turtle"
        tried = []
        for fmt in [fmt_hint, "turtle", "xml", "n3", "nt", "trig", "trix", "json-ld"]:
            if fmt in tried:
                continue
            try:
                g.parse(path, format=fmt)
                return g, f"ok:{fmt}"
            except Exception:
                tried.append(fmt)
        raise RuntimeError(f"parse failed (tried={tried})")

    def _sanitize_and_parse(path: str) -> Tuple[Graph, str]:
        raw = Path(path).read_bytes()
        try:
            txt = raw.decode("utf-8", errors="replace")
        except Exception:
            txt = raw.decode("latin1", errors="replace")
        m = re.search(r"(@prefix|<)", txt)
        if m:
            txt = txt[m.start():]
        with tempfile.NamedTemporaryFile("w", suffix=".ttl", delete=False, encoding="utf-8") as t:
            t.write(txt)
            tmp = t.name
        try:
            return _try_parse(tmp)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    candidates = [
        "data/fibo_full.ttl",
        "data/fibo_trimmed.ttl",
        "fibo.ttl",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return _try_parse(p)
            except Exception:
                try:
                    return _sanitize_and_parse(p)
                except Exception as e:
                    return Graph(), f"error: {p}: {e}"
    return Graph(), "missing"


graph, fibo_status = load_fibo_graph()


# ------------- FIBO class index & search -------------
def _labels_for(graph: Graph, node: rdflib.term.Identifier) -> List[str]:
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
    lab = set()
    for pred in (RDFS.label, SKOS.prefLabel, SKOS.altLabel):
        for _, _, v in graph.triples((node, pred, None)):
            try:
                lab.add(str(v))
            except Exception:
                pass
    tail = str(node).split("/")[-1]
    if tail:
        lab.add(tail)
    return sorted(lab)


@st.cache_data(show_spinner=False)
def fibo_index_classes(graph_str: str) -> List[Dict]:
    """
    Build a lightweight class index: uri, label, ns, search_text.
    We pass graph serialized string to make it cache-friendly.
    """
    g = Graph()
    g.parse(data=graph_str, format="turtle") if graph_str.strip().startswith("@prefix") else g.parse(data=graph_str)
    classes = []
    seen = set()
    for ctype in (OWL.Class, RDFS.Class):
        for s, _, _ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen:
                continue
            seen.add(su)
            labs = _labels_for(g, s)
            label = labs[0] if labs else su.split("/")[-1]
            ns = su.rsplit("/", 1)[0] + "/"
            search_text = " ".join(l.lower() for l in labs)
            classes.append({"uri": su, "label": label, "ns": ns, "search_text": search_text})
    return classes


def get_class_index(graph: Graph) -> List[Dict]:
    try:
        # Persist graph in a compact N-Triples string for cache stability
        data = graph.serialize(format="nt")
        return fibo_index_classes(data.decode("utf-8") if isinstance(data, bytes) else data)
    except Exception:
        # Fallback: rebuild directly
        classes = []
        seen = set()
        for ctype in (OWL.Class, RDFS.Class):
            for s, _, _ in graph.triples((None, RDF.type, ctype)):
                su = str(s)
                if su in seen:
                    continue
                seen.add(su)
                labs = _labels_for(graph, s)
                label = labs[0] if labs else su.split("/")[-1]
                ns = su.rsplit("/", 1)[0] + "/"
                search_text = " ".join(l.lower() for l in labs)
                classes.append({"uri": su, "label": label, "ns": ns, "search_text": search_text})
        return classes


# ------------- D3 subgraph renderer -------------
def render_d3_subgraph(graph: Graph, focus_uri: str, hops: int = 2):
    """
    Try backend API /fibo/subgraph first; fallback to local neighborhood.
    """
    import json as _json
    from pathlib import Path as _Path
    api_base = os.environ.get("LEXI_API_BASE", "http://127.0.0.1:8000")
    subg = None
    # Try API
    try:
        import requests
        r = requests.get(f"{api_base}/fibo/subgraph", params={"focus": focus_uri, "hops": hops}, timeout=6)
        r.raise_for_status()
        subg = r.json()
    except Exception:
        pass

    # Fallback local
    if subg is None:
        neighbors = set([focus_uri])
        frontier = set([focus_uri])
        for _ in range(max(0, hops)):
            nxt = set()
            for u in list(frontier):
                uref = rdflib.URIRef(u)
                for _, _, o in graph.triples((uref, RDFS.subClassOf, None)):
                    nxt.add(str(o))
                for s, _, _ in graph.triples((None, RDFS.subClassOf, uref)):
                    nxt.add(str(s))
            frontier = nxt - neighbors
            neighbors |= frontier

        def _lbl(n):
            ls = _labels_for(graph, rdflib.URIRef(n))
            return ls[0] if ls else n.split("/")[-1]

        nodes = [{"id": u, "label": _lbl(u)} for u in neighbors]
        links = []
        for s, o in graph.subject_objects(RDFS.subClassOf):
            su, ou = str(s), str(o)
            if su in neighbors and ou in neighbors:
                links.append({"source": su, "target": ou, "kind": "subClassOf"})
        subg = {"nodes": nodes, "links": links}

    # Inject into component
    html_path = _Path(__file__).resolve().parent.parent / "components" / "fibo_graph.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("{{GRAPH_JSON}}", _json.dumps(subg, ensure_ascii=False))
    html = html.replace("{{API_BASE}}", _json.dumps(api_base))
    st.components.v1.html(html, height=640)


# ------------- SPARQL attribute discovery -------------
def _sparql_props_by_domain(class_uri: str) -> str:
    return f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?p WHERE {{
  VALUES ?c {{ <{class_uri}> }}
  ?c rdfs:subClassOf* ?sc .
  ?p rdfs:domain ?sc .
}}
"""


def _sparql_props_by_restriction(class_uri: str) -> str:
    return f"""
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
SELECT DISTINCT ?p WHERE {{
  VALUES ?c {{ <{class_uri}> }}
  {{ ?c (rdfs:subClassOf|owl:equivalentClass)* ?e . }}
  {{
    ?e a owl:Restriction ; owl:onProperty ?p .
  }}
  UNION
  {{
    ?e owl:intersectionOf ?lst .
    ?lst (rdf:rest*/rdf:first) ?r .
    ?r a owl:Restriction ; owl:onProperty ?p .
  }}
  UNION
  {{
    ?e owl:unionOf ?lst2 .
    ?lst2 (rdf:rest*/rdf:first) ?r2 .
    ?r2 a owl:Restriction ; owl:onProperty ?p .
  }}
}}
"""


def attributes_for_class_sparql(graph: Graph, class_uri: str) -> dict:
    props = set()
    try:
        for row in graph.query(_sparql_props_by_domain(class_uri)):
            props.add(row.p)
    except Exception:
        pass
    try:
        for row in graph.query(_sparql_props_by_restriction(class_uri)):
            props.add(row.p)
    except Exception:
        pass
    rows = [{"property": str(p), "labels": _labels_for(graph, p)} for p in props]
    rows.sort(key=lambda r: (r["labels"][0].lower() if r["labels"] else r["property"]))
    return {"attributes": rows, "count": len(rows)}


# ------------- PDF reading & boxes (no Tesseract) -------------
def _open_pdf(uploaded_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=uploaded_bytes, filetype="pdf")


def _page_to_image(page: fitz.Page, dpi: int) -> Tuple[Image.Image, float, float]:
    # scale factors
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    sx = zoom
    sy = zoom
    return img, sx, sy


def _page_word_boxes(page: fitz.Page) -> List[Tuple[float, float, float, float, str]]:
    """
    Return list of (x0,y0,x1,y1,text) in PDF point coordinates.
    We rely on page.get_text("words") which returns precise word boxes.
    """
    words = page.get_text("words") or []
    # sort top-to-bottom, then left-to-right
    words.sort(key=lambda w: (w[3], w[0]))
    boxes = [(w[0], w[1], w[2], w[3], w[4]) for w in words if len(w) >= 5]
    return boxes


def _extract_text_blocks(doc: fitz.Document) -> Tuple[str, List[Dict]]:
    """
    Aggregate plain text (for similarity/autolink) and per-page boxes for rendering.
    """
    all_text = []
    pages = []
    for pno in range(len(doc)):
        page = doc[pno]
        words = _page_word_boxes(page)
        text = page.get_text() or " ".join([w[-1] for w in words])
        all_text.append(text)
        pages.append({"page": pno + 1, "text": text, "boxes": words})
    return "\n".join(all_text), pages


def draw_boxes(img: Image.Image, boxes: List[Tuple[float, float, float, float, str]],
               sx: float, sy: float, max_boxes: int = 1500) -> Image.Image:
    """
    Draw rectangles for up to max_boxes words. Convert PDF points to pixels via sx/sy.
    """
    vis = img.copy()
    drw = ImageDraw.Draw(vis)
    count = 0
    for (x0, y0, x1, y1, _txt) in boxes:
        if count >= max_boxes:
            break
        # map to pixel space via the same zoom
        rx0, ry0 = x0 * sx, y0 * sy
        rx1, ry1 = x1 * sx, y1 * sy
        # clamp
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(vis.width - 1, rx1), min(vis.height - 1, ry1)
        drw.rectangle([rx0, ry0, rx1, ry1], outline=(0, 180, 255), width=1)
        count += 1
    return vis


# ------------- UI -------------
st.title("📄 LexiGraph — FIBO‑guided Document Understanding")

# FIBO status
with st.expander("FIBO status", expanded=False):
    st.write(f"Graph load: **{fibo_status}**")
    st.write(f"Triples: {len(graph)}")

uploaded = st.file_uploader("Upload a PDF (scanned or digital)", type=["pdf"])
if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

# Read PDF
try:
    doc = _open_pdf(uploaded.read())
except Exception as e:
    st.error(f"Failed to open PDF: {e}")
    st.stop()

doc_text, page_data = _extract_text_blocks(doc)

# Preview first pages with boxes
st.markdown("### 1) Preview & word boxes (no Tesseract)")
num_preview = st.slider("Preview how many pages", 1, min(6, len(page_data)), min(2, len(page_data)))
cols = st.columns(1)
for i in range(num_preview):
    page = doc.load_page(i)
    img, sx, sy = _page_to_image(page, dpi=dpi)
    boxed = draw_boxes(img, page_data[i]["boxes"], sx, sy, max_boxes=max_boxes_to_draw)
    st.image(boxed, caption=f"Page {i+1} — {len(page_data[i]['boxes'])} words (showing ≤ {max_boxes_to_draw})", use_column_width=True)

# ------------- 2) Auto‑link to a FIBO class (override if needed) -------------
st.markdown("---")
st.markdown("### 2) Auto‑link to a FIBO class (override if needed)")

classes = get_class_index(graph)
selected_uri = ""
selected_label = ""
selected_score = 0.0
top_suggestion = None

if not classes:
    st.error("No FIBO classes found. Upload a valid TTL from the sidebar.")
else:
    class_texts = [c["search_text"] for c in classes]
    uris = [c["uri"] for c in classes]
    labels = [c["label"] for c in classes]

    try:
        if doc_text.strip() and len(class_texts) > 0:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
            X = vec.fit_transform(class_texts + [doc_text])
            sims = cosine_similarity(X[-1], X[:-1])[0]
            if sims.size:
                top_i = int(np.argmax(sims))
                top_suggestion = {"uri": uris[top_i], "label": labels[top_i], "score": float(sims[top_i])}
                if top_suggestion["score"] >= autolink_thr:
                    selected_uri = top_suggestion["uri"]
                    selected_label = top_suggestion["label"]
                    selected_score = top_suggestion["score"]
    except Exception as e:
        st.warning(f"Auto‑link failed: {e}")

    if selected_uri:
        st.success(f"Auto‑linked: **{selected_label}** (score {selected_score:.3f} ≥ {autolink_thr:.2f})")
        st.code(selected_uri, language="text")
    elif top_suggestion:
        st.info(
            f"Top candidate (below threshold {autolink_thr:.2f}): "
            f"**{top_suggestion['label']}**  (score: {top_suggestion['score']:.3f})"
        )
        c1, c2 = st.columns(2)
        if c1.button(f"Use ➜ {top_suggestion['label']}", key=f"use_suggest_{top_suggestion['uri']}"):
            selected_uri = top_suggestion["uri"]
            selected_label = top_suggestion["label"]
        if c2.button("Preview graph", key=f"view_suggest_{top_suggestion['uri']}"):
            render_d3_subgraph(graph, top_suggestion["uri"], hops=subgraph_hops)
    else:
        st.caption("No suggestion. Use manual search below.")

    # Manual search (always visible)
    q = st.text_input("Manual search (label / altLabel / local name)", value="")
    if q.strip():
        ql = q.strip().lower()
        hits = [
            {"label": labels[i], "uri": uris[i]}
            for i, txt in enumerate(class_texts)
            if ql in txt
        ][:50]
        if hits:
            st.write("Matches:")
            for h in hits:
                cc = st.columns([0.55, 0.20, 0.25])
                cc[0].markdown(f"**{h['label']}**  \n`{h['uri']}`")
                if cc[1].button("Use ➜", key=f"use_{h['uri']}"):
                    selected_uri = h["uri"]
                    selected_label = h["label"]
                if cc[2].button("View graph", key=f"view_{h['uri']}"):
                    render_d3_subgraph(graph, h["uri"], hops=subgraph_hops)
        else:
            st.caption("No matches.")

# Debug peek
with st.expander("🔎 Debug: property counts for selected class", expanded=False):
    if selected_uri:
        try:
            dbg = attributes_for_class_sparql(graph, selected_uri)
            st.write(f"Candidate properties: {dbg['count']}")
            for r in dbg["attributes"][:12]:
                st.write("•", ", ".join(r["labels"]) or r["property"])
        except Exception as e:
            st.write(f"SPARQL failed: {e}")
    else:
        st.caption("No class selected.")

# ------------- 3) Attribute coverage -------------
st.markdown("---")
st.markdown("### 3) Attribute coverage (from FIBO)")

if selected_uri:
    try:
        attrs = attributes_for_class_sparql(graph, selected_uri)
    except Exception as e:
        st.error(f"Attribute discovery failed: {e}")
        attrs = {"attributes": [], "count": 0}

    # Build simple KV from words: naive header:value harvesting (best‑effort)
    # Note: You likely already have a better extractor; keep this as a fallback demo.
    # We'll look for patterns "Key : Value" in page text.
    kv: Dict[str, str] = {}
    for p in page_data:
        lines = (p["text"] or "").splitlines()
        for ln in lines:
            if ":" in ln:
                k, v = ln.split(":", 1)
                k = k.strip()
                v = v.strip()
                if len(k) > 0 and len(v) > 0 and k.lower() not in kv:
                    kv[k.lower()] = v

    rows = []
    matched = 0
    for a in attrs.get("attributes", []):
        labels_here = [s.lower() for s in a.get("labels", [])]
        match_key = next((l for l in labels_here if l in kv), None)
        val = kv.get(match_key) if match_key else None
        if val is not None:
            matched += 1
        rows.append(
            {
                "property": a.get("property"),
                "labels": ", ".join(a.get("labels", [])),
                "matched_key": match_key,
                "value": val,
            }
        )
    st.write(f"Coverage: **{matched}/{len(rows)}**")
    st.dataframe(rows, use_container_width=True)

    st.download_button(
        "⬇️ Download JSON (class + kv + mappings)",
        data=json.dumps(
            {
                "class_uri": selected_uri,
                "class_label": selected_label,
                "kv": kv,
                "mappings": rows,
            },
            indent=2,
        ),
        file_name=f"{uploaded.name}.lexigraph.json",
        mime="application/json",
    )
else:
    st.info("No class selected yet — pick from auto‑link or manual search above.")
