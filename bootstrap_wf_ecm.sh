#!/usr/bin/env bash
set -euo pipefail

# --- folders ---
mkdir -p data config/ecm_catalog app/core ui

# --- requirements (isolated so it won't collide with your main reqs) ---
cat > requirements-wf.txt <<'REQ'
streamlit==1.36.0
streamlit-drawable-canvas==0.9.3
rdflib==7.0.0
PyMuPDF==1.24.7
scikit-learn==1.4.2
numpy>=1.24,<2.0
Pillow==10.3.0
PyYAML==6.0.1
REQ

# --- WF-ECM overlay ontology (simple, FIBO-aligned where practical) ---
cat > data/wf_ecm.ttl <<'TTL'
@prefix wf:  <https://wells.local/ecm#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# Pragmatic Wells Fargo ECM classes
wf:Invoice             a owl:Class ; rdfs:label "Invoice (ECM)" .
wf:LoanAgreement       a owl:Class ; rdfs:label "Loan Agreement (ECM)" .
wf:LeaseAgreement      a owl:Class ; rdfs:label "Lease Agreement (ECM)" .
wf:TaxLevyNotice       a owl:Class ; rdfs:label "Tax Levy Notice (ECM)" .
wf:FundingDetailSheet  a owl:Class ; rdfs:label "Funding Detail Sheet (ECM)" .
wf:ContinuingGuarantee a owl:Class ; rdfs:label "Continuing Guarantee (ECM)".

# Core ECM properties (you can extend anytime)
wf:hasInvoiceNumber   a owl:DatatypeProperty ; rdfs:label "invoice number"   ; rdfs:domain wf:Invoice ; rdfs:range xsd:string .
wf:hasInvoiceDate     a owl:DatatypeProperty ; rdfs:label "invoice date"     ; rdfs:domain wf:Invoice ; rdfs:range xsd:date .
wf:hasInvoiceTotal    a owl:DatatypeProperty ; rdfs:label "invoice total"    ; rdfs:domain wf:Invoice ; rdfs:range xsd:decimal .

wf:hasBorrower        a owl:ObjectProperty   ; rdfs:label "borrower"         ; rdfs:domain wf:LoanAgreement .
wf:hasPrincipalAmount a owl:DatatypeProperty ; rdfs:label "principal amount" ; rdfs:domain wf:LoanAgreement ; rdfs:range xsd:decimal .
wf:hasInterestRate    a owl:DatatypeProperty ; rdfs:label "interest rate"    ; rdfs:domain wf:LoanAgreement ; rdfs:range xsd:decimal .

wf:hasLessee          a owl:ObjectProperty   ; rdfs:label "lessee"           ; rdfs:domain wf:LeaseAgreement .
wf:hasTermMonths      a owl:DatatypeProperty ; rdfs:label "term (months)"    ; rdfs:domain wf:LeaseAgreement ; rdfs:range xsd:integer .

wf:hasGrandTotal      a owl:DatatypeProperty ; rdfs:label "grand total"      ; rdfs:domain wf:FundingDetailSheet ; rdfs:range xsd:decimal .
TTL

# --- WF mappings & aliases (drives autolink + coverage) ---
cat > config/wf_mappings.yaml <<'YAML'
classes:
  wf:Invoice:
    properties:
      wf:hasInvoiceNumber:
        fibo_equiv: "fibo:invoiceIdentifier"
        aliases: ["invoice number", "invoice no", "invoice#", "inv#", "reference no", "invoice id"]
        type: string
      wf:hasInvoiceDate:
        fibo_equiv: "fibo:dateIssued"
        aliases: ["invoice date", "date of invoice", "issued on", "date:"]
        type: date
      wf:hasInvoiceTotal:
        fibo_equiv: "fibo:totalAmount"
        aliases: ["total due", "amount due", "invoice total", "grand total", "balance due"]
        type: money

  wf:LoanAgreement:
    properties:
      wf:hasBorrower:
        fibo_equiv: "fibo:hasBorrower"
        aliases: ["borrower", "applicant", "customer name", "lessee (loan)"]
        type: party
      wf:hasPrincipalAmount:
        fibo_equiv: "fibo:principalAmount"
        aliases: ["principal", "original amount", "loan amount", "amount financed"]
        type: money
      wf:hasInterestRate:
        fibo_equiv: "fibo:interestRate"
        aliases: ["interest rate", "rate", "apr", "annual percentage rate"]
        type: percent

  wf:LeaseAgreement:
    properties:
      wf:hasLessee:
        fibo_equiv: "fibo:hasLessee"
        aliases: ["lessee", "tenant", "customer name"]
        type: party
      wf:hasTermMonths:
        fibo_equiv: "fibo:term"
        aliases: ["lease term", "term", "months"]
        type: integer

  wf:FundingDetailSheet:
    properties:
      wf:hasGrandTotal:
        fibo_equiv: "fibo:totalAmount"
        aliases: ["grand total", "total funding", "net funded amount", "funding total"]
        type: money
YAML

# --- ECM prompt catalog examples (optional) ---
mkdir -p config/ecm_catalog
cat > config/ecm_catalog/invoice.json <<'JS'
{
  "class": "wf:Invoice",
  "fields": [
    {"name":"wf:hasInvoiceNumber","prompt":"Extract invoice number.","required":true},
    {"name":"wf:hasInvoiceDate","prompt":"Extract invoice date (YYYY-MM-DD).","required":true},
    {"name":"wf:hasInvoiceTotal","prompt":"Extract invoice grand total numeric value.","required":true}
  ]
}
JS

# --- app/core/wf_ecm.py ---
cat > app/core/wf_ecm.py <<'PY'
from rdflib import Graph, RDFS, OWL, URIRef, Namespace
from rdflib.util import guess_format
from pathlib import Path
import yaml, re

WF = Namespace("https://wells.local/ecm#")

DATA = Path("data")
CFG  = Path("config")
WF_TTL = DATA/"wf_ecm.ttl"
WF_MAP = CFG/"wf_mappings.yaml"

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")

class WFOnto:
    def __init__(self):
        if not WF_TTL.exists():
            raise FileNotFoundError("Missing data/wf_ecm.ttl")
        self.g = Graph()
        self.g.parse(WF_TTL, format=guess_format(str(WF_TTL)) or "turtle")
        self.cfg = yaml.safe_load(WF_MAP.read_text()) if WF_MAP.exists() else {"classes":{}}

    def classes(self):
        out = []
        # Any subject that is declared as Class in our namespace
        for s,_,_ in self.g.triples((None, None, None)):
            if isinstance(s, URIRef) and str(s).startswith(str(WF)):
                if (s, None, OWL.Class) in self.g or (s, None, RDFS.Class) in self.g:
                    lbl = self.g.value(s, RDFS.label)
                    out.append({"uri": str(s), "label": str(lbl) if lbl else s.split("#")[-1]})
        uniq = {c["uri"]: c for c in out}
        return list(uniq.values())

    def props_for(self, class_uri: str):
        cu = URIRef(class_uri)
        props=[]
        for p,_,dom in self.g.triples((None, RDFS.domain, None)):
            if dom == cu:
                lbl = self.g.value(p, RDFS.label)
                row = {"property": str(p), "label": (str(lbl) if lbl else str(p).split("#")[-1])}
                # overlay aliases/types from config
                q = self._qname(str(p))
                cls_cfg = self.cfg.get("classes",{}).get(self._qname(class_uri), {})
                prop_cfg = (cls_cfg.get("properties") or {}).get(q, {})
                row["aliases"] = prop_cfg.get("aliases", [])
                row["type"] = prop_cfg.get("type", "string")
                row["fibo_equiv"] = prop_cfg.get("fibo_equiv")
                props.append(row)
        props.sort(key=lambda x: x["label"].lower())
        return props

    def search(self, q: str, limit=20):
        ql = q.strip().lower()
        if not ql: return []
        hits=[]
        for c in self.classes():
            hay = (c["label"]+" "+c["uri"].split("#")[-1]+" "+_CAMEL_RE.sub(" ", c["uri"].split("#")[-1])).lower()
            if ql in hay:
                hits.append(c)
                if len(hits)>=limit: break
        return hits

    def _qname(self, uri: str) -> str:
        if uri.startswith(str(WF)): return "wf:" + uri.split("#")[-1]
        return uri

WF_ONTO = None
def get_wf():
    global WF_ONTO
    if WF_ONTO is None:
        WF_ONTO = WFOnto()
    return WF_ONTO
PY

# --- app/core/wf_vec.py ---
cat > app/core/wf_vec.py <<'PY'
from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, json
from .wf_ecm import get_wf

STORE = Path("data/wf_vec")
STORE.mkdir(parents=True, exist_ok=True)

def build_wf_index() -> Dict:
    wf = get_wf()
    classes = wf.classes()
    recs=[]
    for c in classes:
        props = wf.props_for(c["uri"])
        txt = " ".join([c["label"]] + [p["label"] for p in props] + sum([p.get("aliases",[]) for p in props], []))
        recs.append({"uri": c["uri"], "label": c["label"], "text": txt})
    vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1,2))
    X = vec.fit_transform([r["text"] for r in recs])
    joblib.dump(vec, STORE/"vec.joblib")
    joblib.dump(X, STORE/"X.joblib")
    (STORE/"meta.json").write_text(json.dumps(recs))
    return {"classes": len(recs)}

def _load():
    vec = joblib.load(STORE/"vec.joblib")
    X = joblib.load(STORE/"X.joblib")
    meta = json.loads((STORE/"meta.json").read_text())
    return vec, X, meta

def suggest_class(text: str, top_k=5):
    vec, X, meta = _load()
    q = vec.transform([text or ""])
    sims = cosine_similarity(q, X)[0]
    order = sims.argsort()[::-1][:top_k]
    return [{"uri": meta[i]["uri"], "label": meta[i]["label"], "score": float(sims[i])} for i in order]
PY

# --- ui/streamlit_wf.py (self-contained WF-ECM demo UI) ---
cat > ui/streamlit_wf.py <<'PY'
import io, json, time, base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import numpy as np

from app.core.wf_ecm import get_wf
from app.core.wf_vec import build_wf_index, suggest_class

# ---------- page config FIRST ----------
st.set_page_config(page_title="LexiGraph — WF‑ECM HITL", layout="wide")

# ---------- helpers ----------
def extract_text_blocks(pdf_bytes: bytes) -> Dict[int, List[Dict]]:
    """Return per-page words with bounding boxes using PyMuPDF (no Tesseract)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    for i, page in enumerate(doc):
        words = page.get_text("words")  # list of (x0,y0,x1,y1, word, block, line, word_no)
        rows = []
        for (x0, y0, x1, y1, w, b, l, wn) in words:
            rows.append({"text": w, "bbox": [float(x0), float(y0), float(x1), float(y1)], "block": int(b), "line": int(l)})
        pages[i] = rows
    return pages

def render_page_image(pdf_bytes: bytes, page_no: int, zoom: float=2.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_no]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def draw_boxes(img: Image.Image, boxes: List[Tuple[int,int,int,int]], fill=None, outline=(255,0,0), width=2) -> Image.Image:
    im = img.copy()
    dr = ImageDraw.Draw(im, "RGBA")
    for (x0,y0,x1,y1) in boxes:
        if fill is not None:
            dr.rectangle([x0,y0,x1,y1], fill=fill, outline=outline, width=width)
        else:
            dr.rectangle([x0,y0,x1,y1], outline=outline, width=width)
    return im

def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter = max(0, min(ax1,bx1)-max(ax0,bx0)) * max(0, min(ay1,by1)-max(ay0,by0))
    if inter == 0: return 0.0
    area = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter/area if area>0 else 0.0

def fuzzy_match_candidates(words: List[Dict], alias_tokens: List[str]) -> List[Dict]:
    # naive window match: find lines containing any alias tokens; score by count and token distance
    hits=[]
    alow = [a.lower() for a in alias_tokens]
    # group by line id
    by_line = {}
    for w in words:
        by_line.setdefault(w["line"], []).append(w)
    for line_id, ws in by_line.items():
        line_text = " ".join([w["text"] for w in ws]).lower()
        score = sum(1 for a in alow if a in line_text)
        if score>0:
            # bbox covering the line
            x0 = min(w["bbox"][0] for w in ws); y0=min(w["bbox"][1] for w in ws)
            x1 = max(w["bbox"][2] for w in ws); y1=max(w["bbox"][3] for w in ws)
            hits.append({"line": line_id, "score": score, "bbox":[x0,y0,x1,y1], "text": line_text})
    hits.sort(key=lambda r: r["score"], reverse=True)
    return hits[:5]

def canvas_to_pdf_coords(canvas_box, img_w, img_h, pdf_w, pdf_h):
    # canvas returns left/top/width/height on image; convert back to PDF coords (assuming same aspect)
    x0 = canvas_box["left"] * (pdf_w / img_w)
    y0 = canvas_box["top"]  * (pdf_h / img_h)
    x1 = (canvas_box["left"]+canvas_box["width"])  * (pdf_w / img_w)
    y1 = (canvas_box["top"] +canvas_box["height"]) * (pdf_h / img_h)
    return [x0,y0,x1,y1]

# ---------- sidebar ----------
st.sidebar.title("WF‑ECM")
if st.sidebar.button("Rebuild WF vector index"):
    info = build_wf_index()
    st.sidebar.success(f"WF classes indexed: {info['classes']}")

st.sidebar.markdown("---")
st.sidebar.caption("Load optional ECM prompt (demo)")
demo_choice = st.sidebar.selectbox("ECM Catalog", ["(none)", "Invoice"], index=0)
ecm_schema = None
if demo_choice == "Invoice":
    ecm_schema = json.loads(open("config/ecm_catalog/invoice.json").read())

# ---------- main ----------
st.title("LexiGraph — WF‑ECM HITL demo")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if "state" not in st.session_state:
    st.session_state["state"] = {}

if uploaded:
    pdf_bytes = uploaded.read()
    st.session_state["state"]["pdf_name"] = uploaded.name
    st.session_state["state"]["pdf"] = pdf_bytes

    # OCR words (digital PDF → positions)
    with st.spinner("Indexing PDF…"):
        pages_words = extract_text_blocks(pdf_bytes)
    st.success(f"Indexed {len(pages_words)} pages")

    # Suggest WF class from page-1 summary
    page0_text = " ".join([w["text"] for w in pages_words.get(0, [])])[:5000]
    try:
        hits = suggest_class(page0_text, top_k=5)
    except Exception as e:
        hits = []
        st.warning(f"WF vector index missing or empty. Click 'Rebuild WF vector index' in sidebar. ({e})")

    colA, colB = st.columns([1,1])
    with colA:
        st.subheader("1) Pick WF‑ECM class")
        st.write("Suggestions:", hits if hits else "—")
        wf = get_wf()
        q = st.text_input("Search WF‑ECM classes")
        if q:
            st.write(wf.search(q, limit=20))
        chosen_uri = st.selectbox(
            "Choose WF‑ECM class",
            options=[h["uri"] for h in hits] if hits else [c["uri"] for c in wf.classes()],
            format_func=lambda u: next((h["label"] for h in hits if h.get("uri")==u), u.split("#")[-1]) if hits else u.split("#")[-1]
        )
        props = wf.props_for(chosen_uri) if chosen_uri else []
        st.session_state["state"]["wf_class"] = chosen_uri
        st.session_state["state"]["wf_props"] = props
        st.caption(f"{len(props)} properties")

        if ecm_schema and ecm_schema.get("class")==chosen_uri:
            st.info("Loaded ECM catalog for this class.")
            # show required fields from catalog
            reqs = [f["name"] for f in ecm_schema.get("fields",[])]
            st.write("Required:", reqs)

    with colB:
        st.subheader("2) Property coverage & autolink")
        results = st.session_state["state"].setdefault("results", {})
        page_no = st.number_input("Page", min_value=0, max_value=max(pages_words.keys()), value=0, step=1)
        img = render_page_image(pdf_bytes, page_no, zoom=2.0)
        img_w, img_h = img.size

        # autolink each property using alias windows
        words = pages_words.get(page_no, [])
        auto_boxes=[]
        for p in props:
            aliases = p.get("aliases") or [p["label"]]
            cands = fuzzy_match_candidates(words, aliases)
            if cands:
                bb = cands[0]["bbox"]
                results[p["property"]] = {"value": None, "page": int(page_no), "bbox": bb, "source":"auto", "text": cands[0]["text"]}
                auto_boxes.append(tuple(int(x*2.0) for x in bb))  # scaled to image (zoom=2)

        img_auto = draw_boxes(img, auto_boxes, outline=(0, 153, 255))
        st.image(img_auto, caption="Auto suggestions (blue boxes)")

        st.markdown("**Manual lasso (click and drag)** — select a region to bind to a property")
        canvas = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=img_auto,
            update_streamlit=True,
            height=img_h,
            width=img_w,
            drawing_mode="rect",
            key=f"canvas_{page_no}"
        )

        # When user draws a rectangle, bind it to a chosen property
        prop_names = [p["property"] for p in props]
        target_prop = st.selectbox("Bind lasso to property", options=prop_names) if prop_names else None
        if target_prop and canvas.json_data and len(canvas.json_data.get("objects",[]))>0:
            last = canvas.json_data["objects"][-1]
            if last.get("type")=="rect":
                pdf_w = img_w/2.0; pdf_h=img_h/2.0  # inverse of zoom=2
                bbox_pdf = canvas_to_pdf_coords(last, img_w, img_h, pdf_w, pdf_h)
                results[target_prop] = {"value": None, "page": int(page_no), "bbox": bbox_pdf, "source":"lasso", "text": ""}
                st.success(f"Bound lasso to {target_prop}")

        # Allow value entry per bound property
        st.markdown("**Captured fields**")
        for p in props:
            row = results.get(p["property"])
            if not row: continue
            row["value"] = st.text_input(p["property"], row.get("value") or "", key=f"val_{p['property']}")
        st.session_state["state"]["results"] = results

    # Coverage
    st.subheader("3) Coverage")
    bound = sum(1 for p in props if p["property"] in results)
    st.write(f"{bound} / {len(props)} properties linked")
    if st.button("Export result.json"):
        payload = {
            "doc": st.session_state["state"]["pdf_name"],
            "class": st.session_state["state"]["wf_class"],
            "results": st.session_state["state"]["results"]
        }
        st.download_button("Download result.json", data=json.dumps(payload, indent=2), file_name="result.json", mime="application/json")

st.caption("Tip: Rebuild WF index (sidebar) if suggestions show empty.")
PY

# --- README (short) ---
cat > README-WF.md <<'MD'
# LexiGraph — WF‑ECM overlay (HITL demo)

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-wf.txt
# Optional: put your full FIBO at data/fibo_full.ttl (not required for WF overlay)
python -c "from app.core.wf_vec import build_wf_index; print(build_wf_index())"
streamlit run ui/streamlit_wf.py
