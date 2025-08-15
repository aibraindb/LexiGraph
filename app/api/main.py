
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import json, uuid

from app.core.embeddings import SimpleVectorStore
from app.core.utils import read_text_from_upload
from app.core import fibo_index
from app.core.attr_extract import extract_by_schema as _extract_by_schema
from app.core.attr_extract import extract_by_schema as _extract_by_schema

app = FastAPI(title="LexiGraph v9")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DOCS_DIR = Path("data/docs"); DOCS_DIR.mkdir(parents=True, exist_ok=True)
RDF_DIR  = Path("data/rdf");  RDF_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR  = Path("data/raw");  RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR= Path("data/index");INDEX_DIR.mkdir(parents=True, exist_ok=True)
ALL_TTL  = RDF_DIR / "all.ttl"
SAMPLES_DIR = Path("data/samples"); SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
UI_SEL   = Path("data/ui_selection.json")

# Init
fibo_index.build_index(force=False)
store = SimpleVectorStore(str(INDEX_DIR))

@app.get("/health")
def health():
    return {"status":"ok", **fibo_index.get_health()}

@app.post("/fibo/reindex")
def fibo_reindex(force: bool=True):
    idx = fibo_index.build_index(force=force)
    return {"status":"reindexed", "num_classes": len(idx.get("classes",[])), "num_namespaces": len(idx.get("namespaces",[]))}

class ScopeReq(BaseModel):
    namespaces: list[str] = []

@app.post("/fibo/scope")
def fibo_set_scope(req: ScopeReq):
    return fibo_index.set_scope(req.namespaces)

@app.get("/fibo/namespaces")
def fibo_namespaces():
    return fibo_index.get_namespaces()

@app.get("/fibo/classes")
def fibo_classes():
    return fibo_index.get_scoped_classes()

@app.get("/fibo/search")
def fibo_search(q: str, limit: int = 25, fallback_all: bool = True):
    return fibo_index.search_scoped(q, limit=limit, fallback_all=fallback_all)

@app.get("/fibo/subgraph")
def fibo_subgraph(focus: str, hops: int = 2):
    return fibo_index.subgraph_scoped(focus, hops=hops)

@app.get("/fibo/tree")
def fibo_tree(focus: str | None = None, depth: int = 3, scope_only: bool = True):
    return fibo_index.tree_from(focus, depth=depth, scope_only=scope_only)

class NodeInfoReq(BaseModel):
    uris: list[str]
@app.post("/fibo/nodeinfo_bulk")
def fibo_nodeinfo(req: NodeInfoReq):
    return fibo_index.nodeinfo_bulk(req.uris or [])

@app.get("/fibo/schema")
def fibo_schema(class_uri: str):
    return fibo_index.schema_for_class(class_uri)

@app.get("/fibo/attributes")
def fibo_attributes(class_uri: str):
    return fibo_index.attributes_for_class(class_uri)

def fibo_schema(class_uri: str):
    return fibo_index.schema_for_class(class_uri)

# Upload & Embeddings
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    text = read_text_from_upload(file.filename, data)
    if not text.strip():
        raise HTTPException(422, "empty_text_extracted: likely scanned; OCR not in this build")
    doc_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower() or ".bin"
    raw_path = RAW_DIR / f"{doc_id}{ext}"
    raw_path.write_bytes(data)
    rec = {"doc_id": doc_id, "filename": file.filename, "text": text, "fibo_class_uri": None, "raw_path": str(raw_path)}
    (DOCS_DIR/f"{doc_id}.json").write_text(json.dumps(rec, indent=2))
    store.add(text=text, doc_id=doc_id, fibo_class_uri=None)
    nbrs = store.search(text, topk=8)
    return {"doc_id": doc_id, "neighbors": nbrs}

@app.get("/docs/raw/{doc_id}")
def docs_raw(doc_id: str):
    dj = DOCS_DIR / f"{doc_id}.json"
    if not dj.exists(): raise HTTPException(404, "doc_id_not_found")
    meta = json.loads(dj.read_text())
    raw_path = meta.get("raw_path")
    if not raw_path or not Path(raw_path).exists():
        raise HTTPException(404, "raw_file_not_found")
    return FileResponse(path=raw_path, filename=Path(raw_path).name)

@app.get("/docs/list")
def docs_list():
    out = []
    for p in DOCS_DIR.glob("*.json"):
        out.append(json.loads(p.read_text()))
    return out

@app.get("/neighbors")
def neighbors(doc_id: str, topk: int=8):
    p = DOCS_DIR / f"{doc_id}.json"
    if not p.exists(): raise HTTPException(404, f"doc_id_not_found: {doc_id}")
    doc = json.loads(p.read_text())
    return {"doc_id": doc_id, "neighbors": store.search(doc["text"], topk=topk)}

# Linking / RDF
class LinkReq(BaseModel):
    doc_id: str
    fibo_class_uri: str

def _rdf_type_turtle(doc_id: str, class_uri: str) -> str:
    return f"<http://example.org/doc/{doc_id}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <{class_uri}> .\n"

@app.post("/label/link")
def link(req: LinkReq):
    dj = DOCS_DIR/f"{req.doc_id}.json"
    if not dj.exists(): raise HTTPException(404, f"doc_id_not_found: {req.doc_id}")
    doc = json.loads(dj.read_text())
    doc["fibo_class_uri"] = req.fibo_class_uri
    dj.write_text(json.dumps(doc, indent=2))
    store.update_label(req.doc_id, req.fibo_class_uri)
    ttl = _rdf_type_turtle(req.doc_id, req.fibo_class_uri)
    (RDF_DIR/f"{req.doc_id}.ttl").write_text(ttl)
    all_ttl = RDF_DIR / "all.ttl"
    if all_ttl.exists(): all_ttl.write_text(all_ttl.read_text() + "\n" + ttl)
    else: all_ttl.write_text(ttl)
    return {"status":"ok","doc_id":req.doc_id,"fibo_class_uri":req.fibo_class_uri,"rdf_turtle":ttl}

@app.get("/rdf/{doc_id}")
def get_rdf(doc_id: str):
    p = RDF_DIR/f"{doc_id}.ttl"
    if not p.exists(): raise HTTPException(404, "rdf_not_found")
    return {"doc_id": doc_id, "turtle": p.read_text()}

# UI selection bridge
@app.post("/ui/set_selection")
def ui_set_selection(uri: str = Query(...)):
    Path("data/ui_selection.json").write_text(json.dumps({"uri": uri})); return {"status":"ok","uri":uri}

@app.get("/ui/get_selection")
def ui_get_selection():
    p = Path("data/ui_selection.json")
    try: return json.loads(p.read_text())
    except Exception: return {"uri": None}

# Suggest class
@app.post("/suggest/class")
def suggest(doc_id: str = Query(...)):
    return store.suggest_class(doc_id=doc_id, topk_neighbors=8)

# Context Association
CA_STATE = {}
class CAReq(BaseModel):
    case_id: str
    product_id: str
    customer_id: str

@app.post("/ca/associate")
def ca_associate(req: CAReq):
    expected = ["w2","bank_statement","lease_agreement","loan_agreement","invoice","funding_detail_sheet","nominal_lease_agreement"]
    CA_STATE[req.case_id] = {"case_id": req.case_id, "product_id": req.product_id, "customer_id": req.customer_id,
                             "expected": expected, "present": [], "missing": expected[:]}
    return CA_STATE[req.case_id]

@app.post("/ca/mark_present")
def ca_mark_present(case_id: str = Query(...), doc_type: str = Query(...)):
    st = CA_STATE.get(case_id)
    if not st:
        return {"status": "no_case"}
    if doc_type not in st["present"]:
        st["present"].append(doc_type)
    if doc_type in st["missing"]:
        st["missing"].remove(doc_type)
    return st

# Email Simulator
EMAILS = [
  {
    "id": "EML-LOAN-001",
    "subject": "Loan options and required docs",
    "body": "Hi, attaching my W-2, 401k statement and term deposit certificate for the loan application.",
    "suggested_types": ["loan_agreement","w2","retirement_account_statement","term_deposit_certificate"],
    "sample_dirs": ["LoanAgreement","FundingDetailSheet","Invoice"]
  },
  {
    "id": "EML-LEASE-001",
    "subject": "Lease documents for review",
    "body": "Sharing the signed lease agreement. Let me know if bank statements are needed.",
    "suggested_types": ["lease_agreement","bank_statement"],
    "sample_dirs": ["LeaseAgreement","NominalLeaseAgreement"]
  }
]

@app.get("/demo/threads")
def demo_threads():
    return EMAILS

class PrimeReq(BaseModel):
    case_id: str
    email_id: str

def _first_pdf_in(dir_name: str):
    d = SAMPLES_DIR/dir_name
    if not d.exists(): return None
    for p in sorted(d.glob("*.pdf")):
        return p
    return None

@app.post("/demo/prime")
def demo_prime(req: PrimeReq):
    doc_id = str(uuid.uuid4())[:8]
    body = next((e["body"] for e in EMAILS if e["id"]==req.email_id), "")
    rec = {"doc_id": doc_id, "filename": f"{req.email_id}.eml.txt", "text": body, "fibo_class_uri": None, "source":"email"}
    (DOCS_DIR/f"{doc_id}.json").write_text(json.dumps(rec, indent=2))
    store.add(text=body, doc_id=doc_id, fibo_class_uri=None)
    return {"primed_doc_id": doc_id}

@app.post("/demo/ingest")
def demo_ingest(req: PrimeReq):
    email = next((e for e in EMAILS if e["id"]==req.email_id), None)
    if not email: raise HTTPException(404, "email_not_found")
    uploaded = []
    for folder in email.get("sample_dirs", []):
        pdf = _first_pdf_in(folder)
        if not pdf: continue
        data = pdf.read_bytes()
        text = read_text_from_upload(pdf.name, data)
        doc_id = str(uuid.uuid4())[:8]
        ext = pdf.suffix.lower()
        raw_path = RAW_DIR / f"{doc_id}{ext}"
        raw_path.write_bytes(data)
        rec = {"doc_id": doc_id, "filename": pdf.name, "text": text, "fibo_class_uri": None, "raw_path": str(raw_path)}
        (DOCS_DIR/f"{doc_id}.json").write_text(json.dumps(rec, indent=2))
        store.add(text=text, doc_id=doc_id, fibo_class_uri=None)
        uploaded.append({"doc_id": doc_id, "filename": pdf.name})
    st = CA_STATE.get(req.case_id)
    if st:
        for t in email.get("suggested_types", []):
            if t not in st["present"]:
                st["present"].append(t)
            if t in st["missing"]:
                st["missing"].remove(t)
    return {"uploaded": uploaded, "ca": st or {}}

# Simple extraction + coverage
@app.post("/extract/simple")
def extract_simple(doc_id: str = Query(...), class_uri: str = Query(...)):
    p = DOCS_DIR / f"{doc_id}.json"
    if not p.exists(): raise HTTPException(404, "doc_id_not_found")
    meta = json.loads(p.read_text()); text = meta.get("text","")
    import re
    amounts = re.findall(r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b", text)[:10]
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)[:10]
    ids = re.findall(r"\b[A-Z]{2,5}-\d{3,}\b|\b\d{9,}\b", text)[:10]
    extracted = {"amounts": amounts, "dates": dates, "ids": ids}

    sch = fibo_index.schema_for_class(class_uri)
    labels = [p["label"].lower() for p in sch.get("properties", [])]
    tlow = text.lower()
    hit = 0
    for lab in labels:
        tokens = [t for t in re.split(r"[^a-z0-9]+", lab) if t]
        if tokens and any(tok in tlow for tok in tokens):
            hit += 1
    coverage = (hit / max(1, len(labels))) if labels else 0.0
    return {"extracted": extracted, "schema": sch, "coverage": coverage}


# ---- Attribute-driven extraction ----
@app.post("/extract/by_schema")
def api_extract_by_schema(doc_id: str = Query(...), class_uri: str = Query(...)):
    return _extract_by_schema(doc_id, class_uri)


# ---- Attribute-driven extraction ----
@app.post("/extract/by_schema")
def api_extract_by_schema(doc_id: str = Query(...), class_uri: str = Query(...)):
    return _extract_by_schema(doc_id, class_uri)
