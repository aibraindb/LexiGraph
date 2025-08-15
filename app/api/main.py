
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json, uuid

from app.core.embeddings import SimpleVectorStore
from app.core.utils import read_text_from_upload
from app.core import fibo_index

app = FastAPI(title="LexiGraph Onboard v6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DOCS_DIR = Path("data/docs"); DOCS_DIR.mkdir(parents=True, exist_ok=True)
RDF_DIR  = Path("data/rdf");  RDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR= Path("data/index");INDEX_DIR.mkdir(parents=True, exist_ok=True)
ALL_TTL  = RDF_DIR / "all.ttl"
UI_SEL   = Path("data/ui_selection.json")

store = SimpleVectorStore(str(INDEX_DIR))
fibo_index.build_index(force=False)

@app.get("/health")
def health():
    return {"status":"ok", **fibo_index.get_health()}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    text = read_text_from_upload(file.filename, data)
    if not text.strip():
        raise HTTPException(422, "empty_text_extracted: likely scanned; OCR not in MVP")
    doc_id = str(uuid.uuid4())[:8]
    rec = {"doc_id": doc_id, "filename": file.filename, "text": text, "fibo_class_uri": None}
    (DOCS_DIR/f"{doc_id}.json").write_text(json.dumps(rec, indent=2))
    store.add(text=text, doc_id=doc_id, fibo_class_uri=None)
    nbrs = store.search(text, topk=8)
    return {"doc_id": doc_id, "neighbors": nbrs}

class SearchReq(BaseModel):
    query: str
    topk: int = 8

@app.post("/search")
def search(req: SearchReq):
    return {"results": store.search(req.query, topk=req.topk)}

@app.get("/docs/list")
def docs_list():
    return [json.loads(p.read_text()) for p in DOCS_DIR.glob("*.json")]

# FIBO scoped endpoints
@app.get("/fibo/namespaces")
def fibo_namespaces():
    return fibo_index.get_namespaces()

class ScopeReq(BaseModel):
    namespaces: list[str] = []

@app.post("/fibo/scope")
def fibo_set_scope(req: ScopeReq):
    return fibo_index.set_scope(req.namespaces)

@app.get("/fibo/classes")
def fibo_classes():
    return fibo_index.get_scoped_classes()

@app.get("/fibo/search")
def fibo_search(q: str, limit: int = 25):
    return fibo_index.search_scoped(q, limit=limit)

@app.get("/fibo/subgraph")
def fibo_subgraph(focus: str, hops: int = 2):
    return fibo_index.subgraph_scoped(focus, hops=hops)

# Linking & RDF (type triple only)
class LinkReq(BaseModel):
    doc_id: str
    fibo_class_uri: str

from rdflib import Graph, URIRef, RDF
def rdf_type_turtle(doc_id: str, class_uri: str) -> str:
    g = Graph(); subj = URIRef(f"http://example.org/doc/{doc_id}")
    g.add((subj, RDF.type, URIRef(class_uri)))
    return g.serialize(format="turtle")

@app.post("/label/link")
def link(req: LinkReq):
    dj = DOCS_DIR/f"{req.doc_id}.json"
    if not dj.exists():
        raise HTTPException(404, f"doc_id_not_found: {req.doc_id}")
    doc = json.loads(dj.read_text())
    doc["fibo_class_uri"] = req.fibo_class_uri
    dj.write_text(json.dumps(doc, indent=2))
    store.update_label(req.doc_id, req.fibo_class_uri)
    ttl = rdf_type_turtle(req.doc_id, req.fibo_class_uri)
    (RDF_DIR/f"{req.doc_id}.ttl").write_text(ttl)
    if ALL_TTL.exists():
        ALL_TTL.write_text(ALL_TTL.read_text() + "\n" + ttl)
    else:
        ALL_TTL.write_text(ttl)
    return {"status":"ok","doc_id":req.doc_id,"fibo_class_uri":req.fibo_class_uri,"rdf_turtle":ttl}

@app.get("/rdf/{doc_id}")
def get_rdf(doc_id: str):
    p = RDF_DIR/f"{doc_id}.ttl"
    if not p.exists(): raise HTTPException(404, "rdf_not_found")
    return {"doc_id": doc_id, "turtle": p.read_text()}

# Neighbors & Suggest class
@app.get("/neighbors")
def neighbors(doc_id: str, topk: int = 8):
    p = DOCS_DIR/f"{doc_id}.json"
    if not p.exists(): raise HTTPException(404, f"doc_id_not_found: {doc_id}")
    doc = json.loads(p.read_text())
    return {"doc_id": doc_id, "neighbors": store.search(doc["text"], topk=topk)}

@app.post("/suggest/class")
def suggest(doc_id: str = Query(...)):
    return store.suggest_class(doc_id=doc_id, topk_neighbors=8)

# Minimal CA
class CAReq(BaseModel):
    case_id: str; product_id: str; customer_id: str
CA_STATE = {}

@app.post("/ca/associate")
def ca_associate(req: CAReq):
    expected = ["w2","bank_statement","lease_agreement","loan_agreement","invoice","funding_detail_sheet"]
    CA_STATE[req.case_id] = {"case_id":req.case_id,"product_id":req.product_id,"customer_id":req.customer_id,"expected":expected,"present":[],"missing":expected[:]}
    return CA_STATE[req.case_id]

@app.post("/ca/mark_present")
def ca_mark_present(case_id: str = Query(...), doc_type: str = Query(...)):
    st = CA_STATE.get(case_id); 
    if not st: return {"status":"no_case"}
    if doc_type not in st["present"]: st["present"].append(doc_type)
    if doc_type in st["missing"]: st["missing"].remove(doc_type)
    return st

from app.core import fibo_index
from pydantic import BaseModel

# build (or load) index once at startup
fibo_index.build_index(force=False)

@app.get("/health")
def health():
    return {"status": "ok", **fibo_index.get_health()}

@app.post("/fibo/reindex")
def fibo_reindex(force: bool = True):
    idx = fibo_index.build_index(force=force)
    return {"status": "reindexed", "num_classes": len(idx.get("classes", [])), "num_namespaces": len(idx.get("namespaces", []))}

@app.get("/fibo/namespaces")
def fibo_namespaces():
    return fibo_index.get_namespaces()

class ScopeReq(BaseModel):
    namespaces: list[str] = []

@app.post("/fibo/scope")
def fibo_set_scope(req: ScopeReq):
    return fibo_index.set_scope(req.namespaces)

@app.get("/fibo/classes")
def fibo_classes():
    return fibo_index.get_scoped_classes()

@app.get("/fibo/search")
def fibo_search(q: str, limit: int = 25):
    return fibo_index.search_scoped(q, limit=limit)

@app.get("/fibo/subgraph")
def fibo_subgraph(focus: str, hops: int = 2):
    return fibo_index.subgraph_scoped(focus, hops=hops)

# UI selection bridge (graph -> dropdown)
@app.post("/ui/set_selection")
def ui_set_selection(uri: str = Query(...)):
    UI_SEL.write_text(json.dumps({"uri": uri})); return {"status":"ok","uri":uri}

@app.get("/ui/get_selection")
def ui_get_selection():
    try: return json.loads(UI_SEL.read_text())
    except Exception: return {"uri": None}
