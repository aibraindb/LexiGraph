from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json, uuid, time

from app.core.utils import read_file_as_text
from app.core.embeddings import SimpleVectorStore
from app.core.fibo import list_doc_classes, make_doc_rdf, save_doc_rdf
from app.core.extract import simple_extract

app = FastAPI(title="LexiGraph Simple v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOC_DIR = Path("data/docs")
DOC_DIR.mkdir(parents=True, exist_ok=True)
INDEX = SimpleVectorStore("data/docs/index")

def new_doc_id():
    return "DOC-" + uuid.uuid4().hex[:8].upper()

@app.get("/health")
def health():
    return {"status":"ok"}

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    size: int
    preview: str
    neighbors: list

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), topk: int = Query(5)):
    data = await file.read()
    text = read_file_as_text(file.filename, data)
    if not text or not text.strip():
        raise HTTPException(422, "empty_text_extracted: scanned PDF not supported in MVP")
    doc_id = new_doc_id()
    rec = {
        "doc_id": doc_id,
        "filename": file.filename,
        "size": len(data),
        "created_at": int(time.time()),
        "text": text
    }
    (DOC_DIR / f"{doc_id}.json").write_text(json.dumps(rec, ensure_ascii=False))
    INDEX.add_document(doc_id, text)
    neigh = INDEX.query(text, topk=topk)
    return UploadResponse(doc_id=doc_id, filename=file.filename, size=len(data),
                          preview=text[:600], neighbors=neigh)

class SearchRequest(BaseModel):
    query: str
    topk: int = 5

@app.post("/search")
def search(req: SearchRequest):
    res = INDEX.query(req.query, topk=req.topk)
    for r in res:
        p = DOC_DIR / f"{r['doc_id']}.json"
        try:
            j = json.loads(p.read_text())
            r["filename"] = j.get("filename")
        except Exception:
            pass
    return res

@app.get("/docs/list")
def docs_list():
    out = []
    for p in DOC_DIR.glob("*.json"):
        try:
            j = json.loads(p.read_text())
            out.append({k:j.get(k) for k in ["doc_id","filename","size","created_at"]})
        except Exception:
            continue
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return out

@app.get("/fibo/classes")
def fibo_classes():
    return list_doc_classes()

class LinkRequest(BaseModel):
    doc_id: str
    class_label: str
    class_uri: Optional[str] = None
    extract: bool = True

@app.post("/fibo/link")
def fibo_link(req: LinkRequest):
    p = DOC_DIR / f"{req.doc_id}.json"
    if not p.exists():
        raise HTTPException(404, "doc_not_found")
    j = json.loads(p.read_text())
    text = j.get("text","")

    class_uri = req.class_uri
    if not class_uri:
        from app.core.schema import suggest_schema_for
        spec = suggest_schema_for(req.class_label)
        if spec:
            class_uri = spec.get("class_uri")
    if not class_uri:
        raise HTTPException(400, "class_uri_missing_and_unresolvable")

    fields = {}
    if req.extract:
        fields = simple_extract(text, req.class_label) or {}

    g = make_doc_rdf(req.doc_id, class_uri, fields)
    path = save_doc_rdf(req.doc_id, g)
    return {
        "status":"ok",
        "doc_id": req.doc_id,
        "class_uri": class_uri,
        "fields": fields,
        "rdf_path": path,
        "rdf_turtle": g.serialize(format="turtle")
    }
