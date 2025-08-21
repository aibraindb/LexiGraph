from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import json

from app.core.pipeline import propose_schema, apply_schema
from app.core.fibo_vec import rebuild_fibo_index, ensure_fibo_index
from app.core.fibo_index import build_index, get_health, get_namespaces, set_scope, search_scoped, subgraph_scoped

app = FastAPI(title="LexiGraph API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ApplyPayload(BaseModel):
    doc_id: str
    schema: Dict[str, Any]

@app.get("/health")
def health():
    info = get_health()
    try:
        ensure_fibo_index()
        info["fibo_vec"] = "ok"
    except Exception as e:
        info["fibo_vec"] = f"degraded: {e}"
    return info

# FIBO routes
@app.post("/fibo/rebuild")
def fibo_rebuild():
    a = build_index(force=True)
    b = rebuild_fibo_index()
    return {"index": {"classes": len(a.get("classes",[]))}, "vec": b}

@app.get("/fibo/namespaces")
def fibo_namespaces():
    return get_namespaces()

@app.post("/fibo/scope")
def fibo_scope(namespaces: List[str] = Body(default=[])):
    return set_scope(namespaces)

@app.get("/fibo/search")
def fibo_search(q: str, limit: int = 25):
    return search_scoped(q, limit=limit, fallback_all=True)

@app.get("/fibo/subgraph")
def fibo_subgraph(focus: str, hops: int = 2):
    return subgraph_scoped(focus_uri=focus, hops=hops, include_properties=True)

# Pipeline
@app.post("/pipeline/propose")
async def pipeline_propose(file: UploadFile = File(...), topk: int = Form(5), score_floor: float = Form(0.25)):
    fb = await file.read()
    out = propose_schema(fb, file.filename, topk_classes=topk, score_floor=score_floor)
    return out

@app.post("/pipeline/apply")
async def pipeline_apply(doc_id: str = Form(...), schema: str = Form(...), file: UploadFile = File(...)):
    fb = await file.read()
    try:
        schema_obj = json.loads(schema)
    except Exception:
        schema_obj = {"documentName": file.filename, "attributes": []}
    out = apply_schema(doc_id, fb, schema_obj)
    return out
