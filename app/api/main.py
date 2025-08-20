# app/api/main.py
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.core.fibo_vec import build_fibo_vec, search_fibo
from app.core.fibo_attrs import get_attributes_for_class
from app.core.pdf_text import extract_text_blocks, focused_summary
from app.core.value_mapper import map_values_to_attributes

app = FastAPI(title="LexiGraph API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class LinkResponse(BaseModel):
    summary: str
    candidates: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/fibo/rebuild")
def fibo_rebuild():
    idx = build_fibo_vec(force=True)
    return {"source": idx.get("source"), "classes": len(idx.get("classes", []))}

@app.get("/fibo/search")
def fibo_search(q: str = Query(..., min_length=2), k: int = 10):
    hits = search_fibo(q, topk=k)
    return hits

@app.get("/fibo/attributes")
def fibo_attributes(uri: str):
    return get_attributes_for_class(uri)

@app.post("/extract/link", response_model=LinkResponse)
async def extract_and_link(file: UploadFile = File(...), k: int = 10):
    data = await file.read()
    ext = extract_text_blocks(data)
    summ = focused_summary(ext, max_chars=2000)
    hits = search_fibo(summ, topk=k)
    return {"summary": summ, "candidates": hits}

@app.post("/extract/map")
async def extract_and_map(
    file: UploadFile = File(...),
    class_uri: str = Query(...),
    threshold: float = 0.45,
):
    data = await file.read()
    ext = extract_text_blocks(data)
    attrs = get_attributes_for_class(class_uri)
    mapped = map_values_to_attributes(ext["full_text"], attrs["attributes"], threshold=threshold)
    return {
        "attributes": attrs,
        "mapped": mapped,
        "coverage": {"hit": len(mapped), "total": attrs.get("count", 0)}
    }
