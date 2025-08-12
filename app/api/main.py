from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import os, tempfile

from app.core.ocr import extract_text_from_pdf
from app.core.classifier import RuleClassifier
from app.core.schema import resolve_effective_schema
from app.core.extractor import extract_fields
from app.core.fibo import to_rdf
from app.core.embeddings import HybridIndex

CONFIG_DIR = os.environ.get("CONFIG_DIR","config")
VECTOR_DIR = os.environ.get("VECTOR_STORE_DIR", "data/vector_index")

app = FastAPI(title="Document Intelligence MVP")
_classifier = RuleClassifier(CONFIG_DIR)

_index = HybridIndex.load(VECTOR_DIR, dim=256)  # load if exists, else empty

@app.post("/classify")
async def classify(files: List[UploadFile] = File(...)):
    out = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read())
            path = tmp.name
        text = extract_text_from_pdf(path)
        os.unlink(path)
        if not text:
            out.append({"filename": f.filename, "error":"scanned_pdf_unsupported"})
            continue
        res = _classifier.classify(text)
        res["filename"] = f.filename
        res["preview"] = text[:400]
        out.append(res)
    return out

@app.post("/classify/hybrid")
async def classify_hybrid(
    file: UploadFile = File(...),
    mode: str = Query("hybrid", description="'rule' or 'hybrid'"),
    add_to_index: bool = Query(False),
    alpha: float = Query(0.7),
    beta: float = Query(0.3),
):
    """
    mode=rule: returns rule-based top-k
    mode=hybrid: rules + embedding similarity-to-centroid per variant
    add_to_index=true will add this doc text with chosen label (top result) and persist.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path); os.unlink(path)
    if not text:
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    ranked = _classifier.classify(text, topk=5)  # requires updated RuleClassifier w/ topk
    if mode == "rule":
        # Optionally add to index with top candidate
        if add_to_index and ranked and ranked[0].get("variant_id"):
            _index.add([text], [ranked[0]["variant_id"]])
            _index.save(VECTOR_DIR)
        return {"mode": "rule", "topk": ranked, "preview": text[:800]}

    # hybrid: combine
    for cand in ranked:
        vid = cand.get("variant_id")
        sim = _index.similarity_to_label(vid, text) if vid else 0.0
        cand["similarity"] = round(max(sim, 0.0), 4)
        cand["hybrid_score"] = round(alpha * cand["score"] + beta * max(sim, 0.0), 4)
    ranked.sort(key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)

    if add_to_index and ranked and ranked[0].get("variant_id"):
        _index.add([text], [ranked[0]["variant_id"]])
        _index.save(VECTOR_DIR)

    return {"mode": "hybrid", "topk": ranked, "preview": text[:800]}

@app.get("/index/stats")
def index_stats():
    from collections import Counter
    c = Counter(_index.labels)
    return {"size": len(_index.labels), "by_label": c}

@app.post("/index/add")
def index_add(payload: Dict[str, Any] = Body(...)):
    """
    { "text": "...", "label": "lease_docusign_v1" }
    """
    text = payload.get("text", "")
    label = payload.get("label")
    if not text or not label:
        raise HTTPException(status_code=400, detail="text and label required")
    _index.add([text], [label])
    _index.save(VECTOR_DIR)
    return {"status": "ok"}



@app.get("/schema")
def schema(variant_id: str = Query(...)):
    try:
        return resolve_effective_schema(CONFIG_DIR, variant_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="variant not found")

# --- add near top if not present ---
from os import environ
VECTOR_DIR = environ.get("VECTOR_STORE_DIR", "data/vector_index")

# if you haven't already switched:
from app.core.embeddings import HybridIndex
_index = HybridIndex.load(VECTOR_DIR, dim=256)

# --- helper just above the /extract endpoint ---
def _rank_candidates(text: str, mode: str = "hybrid", topk: int = 5, alpha: float = 0.7, beta: float = 0.3):
    """
    Returns ranked candidates from rules; if hybrid, mixes in similarity-to-centroid.
    Each candidate has: type, variant_id, score, confidence, rationale, [similarity], [hybrid_score]
    """
    ranked = _classifier.classify(text, topk=topk)  # list
    if mode == "hybrid":
        for c in ranked:
            vid = c.get("variant_id")
            sim = _index.similarity_to_label(vid, text) if vid else 0.0
            c["similarity"] = max(sim, 0.0)
            c["hybrid_score"] = alpha * c["score"] + beta * c["similarity"]
        ranked.sort(key=lambda x: x.get("hybrid_score", x["score"]), reverse=True)
    else:
        ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

# --- replace the entire /extract function with this ---
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    include_rdf: bool = False,
    mode: str = Query("hybrid", description="'rule' or 'hybrid'"),
    alpha: float = Query(0.7),
    beta: float = Query(0.3),
    add_to_index: bool = Query(False, description="Add this doc to the vector index using chosen variant"),
):
    import tempfile, os, os.path as p

    # Save with correct suffix; support .txt for easy demos
    suffix = p.splitext(file.filename or "")[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".pdf") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        if suffix == ".txt":
            with open(path, "r", errors="ignore") as fh:
                text = fh.read()
        else:
            text = extract_text_from_pdf(path)

    finally:
        try: os.unlink(path)
        except Exception: pass

    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    # Rank candidates (rule or hybrid)
    ranked = _rank_candidates(text, mode=mode, topk=5, alpha=alpha, beta=beta)
    best = ranked[0] if ranked else None
    if not best or not best.get("variant_id"):
        raise HTTPException(status_code=422, detail=f"no_variant_matched: {ranked}")

    vid = best["variant_id"]

    # Learn (optional) — persist to disk
    if add_to_index:
        _index.add([text], [vid])
        _index.save(VECTOR_DIR)

    # Resolve schema and extract fields
    try:
        eff = resolve_effective_schema(CONFIG_DIR, vid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"variant not found: {vid}")

    fields = extract_fields(text, eff)

    # Build response
    out = {
        "classification": {"mode": mode, "best": best, "topk": ranked},
        "variant": vid,
        "fields": fields,
        "embedding_ids": _index.add([], []),  # no-op; keep response shape stable
    }

    # Optional RDF (best-effort)
    if include_rdf:
        enriched = {}
        for k, v in fields.items():
            fibo_prop = eff["fields"].get(k, {}).get("fibo_property")
            e = dict(v)
            if fibo_prop:
                e["fibo_property"] = fibo_prop
            enriched[k] = e
        # Map doc type -> simple FIBO class name (same heuristic as before)
        fibo_cls = f"fibo-fbc-fi-fi:{best['type'].replace('_',' ').title().replace(' ','')}"
        out["rdf_turtle"] = to_rdf("doc001", fibo_cls, enriched)

    return JSONResponse(out)

@app.post("/feedback")
def feedback(payload: Dict[str,Any] = Body(...)):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/feedback.jsonl","a") as f:
        import json; f.write(json.dumps(payload)+"\n")
    return {"status":"ok"}

@app.post("/missing")
def missing(classified_types: List[str] = Body(...), workflow: str = Query("loan_underwriting")):
    import yaml, os.path as p
    rules = yaml.safe_load(open(p.join(CONFIG_DIR,"required_sets.yaml")))
    req = rules.get(workflow,{})
    required = set(req.get("required",[]))
    have = set(classified_types)
    missing = sorted(list(required - have))
    text = "All required documents present." if not missing else "Please upload: " + ", ".join(missing) + "."
    return {"workflow":workflow, "required_missing":missing, "customer_request_text": text}

@app.post("/classify/debug")
async def classify_debug(file: UploadFile = File(...), topk: int = 5):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path); os.unlink(path)
    if not text:
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    ranked = _classifier.classify(text, topk=topk)
    return {"topk": ranked, "preview": text[:800]}

@app.post("/embeddings/dump")
async def dump_embeddings(files: List[UploadFile] = File(...)):
    import tempfile, os, json
    out = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await f.read()); path = tmp.name
        text = extract_text_from_pdf(path); os.unlink(path)
        if not text: continue
        best = _classifier.classify(text, topk=1)[0]
        out.append({"doc_id": f.filename, "type": best.get("type"), "text": text})
    return JSONResponse(out)
