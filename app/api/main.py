from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os, yaml, tempfile, os.path as p, uuid, re, glob
from collections import Counter

from app.core.ocr import extract_text_from_pdf
from app.core.classifier import RuleClassifier
from app.core.schema import resolve_effective_schema, load_variant
from app.core.extractor import extract_fields
from app.core.embeddings import HybridIndex
from app.core.fibo import to_rdf
from app.core.dataset import ensure_dirs, write_text_copy, append_metadata

CONFIG_DIR = os.environ.get("CONFIG_DIR","config")
CFG = yaml.safe_load(open(p.join(CONFIG_DIR,'config.yaml'))) if os.path.exists(p.join(CONFIG_DIR,'config.yaml')) else {
    'app': {'vector_store_dir':'data/vector_index','embedding_dim':256,'default_mode':'hybrid','alpha':0.7,'beta':0.3}
}
VECTOR_DIR = CFG['app']['vector_store_dir']
EMBED_DIM = int(CFG['app'].get('embedding_dim',256))

app = FastAPI(title="LexiGraph API v3")
_classifier = RuleClassifier(CONFIG_DIR)
_index = HybridIndex.load(VECTOR_DIR, dim=EMBED_DIM)
ensure_dirs()

@app.get("/health")
def health():
    return {"status":"ok"}

def _rank_candidates(text: str, mode: str = "hybrid", topk: int = 5, alpha: float = 0.7, beta: float = 0.3):
    ranked = _classifier.classify(text, topk=topk)
    if mode == 'hybrid':
        for c in ranked:
            vid = c.get('variant_id')
            sim = _index.similarity_to_label(vid, text) if vid else 0.0
            c['similarity'] = max(sim, 0.0)
            conf = float(c.get('confidence', 0.6))
            local_alpha = alpha * (0.5 + 0.5 * conf)
            local_beta  = beta  * (1.5 - 0.5 * conf)
            c['hybrid_score'] = local_alpha * c['score'] + local_beta * c['similarity']
        ranked.sort(key=lambda x: x.get('hybrid_score', x['score']), reverse=True)
    else:
        ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked

@app.post("/classify/hybrid")
async def classify_hybrid(
    file: UploadFile = File(...),
    mode: str = Query(CFG['app'].get('default_mode','hybrid')),
    add_to_index: bool = Query(False),
    alpha: float = Query(CFG['app'].get('alpha',0.7)),
    beta: float = Query(CFG['app'].get('beta',0.3)),
    topk: int = Query(5)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=p.splitext(file.filename or "")[1] or ".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path)
    try: os.unlink(path)
    except Exception: pass
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    ranked = _rank_candidates(text, mode=mode, topk=topk, alpha=alpha, beta=beta)
    best = ranked[0]
    added_ids = []
    if add_to_index and best.get('variant_id'):
        added_ids = _index.add([text],[best['variant_id']]); _index.save(VECTOR_DIR)

    return {"mode": mode, "topk": ranked, "preview": text[:800], "added_ids": added_ids}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    include_rdf: bool = False,
    mode: str = Query(CFG['app'].get('default_mode','hybrid')),
    alpha: float = Query(CFG['app'].get('alpha',0.7)),
    beta: float = Query(CFG['app'].get('beta',0.3)),
    add_to_index: bool = Query(False),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=p.splitext(file.filename or "")[1] or ".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path)
    try: os.unlink(path)
    except Exception: pass
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    ranked = _rank_candidates(text, mode=mode, topk=5, alpha=alpha, beta=beta)
    best = ranked[0] if ranked else None
    if not best or not best.get("variant_id"):
        raise HTTPException(status_code=422, detail=f"no_variant_matched: {ranked}")
    vid = best["variant_id"]
    added_ids = []
    if add_to_index:
        added_ids = _index.add([text],[vid]); _index.save(VECTOR_DIR)

    eff = resolve_effective_schema(CONFIG_DIR, vid)
    fields = extract_fields(text, eff)

    out = {"classification": {"mode":mode, "best":best, "topk": ranked}, "variant": vid, "fields": fields, "added_ids": added_ids}
    if include_rdf:
        enriched = {}
        for k,v in fields.items():
            fibo_prop = eff["fields"].get(k,{}).get("fibo_property")
            e = dict(v); 
            if fibo_prop: e["fibo_property"] = fibo_prop
            enriched[k]=e
        fibo_cls = f"fibo-fbc-fi-fi:{best['type'].replace('_',' ').title().replace(' ','')}"
        out["rdf_turtle"] = to_rdf("doc001", fibo_cls, enriched)
    return JSONResponse(out)

# ---------- Variants listing/suggest/create ----------
@app.get("/variants/list")
def list_variants():
    vdir = p.join(CONFIG_DIR, "variants")
    out = []
    for name in os.listdir(vdir):
        if not name.endswith(".yaml"): 
            continue
        obj = yaml.safe_load(open(p.join(vdir, name)))
        out.append({"variant_id": obj.get("variant_id"), "doc_type": obj.get("doc_type"), "file": name})
    out.sort(key=lambda x: (x["doc_type"] or "", x["variant_id"] or ""))
    return {"variants": out}

def _tokenize(s): 
    return re.sub(r"[^A-Za-z0-9]+"," ",s).lower().split()

def _ngrams(tokens, n=1, m=3):
    out=[]
    for k in range(n, m+1):
        out += [" ".join(tokens[i:i+k]) for i in range(0, max(0,len(tokens)-k+1))]
    return out

@app.post("/variants/suggest")
async def variants_suggest(file: UploadFile = File(...), page_scope:int=1):
    with tempfile.NamedTemporaryFile(delete=False, suffix=p.splitext(file.filename or "")[1] or ".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path); os.unlink(path)
    if not text or not text.strip():
        raise HTTPException(422, "scanned_pdf_unsupported")

    sample = text.split("\f")[0] if "\f" in text and page_scope==1 else text
    toks = _tokenize(sample)
    grams = Counter(_ngrams(toks,1,3))
    anchors = [g for g,_ in grams.most_common(40) if len(g)>=4][:12]

    fields = {
      "agreement_type": {"strategy":"literal","value":"(fill me)"},
      "agreement_date": {"strategy":"regex","patterns":[r"(Agreement Date|Date)[:\s]+([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})"]},
      "lessee_name":    {"strategy":"anchor_window","anchor":"Lessee","window_lines_after":3},
      "lessor_name":    {"strategy":"anchor_window","anchor":"Lessor","window_lines_after":3},
      "total_amount":   {"strategy":"regex","patterns":[r"(Total|Amount Due|Balance Due)[:\s]*\$?([0-9,]+\.[0-9]{2})"]},
    }

    return {
      "identify": {"anchors_any_of": anchors[:8], "anchors_all_of": anchors[:2], "negative_anchors": [], "page_scope": page_scope},
      "extract": {"fields": fields}
    }

@app.post("/variants/create")
def variants_create(payload: dict = Body(...)):
    vid = payload.get("variant_id"); dtype = payload.get("doc_type")
    if not vid or not dtype:
        raise HTTPException(400,"variant_id and doc_type required")
    obj = {"variant_id": vid, "doc_type": dtype, "identify": payload.get("identify",{}), "extract": {"fields": payload.get("extract",{}).get("fields",{})}}
    vpath = p.join(CONFIG_DIR, "variants", f"{vid}.yaml")
    if os.path.exists(vpath):
        raise HTTPException(409, f"variant already exists: {vid}")
    with open(vpath,"w") as f: yaml.safe_dump(obj, f, sort_keys=False)
    global _classifier
    _classifier = RuleClassifier(CONFIG_DIR)
    return {"status":"ok","variant_id":vid,"path":vpath}

# ---------- Training / labeling ----------
@app.post("/train/label")
async def train_label(
    file: UploadFile = File(...),
    variant_id: str = Query(..., description="e.g., lease_docusign_v1"),
    persist_copy: bool = Query(True, description="store text for rule mining"),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=p.splitext(file.filename or "")[1] or ".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path)
    try: os.unlink(path)
    except Exception: pass
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")
    entry_ids = _index.add([text], [variant_id]); _index.save(VECTOR_DIR)
    if persist_copy:
        _ = write_text_copy("data/labeled", variant_id, text)
    append_metadata(variant_id, source_name=file.filename or "upload", chars=len(text))
    return {"status":"ok","variant_id":variant_id,"entry_ids":entry_ids,"chars":len(text)}

@app.post("/upload")
async def upload_and_label(
    file: UploadFile = File(...),
    variant_id: str | None = Query(None, description="If omitted, we auto-classify"),
    mode: str = Query(CFG['app'].get('default_mode','hybrid')),
    alpha: float = Query(CFG['app'].get('alpha',0.7)),
    beta: float = Query(CFG['app'].get('beta',0.3)),
    train_now: bool = Query(True, description="Add this text to embeddings immediately"),
    persist_for_rules: bool = Query(True, description="Keep a copy for rule mining (data/labeled)")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=p.splitext(file.filename or "")[1] or ".pdf") as tmp:
        tmp.write(await file.read()); path = tmp.name
    text = extract_text_from_pdf(path)
    try: os.unlink(path)
    except Exception: pass
    if not text or not text.strip():
        raise HTTPException(status_code=422, detail="scanned_pdf_unsupported")

    used_variant = variant_id
    cls_result = None
    if not used_variant:
        ranked = _rank_candidates(text, mode=mode, topk=5, alpha=alpha, beta=beta)
        if not ranked or not ranked[0].get("variant_id"):
            raise HTTPException(status_code=422, detail="no_variant_matched_for_upload")
        used_variant = ranked[0]["variant_id"]
        cls_result = ranked

    ds_path = write_text_copy("data/dataset", used_variant, text)
    if persist_for_rules:
        _ = write_text_copy("data/labeled", used_variant, text)
    append_metadata(used_variant, source_name=file.filename or "upload", chars=len(text))

    entry_ids = []
    if train_now:
        entry_ids = _index.add([text], [used_variant]); _index.save(VECTOR_DIR)

    return {
        "status": "ok",
        "variant_id": used_variant,
        "trained_now": train_now,
        "entry_ids": entry_ids,
        "dataset_path": ds_path,
        "classified_topk": cls_result or []
    }

# ---------- Index ops ----------
@app.get("/index/stats")
def index_stats():
    from collections import Counter
    c = Counter(_index.labels)
    return {"size": len(_index.labels), "by_label": c}

@app.get("/index/list")
def index_list(label: str | None = None, limit: int = 100):
    return {"entries": _index.list_entries(label=label, limit=limit)}

@app.delete("/index/delete")
def index_delete(id: str = Query(..., description="entry id to remove")):
    ok = _index.delete(id)
    if not ok: raise HTTPException(404, "entry id not found")
    _index.save(VECTOR_DIR)
    return {"status":"ok","deleted":id}

@app.post("/index/undo_last")
def index_undo_last():
    ids = _index.undo_last_add()
    if ids:
        _index.save(VECTOR_DIR)
    return {"status":"ok","undone": ids}

# ---------- Dataset ops ----------
@app.get("/dataset/stats")
def dataset_stats():
    root = "data/dataset"
    variants = []
    total = 0
    if os.path.exists(root):
        for name in os.listdir(root):
            d = p.join(root, name)
            if os.path.isdir(d):
                cnt = len(glob.glob(p.join(d, "*.txt")))
                total += cnt
                variants.append({"variant_id": name, "count": cnt})
    return {"total_texts": total, "by_variant": variants}

@app.post("/index/rebuild")
def index_rebuild_from_dataset():
    texts, labels = [], []
    root = "data/dataset"
    if not os.path.exists(root):
        return {"status":"ok", "replaced": 0}

    for variant in os.listdir(root):
        d = p.join(root, variant)
        if not os.path.isdir(d): continue
        for f in glob.glob(p.join(d, "*.txt")):
            try:
                t = open(f).read()
                if t.strip():
                    texts.append(t); labels.append(variant)
            except Exception:
                pass

    global _index
    _index = HybridIndex(dim=EMBED_DIM)
    if texts:
        _index.add(texts, labels)
    _index.save(VECTOR_DIR)
    return {"status":"ok", "replaced": len(texts)}

# ---------- Confusion / anchors report ----------
@app.get("/report/anchors")
def report_anchors(label_a: str, label_b: str, top:int=20):
    """Compare frequent n-grams to suggest positives for A and negatives from B."""
    import collections
    root = "data/dataset"
    def load_vari(label):
        texts=[]
        d = p.join(root, label)
        if os.path.isdir(d):
            for f in glob.glob(p.join(d, "*.txt")):
                try: texts.append(open(f).read())
                except: pass
        return texts
    def toks(s): return re.sub(r"[^A-Za-z0-9]+"," ",s).lower().split()
    def ngrams(tokens, n=1, m=3):
        bag=[]; 
        for k in range(n,m+1):
            bag += [" ".join(tokens[i:i+k]) for i in range(0,max(0,len(tokens)-k+1))]
        return bag
    A = load_vari(label_a); B = load_vari(label_b)
    ca = collections.Counter(); cb = collections.Counter()
    for t in A: ca.update(ngrams(toks(t),1,3))
    for t in B: cb.update(ngrams(toks(t),1,3))
    sugg_pos = [(g, ca[g]/(1+cb[g])) for g in ca if ca[g]>=2]
    sugg_pos.sort(key=lambda x: x[1], reverse=True)
    negs = [(g, cb[g]) for g in cb if cb[g]>=2 and ca.get(g,0)==0]
    negs.sort(key=lambda x: x[1], reverse=True)
    return {"positives_for_A": [g for g,_ in sugg_pos[:top]], "negatives_from_B": [g for g,_ in negs[:top]]}
