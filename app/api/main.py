# app/api/main.py (inside FastAPI app)
from fastapi import FastAPI, UploadFile, File, Form
from app.core.fibo_vec import build_fibo_vec, search_fibo

app = FastAPI()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/fibo/rebuild")
def fibo_rebuild():
    info = build_fibo_vec(force=True)
    return {"status":"rebuilt", "nodes": len(info.get("nodes", []))}

@app.get("/fibo/search")
def fibo_search(q: str, k: int = 10):
    return search_fibo(q, top_k=k)

# Optional “suggest” endpoint: given text returns best few classes
@app.post("/suggest")
async def suggest(text: str = Form(...), k: int = Form(5)):
    return search_fibo(text, top_k=k)
