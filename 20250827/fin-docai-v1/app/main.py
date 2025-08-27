
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, os
from .pipeline import process_document

app = FastAPI(title="fin-docai-v1 API", version="0.1.0")

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/process")
async def process(file: UploadFile = File(...)):
    fd, path = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
    with open(path, "wb") as f: f.write(await file.read())
    try:
        result = process_document(path)
        return JSONResponse(result.model_dump())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        try: os.remove(path)
        except: pass
