import argparse, os, glob, io
from app.core.vector_store import build_from_texts
from pdfminer.high_level import extract_text

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="samples", help="Folder of PDFs to index")
    p.add_argument("--name", default="docs", help="Vector store name")
    args = p.parse_args()
    texts = []
    for fp in glob.glob(os.path.join(args.path, "*.pdf")):
        try:
            txt = extract_text(fp)[:10000]
            texts.append(txt)
        except Exception:
            pass
    if not texts:
        print({"status":"no_docs"})
    else:
        print(build_from_texts(args.name, texts))
