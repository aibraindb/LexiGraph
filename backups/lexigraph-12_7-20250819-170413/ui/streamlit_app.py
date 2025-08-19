import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import json
import streamlit as st
from PIL import Image
from app.core.pdf_io import render_page, page_count, page_words_pixels, draw_boxes
from app.core.ocr_backend import OCRBackend
from app.core.kv_extract import ocr_to_lines, kv_from_lines
from app.core.embed import TFIDFIndex
from app.core.fibo_index import build_index, search_scoped, attributes_for_class_tolerant

st.set_page_config(page_title="LexiGraph 12.7", layout="wide")
st.title("📄 LexiGraph 12.7 — Extract → Embed → FIBO Attributes")

with st.sidebar:
    st.markdown("### Ontology")
    if st.button("Rebuild FIBO index"):
        from pathlib import Path
        idx = Path("data/fibo_index.json")
        if idx.exists(): idx.unlink()
        st.success("FIBO index cleared. It will rebuild on first search.")
    st.divider()
    st.markdown("### OCR")
    prefer_backend = st.selectbox("OCR backend", ["easyocr", "paddle"], index=0)
    dpi = st.slider("Render DPI", 120, 300, 220, 10)

st.markdown("#### 1) Upload a PDF")
uploaded = st.file_uploader("PDF (digital or scanned)", type=["pdf"])

if uploaded:
    # Save
    out = Path("data/uploads"); out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / uploaded.name
    pdf_path.write_bytes(uploaded.read())

    # Extraction loop (first page for speed; adjust if you want more)
    pages_json = []
    agg_kv = {}
    try:
        ocr = OCRBackend(lang="en", prefer=prefer_backend)
        ocr_ok = True
    except Exception as e:
        st.warning(f"OCR init failed: {e}. Digital text only.")
        ocr = None; ocr_ok = False

    n = page_count(pdf_path)
    show_pages = min(n, 2)

    full_text = []
    for p in range(show_pages):
        img, mat = render_page(pdf_path, page_no=p, dpi=dpi)
        items = []
        # digital first
        for w in page_words_pixels(pdf_path, p, mat):
            x0,y0,x1,y1 = w["bbox"]
            items.append({"polygon":[(x0,y0),(x1,y0),(x1,y1),(x0,y1)], "bbox":[x0,y0,x1,y1], "text": w["text"], "conf":1.0})
            full_text.append(w["text"])
        # fallback to OCR
        if not items and ocr is not None:
            items = ocr.run(img)
            full_text.extend([it["text"] for it in items])

        lines = ocr_to_lines(items)
        kv = kv_from_lines(lines)
        for k,v in kv.items(): agg_kv.setdefault(k,v)

        # preview with boxes
        boxed = draw_boxes(img, items, max_boxes=200)
        st.image(boxed, caption=f"Page {p+1} preview with boxes", use_column_width=True)

        with st.expander(f"Page {p+1} KV / Lines (debug)"):
            st.json({"kv": kv, "lines": lines[:8]})

        pages_json.append({"page": p, "kv": kv, "n_boxes": len(items)})

    doc_text = " ".join(full_text)[:200000]

    st.markdown("---")
    st.markdown("#### 2) Embed & Auto-link to FIBO")
    idx = TFIDFIndex()
    idx.add(doc_text, {"name": uploaded.name})
    st.success("Document embedded into TF-IDF index.")

    q = st.text_input("Search FIBO (label/altLabel/local) to override auto-link", "")
    if st.button("Find classes") and q.strip():
        hits = search_scoped(q, limit=25, fallback_all=True)
        for h in hits:
            st.write(f"- **{h['label']}**  
{h['uri']}")

    # naive auto-link: compare doc text vs each class search_text
    # We'll reuse the TFIDF vectorizer: build a tiny corpus of class texts and compare to doc
    import json as _json, numpy as _np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from app.core.fibo_index import build_index as _build_idx
    fibo = _build_idx()
    class_texts = [c.get("search_text","") for c in fibo.get("classes",[])]
    labels = [c.get("label","") for c in fibo.get("classes",[])]
    uris = [c.get("uri","") for c in fibo.get("classes",[])]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
    X = vec.fit_transform(class_texts + [doc_text])
    sims = cosine_similarity(X[-1], X[:-1])[0]
    top_i = int(_np.argmax(sims)) if sims.size else -1
    selected_uri = uris[top_i] if top_i >= 0 else ""
    selected_label = labels[top_i] if top_i >= 0 else ""
    st.info(f"Auto-linked FIBO class: **{selected_label or '—'}**")
    if selected_uri:
        st.code(selected_uri, language="text")

    st.markdown("---")
    st.markdown("#### 3) Attribute coverage (from FIBO)")
    ttl = Path("data/fibo_full.ttl") if Path("data/fibo_full.ttl").exists() else Path("data/fibo_trimmed.ttl")
    if selected_uri:
        attrs = attributes_for_class_tolerant(str(ttl), selected_uri)
        # match by synonyms against extracted KV keys
        kv_lower = {k.lower(): v for k,v in agg_kv.items()}
        rows = []
        matched = 0
        for a in attrs.get("attributes", []):
            labels = [s.lower() for s in a.get("labels", [])]
            match_key = None
            for l in labels:
                if l in kv_lower: match_key = l; break
            val = kv_lower.get(match_key) if match_key else None
            if val is not None: matched += 1
            rows.append({"property": a.get("property"), "labels": ", ".join(a.get("labels", [])), "matched_key": match_key, "value": val})
        st.write(f"Coverage: **{matched}/{len(rows)}**")
        st.dataframe(rows, use_container_width=True)
        st.download_button("Download JSON (class+kv+mappings)",
                           data=json.dumps({"class_uri": selected_uri, "class_label": selected_label,
                                            "kv": agg_kv, "mappings": rows}, indent=2),
                           file_name=f"{uploaded.name}.json",
                           mime="application/json")
    else:
        st.warning("No FIBO class was auto-linked. Try manual search above.")

else:
    st.info("Upload a PDF to begin.")
