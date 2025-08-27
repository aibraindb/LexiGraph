
import streamlit as st, json, os, tempfile
from PIL import Image, ImageDraw
from app.pipeline import process_document

st.set_page_config(page_title="fin-docai-v1", layout="wide")
st.title("fin-docai-v1 — Financial Document AI (v1)")

uploaded = st.file_uploader("Upload PDF", type=["pdf"])
if uploaded:
    fd, path = tempfile.mkstemp(suffix=".pdf"); os.close(fd)
    with open(path,"wb") as f: f.write(uploaded.getbuffer())
    try:
        result = process_document(path)
        st.subheader(f"Detected Type: {result.doc_type}")
        c1, c2 = st.columns([3,2], gap="large")

        with c1:
            size = result.meta.get("page_size", {"w":612,"h":792}); W, H = int(size["w"]), int(size["h"])
            img = Image.new("RGB", (W,H), (250,250,250)); draw = ImageDraw.Draw(img)
            for w in result.words[:1500]:
                b = w.bbox; draw.rectangle([b.x0,b.y0,b.x1,b.y1], outline=(220,220,220), width=1)
            for f in result.fields:
                if f.bbox:
                    b = f.bbox
                    draw.rectangle([b.x0,b.y0,b.x1,b.y1], outline=(0,0,0), width=2)
            st.image(img, use_column_width=True)

        with c2:
            edited=[]
            for f in result.fields:
                nv = st.text_input(f.name, f.value)
                if nv != f.value: f.value = nv
                edited.append({"name":f.name,"value":f.value})

            st.markdown("### Semantics")
            for f in result.fields:
                if getattr(f, "semantics", None):
                    st.write(f"**{f.name}** → {', '.join(f.semantics)}"
                             + (f"  _(normalized: {f.normalized})" if getattr(f, "normalized", None) else ""))

            st.markdown("### Issues")
            if result.issues: st.error("\n".join(result.issues))
            else: st.success("No validation issues detected.")

            if st.button("Save Audit Log"):
                from datetime import datetime
                log = {"doc_type": result.doc_type, "edited_fields": edited,
                       "issues": result.issues, "timestamp": datetime.utcnow().isoformat()+"Z"}
                st.download_button("Download Audit Log JSON", data=json.dumps(log, indent=2),
                                   file_name="audit_log.json", mime="application/json")
    finally:
        try: os.remove(path)
        except: pass
