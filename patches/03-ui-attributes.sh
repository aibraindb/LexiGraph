#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("ui/streamlit_app.py")
s = p.read_text()

inject_after = "st.header(\"2\")" if "st.header(\"2" in s else "st.header(\"2) Label"  # fuzzy anchor
if "/extract/by_schema" in s:
    print("UI already has by_schema block; skipping.")
else:
    block = """

st.markdown("### FIBO Attributes → Extraction")
colA, colB, colC = st.columns([1,1,1])

# Use current selection if available
try:
    sel_uri = requests.get(f"{api_base}/ui/get_selection", timeout=5).json().get("uri")
except Exception:
    sel_uri = None

class_uri = st.text_input("FIBO class URI (auto-filled on graph/tree select)", value=sel_uri or "")
doc_id_for_attr = st.text_input("doc_id for attribute extraction", value=st.session_state.get("last_doc_id",""))

if colA.button("Load Attributes", disabled=not class_uri):
    try:
        attrs = requests.get(f"{api_base}/fibo/attributes", params={"class_uri": class_uri}, timeout=30).json()
        st.session_state["attrs"] = attrs
        st.success(f"Loaded {attrs.get('count',0)} attributes")
        with st.expander("Attributes (labels & synonyms)"):
            st.json(attrs)
    except Exception as e:
        st.error(e)

if colB.button("Extract by Schema", disabled=not (class_uri and doc_id_for_attr)):
    try:
        res = requests.post(f"{api_base}/extract/by_schema", params={"doc_id": doc_id_for_attr, "class_uri": class_uri}, timeout=60).json()
        st.session_state["by_schema"] = res
        st.success(f"Coverage: {round(res.get('coverage',0)*100,1)}%")
    except Exception as e:
        st.error(e)

if colC.button("Show Side-by-Side", disabled=("by_schema" not in st.session_state)):
    res = st.session_state.get("by_schema", {})
    fields = res.get("fields", {})
    if not fields:
        st.info("No extracted fields.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**FIBO Properties**")
            st.text("\\n".join(list(fields.keys())[:200]))
        with right:
            st.markdown("**Best Values (heuristic)**")
            pretty = {k: v.get("best") for k,v in fields.items()}
            st.json(pretty)
"""
    # append at end for reliability
    s = s + block

Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: UI block added."
