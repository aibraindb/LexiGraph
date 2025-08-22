#!/usr/bin/env bash
set -euo pipefail

TARGET="ui/ocr_tree_canvas.py"
BACKUP="${TARGET}.bak.lenfix-$(date +%Y%m%d-%H%M%S)"

if [ ! -f "$TARGET" ]; then
  echo "❌ $TARGET not found. Run this from the repo root."
  exit 1
fi

cp "$TARGET" "$BACKUP"
echo "🗄  Backed up to $BACKUP"

python - "$TARGET" <<'PY'
import re, sys, io

path = sys.argv[1]
src = open(path, 'r', encoding='utf-8').read()

# 1) Replace the upload handler block to normalize OCRIndex vs list
src = re.sub(
    r"""uploaded\s*=\s*st\.sidebar\.file_uploader\([^\n]*\)\s*\nif uploaded:\n\s*try:\n\s*pdf_bytes\s*=\s*uploaded\.read\(\)\n\s*pages\s*=\s*index_pdf_bytes\(pdf_bytes\)\n\s*if not pages:\n\s*    st\.error\([^\n]*\)\n\s*else:\n\s*    st\.session_state\["pages"\]\s*=\s*pages\n\s*    st\.session_state\["cur_page"\]\s*=\s*0\n\s*    st\.session_state\["selected_li"\]\s*=\s*None\n\s*    st\.session_state\["edits"\]\s*=\s*\[\]\n\s*    st\.session_state\["undo_stack"\]\.clear\(\)\n\s*    st\.session_state\["redo_stack"\]\.clear\(\)\n\s*    st\.success\(f"Loaded \{len\(pages\)\} page\(s\)\."\)\n\s*except Exception as e:\n\s*    st\.exception\(e\)\n""",
    """uploaded = st.sidebar.file_uploader("Upload a PDF (scanned or digital)", type=["pdf"])
if uploaded:
    try:
        pdf_bytes = uploaded.read()
        idx_or_pages = index_pdf_bytes(pdf_bytes)
        # Normalize: accept OCRIndex (with .pages) or plain list of pages
        pages = getattr(idx_or_pages, "pages", idx_or_pages)
        if not pages:
            st.error("No pages indexed. OCR/PDF parsing returned empty.")
        else:
            st.session_state["pages"] = pages
            st.session_state["cur_page"] = 0
            st.session_state["selected_li"] = None
            st.session_state["edits"] = []
            st.session_state["undo_stack"].clear()
            st.session_state["redo_stack"].clear()
            st.success(f"Loaded {len(pages)} page(s).")
    except Exception as e:
        st.exception(e)
""",
    src,
    flags=re.MULTILINE
)

# 2) Sidebar counter: ensure it references normalized list
src = re.sub(
    r'st\.sidebar\.write\(f"Pages:\s*\{len\(st\.session_state\[\'pages\'\]\)\}"\)',
    'st.sidebar.write(f"Pages: {len(st.session_state.get(\'pages\', []))}")',
    src
)

open(path, 'w', encoding='utf-8').write(src)
print("✅ Applied OCRIndex len() normalization.")
PY

echo "Done. Run: streamlit run ui/ocr_tree_canvas.py"
