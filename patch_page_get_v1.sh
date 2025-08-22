#!/usr/bin/env bash
set -euo pipefail

TARGET="ui/ocr_tree_canvas.py"
BACKUP="${TARGET}.bak.pagefix-$(date +%Y%m%d-%H%M%S)"

if [ ! -f "$TARGET" ]; then
  echo "❌ $TARGET not found. Run this from the repo root."
  exit 1
fi

cp "$TARGET" "$BACKUP"
echo "🗄  Backed up to $BACKUP"

python - "$TARGET" <<'PY'
import io, re, sys

path = sys.argv[1]
src = open(path, "r", encoding="utf-8").read()

# 1) Ensure we have safe helpers after imports
helpers = r'''
# --- SAFE PAGE ACCESS HELPERS (inserted by patch) ---
from typing import Any, Dict

def _page_to_dict(obj: Any) -> Dict:
    """Normalize an OCR Page object to a dict the UI can read."""
    if isinstance(obj, dict):
        return obj
    d = {}
    # common attributes our OCRIndex.Page exposes
    for k in ("img","lines","dpi","number","width","height"):
        if hasattr(obj, k):
            d[k] = getattr(obj, k)
    # infer width/height from PIL.Image if present
    if "img" in d and d["img"] is not None and hasattr(d["img"], "size"):
        try:
            d["width"], d["height"] = d["img"].size
        except Exception:
            pass
    # ensure fields exist
    d.setdefault("lines", [])
    d.setdefault("dpi", 72)
    return d

def _get_current_page() -> Dict:
    pages = st.session_state.get("pages", [])
    if not pages:
        return {}
    idx = st.session_state.get("cur_page", 0)
    try:
        raw = pages[idx]
    except Exception:
        return {}
    return _page_to_dict(raw)
# --- END HELPERS ---
'''

if "def _get_current_page()" not in src:
    # insert helpers after the first block of imports
    src = re.sub(
        r"(^from .*?\n|^import .*?\n)+",
        lambda m: m.group(0) + "\n" + helpers + "\n",
        src,
        count=1,
        flags=re.MULTILINE
    )

# 2) Replace direct access patterns with safe accessor
# Common patterns we saw in your file:
patterns = [
    r"pg\s*=\s*st\.session_state\[\s*['\"]pages['\"]\s*\]\s*\[\s*cur_page\s*\]",
    r"pg\s*=\s*pages\s*\[\s*cur_page\s*\]",
]
for pat in patterns:
    src = re.sub(pat, "pg = _get_current_page()", src)

# 3) If code later did `lines = pg.get("lines", [])`, it's fine now.
# But if code ever did attribute access (pg.lines), normalize it too:
src = re.sub(r"pg\.lines\b", "pg.get('lines', [])", src)
src = re.sub(r"pg\.img\b", "pg.get('img')", src)

open(path, "w", encoding="utf-8").write(src)
print("✅ Applied safe page access patch.")
PY

echo "Done. Try: streamlit run ui/ocr_tree_canvas.py"
