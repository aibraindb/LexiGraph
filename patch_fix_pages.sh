#!/bin/bash
set -e

TARGET="ui/ocr_tree_canvas.py"

# Insert helper at top if missing
if ! grep -q "_page_to_dict" "$TARGET"; then
  echo ">>> Adding _page_to_dict helper..."
  sed -i.bak '1i\
from typing import Any, Dict\n\
\ndef _page_to_dict(obj: Any) -> Dict:\n\
    """Normalize an OCR Page object to a dict the UI can read."""\n\
    if isinstance(obj, dict):\n\
        return obj\n\
    d = {}\n\
    for k in ("img","lines","dpi","number","width","height"):\n\
        if hasattr(obj, k):\n\
            d[k] = getattr(obj, k)\n\
    if "img" in d and d["img"] is not None and hasattr(d["img"], "size"):\n\
        try:\n\
            d["width"], d["height"] = d["img"].size\n\
        except Exception:\n\
            pass\n\
    d.setdefault("lines", [])\n\
    d.setdefault("dpi", 72)\n\
    return d\n' "$TARGET"
fi

# Normalize pg before .get
echo ">>> Rewriting pg.get calls..."
sed -i.bak 's/lines = pg.get(/pg = _page_to_dict(pg)\n    lines = pg.get(/' "$TARGET"

echo ">>> Patch applied. Backup at ${TARGET}.bak"
