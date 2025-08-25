#!/usr/bin/env bash
set -euo pipefail

FILE="ui/ocr_tree_canvas.py"

# Replace the problematic f-string line
sed -i.bak 's/options = \[f\"\[{\i+1}.*replace.*\]/options = []\nfor i, ln in enumerate(lines):\n    txt = (ln.get("text", "") or "")[:60].replace("\\n", " ")\n    options.append(f"[{i+1}] {txt}")/' $FILE

echo "✅ Patch applied to $FILE. Backup saved at $FILE.bak"
