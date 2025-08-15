#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("components/fibo_graph.html")
s = p.read_text()

# Truncate on JS side (in case API didn't ship 'label' shortened),
# add node <title> with full label or id, and collision
if "function shortLabel(" not in s:
    s = s.replace(
        "<script>",
        "<script>\nfunction shortLabel(d){ const t=(d.label||d.id||''); return t.length>36? (t.slice(0,33)+'…'):t; }"
    )

# Add titles for nodes
if "node.append(\"title\")" not in s:
    s = s.replace(
        "const node = g.append(\"g\").selectAll(\"circle\").data(data.nodes).enter().append(\"circle\").attr(\"r\", 12).attr(\"class\",\"node\");",
        "const node = g.append(\"g\").selectAll(\"circle\").data(data.nodes).enter().append(\"circle\").attr(\"r\", 12).attr(\"class\",\"node\");\n"
        "node.append(\"title\").text(d => (d.full || d.label || d.id));"
    )

# Use the short label for text, and add a title to the text as well
if ".data(data.nodes).enter().append(\"text\")" in s and "shortLabel" not in s:
    s = s.replace(
        ".data(data.nodes).enter().append(\"text\").attr(\"class\",\"label\").text(d => d.label || d.id);",
        ".data(data.nodes).enter().append(\"text\").attr(\"class\",\"label\").text(d => shortLabel(d)).append('title').text(d => d.full || d.label || d.id);"
    )

# Add collide force (avoid overlaps)
if "forceCollide" not in s:
    s = s.replace(
        "const simulation = d3.forceSimulation(data.nodes)\n"
        "  .force(\"link\", d3.forceLink(data.links).id(d => d.id).distance(d => d.kind==='property'?140:90))\n"
        "  .force(\"charge\", d3.forceManyBody().strength(-260))\n"
        "  .force(\"center\", d3.forceCenter(w/2, h/2));",
        "const simulation = d3.forceSimulation(data.nodes)\n"
        "  .force(\"link\", d3.forceLink(data.links).id(d => d.id).distance(d => d.kind==='property'?150:100))\n"
        "  .force(\"charge\", d3.forceManyBody().strength(-300))\n"
        "  .force(\"collide\", d3.forceCollide().radius(48).iterations(2))\n"
        "  .force(\"center\", d3.forceCenter(w/2, h/2));"
    )

# Slightly smaller font + white background for readability (if not already present)
if "background:#ffffff" not in s:
    s = s.replace(
        "<style>",
        "<style>\nhtml,body{background:#ffffff;}\n"
        ".label{font: 12px sans-serif; pointer-events:none;}"
    )

Path(p).write_text(s)
print("Patched:", p)
PY

echo "OK. Reload the UI."
