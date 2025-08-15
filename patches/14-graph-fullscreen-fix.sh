#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("components/fibo_graph.html")
s = p.read_text()

# Ensure light background and responsive resize on fullscreen
if "function resize()" not in s:
    s = s.replace(
        "<style>",
        "<style>\nhtml,body{height:100%; background:#ffffff;}"
    )
    s = s.replace(
        "const w = el.clientWidth, h = el.clientHeight;",
        "let w = el.clientWidth, h = el.clientHeight;"
    )
    s = s.replace(
        "const svg = d3.select(\"#viz\").append(\"svg\").attr(\"width\", w).attr(\"height\", h);",
        "const svg = d3.select(\"#viz\").append(\"svg\").attr(\"width\", w).attr(\"height\", h);"
        "\nfunction resize(){\n"
        "  w = el.clientWidth || window.innerWidth;\n"
        "  h = el.clientHeight || window.innerHeight - 46;\n"
        "  svg.attr('width', w).attr('height', h);\n"
        "  simulation.force('center', d3.forceCenter(w/2, h/2));\n"
        "  simulation.alpha(0.05).restart();\n"
        "}\n"
        "window.addEventListener('resize', resize);\n"
        "document.addEventListener('fullscreenchange', resize);"
    )

Path(p).write_text(s)
print("Patched:", p)
PY
