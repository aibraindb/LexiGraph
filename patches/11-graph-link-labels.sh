#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
from pathlib import Path
p = Path("components/fibo_graph.html")
s = p.read_text()

# Add title for link labels and CSS tweak
if "link.append(\"title\")" not in s:
    s = s.replace(
        "const link = g.append(\"g\").selectAll(\"line\").data(data.links).enter().append(\"line\").attr(\"class\",\"link\");",
        "const link = g.append(\"g\").selectAll(\"line\").data(data.links).enter().append(\"line\").attr(\"class\",\"link\");\n"
        "link.append(\"title\").text(d => d.kind === 'property' ? (d.label || 'property') : 'subClassOf');"
    )

Path(p).write_text(s)
print("Patched", p)
PY

echo "OK: Graph links show property labels on hover."
