import re
from .schema import suggest_schema_for

def simple_extract(text: str, label: str):
    spec = suggest_schema_for(label)
    if not spec:
        return {}
    out = {}
    for f in spec.get("fields", []):
        pat = f.get("pattern")
        pred = f.get("predicate")
        val = None
        if pat:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                if m.groups():
                    val = m.groups()[-1].strip()
                else:
                    val = m.group(0).strip()
        out[pred] = val
    return out
