from __future__ import annotations
from typing import Dict

KEY_HINTS = {
    "lease": 0.92,
    "loan": 0.91,
    "levy": 0.9,
    "guarantee": 0.88,
    "invoice": 0.87,
    "funding detail": 0.86,
    "acceptance": 0.85,
}

def ecm_classify(summary_text: str) -> Dict|None:
    s = (summary_text or "").lower()
    for k,score in KEY_HINTS.items():
        if k in s:
            # In real life return canonical URI from ECM. Here: just a label and score.
            return {"label": k, "score": score}
    return None
