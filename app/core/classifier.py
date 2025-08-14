from typing import Dict, Any, List, Tuple
import os, yaml

def _lower(s): return s.lower() if isinstance(s, str) else s

def _page_split(text: str, sep="\f"):
    return (text or "").split(sep)

class RuleClassifier:
    def __init__(self, config_dir: str):
        self.variants: List[Dict[str,Any]] = []
        vdir = os.path.join(config_dir, "variants")
        for name in os.listdir(vdir):
            if name.endswith(".yaml"):
                with open(os.path.join(vdir, name),'r') as f:
                    self.variants.append(yaml.safe_load(f))

    def _variant_score(self, text: str, v: Dict[str,Any]) -> Tuple[float, List[str]]:
        ident = v.get('identify',{})
        any_of = [_lower(a) for a in ident.get('anchors_any_of',[])]
        all_of = [_lower(a) for a in ident.get('anchors_all_of',[])]
        negatives = [_lower(a) for a in ident.get('negative_anchors',[])]
        page_scope = ident.get('page_scope', 'any')

        pages = _page_split(text)
        if page_scope == 'any':
            search_text = text.lower()
        elif isinstance(page_scope, int):
            search_text = pages[page_scope-1].lower() if len(pages) >= page_scope else ""
        elif isinstance(page_scope, list):
            buf = []
            for p in page_scope:
                if isinstance(p, int) and len(pages) >= p: buf.append(pages[p-1].lower())
            search_text = "\n".join(buf)
        else:
            search_text = text.lower()

        rationale = []
        score = 0.0

        if all_of:
            if all(a in search_text for a in all_of):
                score += 2.0 * len(all_of); rationale += [f"hit(all): {a}" for a in all_of]
            else:
                return 0.0, ["miss(all)"]

        if any_of:
            hits = [a for a in any_of if a in search_text]
            if hits:
                score += 1.0 * len(hits); rationale += [f"hit(any): {a}" for a in hits]
            else:
                return 0.0, ["miss(any)"]

        neg_hits = [n for n in negatives if n in search_text]
        if neg_hits:
            rationale += [f"neg: {n}" for n in neg_hits]
            score -= 2.0 * len(neg_hits)

        return max(score, 0.0), rationale

    def classify(self, text: str, topk: int = 3) -> List[Dict[str,Any]]:
        scored = []
        for v in self.variants:
            s, why = self._variant_score(text, v)
            if s > 0:
                scored.append({
                    "type": v.get('doc_type'),
                    "variant_id": v.get('variant_id'),
                    "score": s,
                    "confidence": min(0.99, 0.5 + 0.07*s),
                    "rationale": why
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:topk] or [{"type":"other","variant_id":None,"confidence":0.0,"rationale":["no anchors matched"],"score":0.0}]
