from pathlib import Path
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib, json
from .wf_ecm import get_wf

STORE = Path("data/wf_vec")
STORE.mkdir(parents=True, exist_ok=True)

def build_wf_index() -> Dict:
    wf = get_wf()
    classes = wf.classes()
    recs=[]
    for c in classes:
        props = wf.props_for(c["uri"])
        txt = " ".join([c["label"]] + [p["label"] for p in props] + sum([p.get("aliases",[]) for p in props], []))
        recs.append({"uri": c["uri"], "label": c["label"], "text": txt})
    vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1,2))
    X = vec.fit_transform([r["text"] for r in recs])
    joblib.dump(vec, STORE/"vec.joblib")
    joblib.dump(X, STORE/"X.joblib")
    (STORE/"meta.json").write_text(json.dumps(recs))
    return {"classes": len(recs)}

def _load():
    vec = joblib.load(STORE/"vec.joblib")
    X = joblib.load(STORE/"X.joblib")
    meta = json.loads((STORE/"meta.json").read_text())
    return vec, X, meta

def suggest_class(text: str, top_k=5):
    vec, X, meta = _load()
    q = vec.transform([text or ""])
    sims = cosine_similarity(q, X)[0]
    order = sims.argsort()[::-1][:top_k]
    return [{"uri": meta[i]["uri"], "label": meta[i]["label"], "score": float(sims[i])} for i in order]
