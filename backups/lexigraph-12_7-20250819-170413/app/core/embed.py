from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INDEX_PKL = Path("data/tfidf_index.pkl")
META_JSONL = Path("data/tfidf_meta.jsonl")

class TFIDFIndex:
    def __init__(self):
        self.vec = None
        self.X = None
        self.meta = []
        self._load()

    def _load(self):
        if INDEX_PKL.exists() and META_JSONL.exists():
            import joblib
            self.vec, self.X = joblib.load(INDEX_PKL)
            self.meta = [json.loads(l) for l in META_JSONL.read_text().splitlines() if l.strip()]

    def _save(self):
        import joblib
        INDEX_PKL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.vec, self.X), INDEX_PKL)
        with META_JSONL.open("w") as f:
            for m in self.meta:
                f.write(json.dumps(m)+"\n")

    def add(self, doc_text: str, meta: Dict[str, Any]):
        doc_text = (doc_text or "").strip()
        if not self.vec:
            self.vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)
            self.X = self.vec.fit_transform([doc_text])
            self.meta = [meta]
        else:
            x = self.vec.transform([doc_text])
            from scipy.sparse import vstack
            self.X = vstack([self.X, x])
            self.meta.append(meta)
        self._save()

    def search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.vec or self.X is None or self.X.shape[0] == 0:
            return []
        q = self.vec.transform([query_text])
        sims = cosine_similarity(q, self.X)[0]
        idxs = np.argsort(-sims)[:k]
        out = []
        for i in idxs:
            out.append({"score": float(sims[i]), "meta": self.meta[int(i)]})
        return out
