import os, json, pickle, numpy as np
from pathlib import Path

try:
    import faiss  # type: ignore
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class SimpleVectorStore:
    """TF-IDF + FAISS (inner product) or sklearn cosine. Index per doc_id."""
    def __init__(self, index_dir="data/docs/index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorizer = None
        self.nn = None
        self.faiss_index = None
        self.doc_ids = []

    def _paths(self):
        return {
            "vecs": self.index_dir / "X.npy",
            "doc_ids": self.index_dir / "doc_ids.json",
            "vectorizer": self.index_dir / "vectorizer.pkl"
        }

    def save(self):
        paths = self._paths()
        if self.vectorizer is not None:
            with open(paths["vectorizer"], "wb") as f:
                pickle.dump(self.vectorizer, f)
        if hasattr(self, "X") and self.X is not None:
            np.save(paths["vecs"], self.X.astype("float32"))
        with open(paths["doc_ids"], "w") as f:
            json.dump(self.doc_ids, f)

    def load(self):
        paths = self._paths()
        if not paths["vectorizer"].exists() or not paths["vecs"].exists() or not paths["doc_ids"].exists():
            return False
        with open(paths["vectorizer"], "rb") as f:
            self.vectorizer = pickle.load(f)
        self.X = np.load(paths["vecs"])
        self.doc_ids = json.load(open(paths["doc_ids"]))
        self._build()
        return True

    def _build(self):
        if _HAVE_FAISS:
            dim = self.X.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            Xn = self._normalize(self.X.astype("float32"))
            self.faiss_index.add(Xn)
        else:
            self.nn = NearestNeighbors(metric="cosine", n_neighbors=5).fit(self.X)

    def _normalize(self, x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n

    def add_document(self, doc_id: str, text: str):
        existed = self.load()
        # reconstruct corpus from docs on disk for consistent TFIDF
        corpus = []
        doc_ids = []
        if existed:
            doc_ids = self.doc_ids
            for did in doc_ids:
                p = Path("data/docs") / f"{did}.json"
                try:
                    j = json.loads(p.read_text())
                    corpus.append(j.get("text",""))
                except Exception:
                    corpus.append("")
        corpus.append(text)
        doc_ids.append(doc_id)

        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.X = self.vectorizer.fit_transform(corpus).astype("float32").toarray()
        self.doc_ids = doc_ids
        self._build()
        self.save()

    def query(self, text: str, topk=5):
        if self.vectorizer is None or (self.faiss_index is None and self.nn is None):
            return []
        q = self.vectorizer.transform([text]).astype("float32").toarray()
        if _HAVE_FAISS:
            D, I = self.faiss_index.search(self._normalize(q), topk)
            sims = D[0].tolist()
            idxs = I[0].tolist()
        else:
            dists, idxs = self.nn.kneighbors(q, n_neighbors=min(topk, len(self.doc_ids)))
            sims = [1 - d for d in dists[0]]
        out = []
        for sim, idx in zip(sims, idxs):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            out.append({"doc_id": self.doc_ids[idx], "similarity": float(sim)})
        return out
