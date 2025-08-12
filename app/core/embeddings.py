# app/core/embeddings.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os, json
import numpy as np

# ----- Optional FAISS; we rebuild it from memory on load -----
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False


def _embed(text: str, dim: int = 256) -> np.ndarray:
    """
    Deterministic toy embedding for MVP. Swap for a real model later.
    """
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim).astype("float32")
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


class HybridIndex:
    """
    Persistent in-memory store w/ optional FAISS acceleration.

    Stores:
      - self.vecs: (N, dim) float32
      - self.labels: List[str] (e.g., variant_id)
      - self.payloads: List[str] (raw text)
    Can:
      - add(texts, labels)
      - save(path) / load(path)
      - similarity_to_label(label, text): cosine to centroid(label)
      - search(query, k)
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vecs: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.payloads: List[str] = []
        self._faiss = None

    # ---------- persistence ----------
    def save(self, store_dir: str):
        os.makedirs(store_dir, exist_ok=True)
        meta = {
            "dim": self.dim,
            "labels": self.labels,
        }
        with open(os.path.join(store_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        # vecs
        if self.vecs is None:
            np.save(os.path.join(store_dir, "vecs.npy"), np.zeros((0, self.dim), dtype="float32"))
        else:
            np.save(os.path.join(store_dir, "vecs.npy"), self.vecs)
        # payloads
        with open(os.path.join(store_dir, "payloads.jsonl"), "w") as f:
            for t in self.payloads:
                f.write(json.dumps({"text": t}) + "\n")

    @classmethod
    def load(cls, store_dir: str, dim: int = 256) -> "HybridIndex":
        obj = cls(dim=dim)
        try:
            with open(os.path.join(store_dir, "meta.json")) as f:
                meta = json.load(f)
            obj.dim = int(meta.get("dim", dim))
            obj.labels = list(meta.get("labels", []))
            obj.vecs = np.load(os.path.join(store_dir, "vecs.npy"))
            obj.payloads = []
            pth = os.path.join(store_dir, "payloads.jsonl")
            if os.path.exists(pth):
                with open(pth) as f:
                    for line in f:
                        if not line.strip(): continue
                        obj.payloads.append(json.loads(line)["text"])
            obj._rebuild_faiss()
        except Exception:
            # fresh empty
            obj.vecs = None
            obj.labels = []
            obj.payloads = []
            obj._faiss = None
        return obj

    # ---------- faiss helper ----------
    def _rebuild_faiss(self):
        if not _FAISS_AVAILABLE:
            self._faiss = None
            return
        self._faiss = faiss.IndexFlatL2(self.dim)
        if self.vecs is not None and len(self.vecs) > 0:
            self._faiss.add(self.vecs.astype("float32"))

    # ---------- core ops ----------
    def add(self, texts: List[str], labels: List[str]) -> List[int]:
        if not texts: return []
        new_vecs = np.stack([_embed(t, self.dim) for t in texts]).astype("float32")
        if self.vecs is None or len(self.vecs) == 0:
            self.vecs = new_vecs
        else:
            self.vecs = np.vstack([self.vecs, new_vecs])

        ids = list(range(len(self.payloads), len(self.payloads) + len(texts)))
        self.payloads.extend(texts)
        self.labels.extend(labels)

        if _FAISS_AVAILABLE:
            if self._faiss is None:
                self._rebuild_faiss()
            self._faiss.add(new_vecs)
        return ids

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, str, str]]:
        if self.vecs is None or len(self.vecs) == 0:
            return []
        q = _embed(query, self.dim).astype("float32")[None, :]
        if _FAISS_AVAILABLE and self._faiss is not None:
            D, I = self._faiss.search(q, k)
            out = []
            for d, i in zip(D[0], I[0]):
                if i == -1: continue
                out.append((int(i), float(d), self.payloads[i], self.labels[i]))
            return out
        # fallback cosine
        V = self.vecs
        num = V @ q[0]
        den = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q[0]) or 1.0)) + 1e-9
        sims = num / den
        dists = 2.0 - 2.0 * sims  # mimic L2-like
        idxs = np.argsort(dists)[:k]
        return [(int(i), float(dists[i]), self.payloads[int(i)], self.labels[int(i)]) for i in idxs]

    def similarity_to_label(self, label: str, text: str) -> float:
        """
        Cosine similarity to centroid of all items with given label.
        """
        if self.vecs is None or len(self.vecs) == 0: return 0.0
        mask = np.array([1 if l == label else 0 for l in self.labels], dtype=bool)
        if not mask.any(): return 0.0
        centroid = self.vecs[mask].mean(axis=0)
        cn = np.linalg.norm(centroid); 
        if cn == 0: return 0.0
        centroid = centroid / cn
        v = _embed(text, self.dim)
        vn = np.linalg.norm(v); 
        if vn == 0: return 0.0
        v = v / vn
        return float(v @ centroid)
