from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os, json
import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

def _embed(text: str, dim: int = 256) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.normal(size=dim).astype('float32')
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)

class HybridIndex:
    def __init__(self, dim: int = 256):
        self.dim = dim
        self.vecs: Optional[np.ndarray] = None
        self.labels: List[str] = []
        self.payloads: List[str] = []
        self.ids: List[str] = []
        self._faiss = None
        self._last_added_ids: List[str] = []

    def save(self, store_dir: str):
        os.makedirs(store_dir, exist_ok=True)
        with open(os.path.join(store_dir, 'meta.json'),'w') as f:
            json.dump({'dim': self.dim, 'labels': self.labels, 'ids': self.ids}, f)
        np.save(os.path.join(store_dir, 'vecs.npy'), self.vecs if self.vecs is not None else np.zeros((0,self.dim),dtype='float32'))
        with open(os.path.join(store_dir, 'payloads.jsonl'),'w') as f:
            for t in self.payloads: f.write(json.dumps({'text':t})+'\n')

    @classmethod
    def load(cls, store_dir: str, dim: int = 256) -> 'HybridIndex':
        obj = cls(dim=dim)
        try:
            with open(os.path.join(store_dir,'meta.json')) as f:
                meta = json.load(f)
            obj.dim = int(meta.get('dim', dim))
            obj.labels = list(meta.get('labels', []))
            obj.ids = list(meta.get('ids', []))
            obj.vecs = np.load(os.path.join(store_dir,'vecs.npy'))
            obj.payloads = []
            pth = os.path.join(store_dir,'payloads.jsonl')
            if os.path.exists(pth):
                with open(pth) as f:
                    for line in f:
                        if line.strip(): obj.payloads.append(json.loads(line)['text'])
            obj._rebuild_faiss()
        except Exception:
            obj.vecs = None; obj.labels = []; obj.payloads = []; obj._faiss = None; obj.ids = []
        return obj

    def _rebuild_faiss(self):
        if not _FAISS_AVAILABLE:
            self._faiss = None; return
        self._faiss = faiss.IndexFlatL2(self.dim)
        if self.vecs is not None and len(self.vecs)>0:
            self._faiss.add(self.vecs.astype('float32'))

    def add(self, texts: List[str], labels: List[str]) -> List[str]:
        if not texts: return []
        import uuid as _uuid
        new_vecs = np.stack([_embed(t, self.dim) for t in texts]).astype('float32')
        if self.vecs is None or len(self.vecs)==0:
            self.vecs = new_vecs
        else:
            self.vecs = np.vstack([self.vecs, new_vecs])
        new_ids = [_uuid.uuid4().hex for _ in texts]
        self.payloads.extend(texts); self.labels.extend(labels); self.ids.extend(new_ids)
        self._last_added_ids = list(new_ids)
        if _FAISS_AVAILABLE:
            if self._faiss is None: self._rebuild_faiss()
            self._faiss.add(new_vecs)
        return new_ids

    def search(self, query: str, k: int = 5):
        if self.vecs is None or len(self.vecs)==0: return []
        q = _embed(query, self.dim).astype('float32')[None,:]
        if _FAISS_AVAILABLE and self._faiss is not None:
            D,I = self._faiss.search(q, k)
            out = []
            for d,i in zip(D[0],I[0]):
                if i==-1: continue
                out.append((int(i), float(d), self.payloads[i], self.labels[i], self.ids[i]))
            return out
        V = self.vecs
        num = V @ q[0]
        den = (np.linalg.norm(V,axis=1) * (np.linalg.norm(q[0]) or 1.0)) + 1e-9
        sims = num/den
        dists = 2.0 - 2.0*sims
        idxs = np.argsort(dists)[:k]
        return [(int(i), float(dists[i]), self.payloads[int(i)], self.labels[int(i)], self.ids[int(i)]) for i in idxs]

    def similarity_to_label(self, label: str, text: str) -> float:
        if self.vecs is None or len(self.vecs)==0: return 0.0
        import numpy as np
        mask = np.array([1 if l==label else 0 for l in self.labels], dtype=bool)
        if not mask.any(): return 0.0
        centroid = self.vecs[mask].mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn==0: return 0.0
        centroid = centroid / cn
        v = _embed(text, self.dim)
        vn = np.linalg.norm(v)
        if vn==0: return 0.0
        v = v / vn
        return float(v @ centroid)

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self.ids: return False
        import numpy as np
        i = self.ids.index(entry_id)
        self.ids.pop(i); self.labels.pop(i); self.payloads.pop(i)
        if self.vecs is not None and len(self.vecs)>0:
            self.vecs = np.delete(self.vecs, i, axis=0)
        self._rebuild_faiss()
        return True

    def list_entries(self, label: str | None = None, limit: int = 100):
        rows=[]
        for i,(eid,lbl) in enumerate(zip(self.ids, self.labels)):
            if label and lbl!=label: continue
            rows.append({"id":eid,"label":lbl,"chars":len(self.payloads[i])})
            if len(rows)>=limit: break
        return rows

    def undo_last_add(self) -> list[str]:
        undone = []
        for eid in list(self._last_added_ids):
            if self.delete(eid):
                undone.append(eid)
        self._last_added_ids = []
        return undone

class FaissIndex(HybridIndex):
    pass
