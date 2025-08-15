
import json, pickle, numpy as np
from pathlib import Path
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class SimpleVectorStore:
    def __init__(self, dirpath: str):
        self.dir = Path(dirpath); self.dir.mkdir(parents=True, exist_ok=True)
        self.vecs_path = self.dir/'vecs.npy'
        self.meta_path = self.dir/'meta.json'
        self.vectorizer_path = self.dir/'vectorizer.pkl'
        self._X=None; self._meta=[]; self._faiss=None; self._nn=None; self._vectorizer=None
        self._load()
    def _load(self):
        if self.vectorizer_path.exists() and self.vecs_path.exists() and self.meta_path.exists():
            with open(self.vectorizer_path,'rb') as f: self._vectorizer = pickle.load(f)
            self._X = np.load(self.vecs_path)
            self._meta = json.loads(self.meta_path.read_text())
            self._build()
        else:
            self._vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    def _build(self):
        if self._X is None or len(self._X)==0: self._faiss=None; self._nn=None; return
        if HAVE_FAISS:
            dim = self._X.shape[1]; self._faiss = faiss.IndexFlatIP(dim)
            Xn = self._normalize(self._X.astype('float32')); self._faiss.add(Xn)
        else:
            self._nn = NearestNeighbors(metric='cosine', n_neighbors=min(8,len(self._meta))).fit(self._X)
    def _normalize(self, x):
        n = np.linalg.norm(x, axis=1, keepdims=True)+1e-9; return x/n
    def add(self, text: str, doc_id: str, fibo_class_uri: str|None=None):
        corpus = [m['text'] for m in self._meta]+[text]
        self._vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        X = self._vectorizer.fit_transform(corpus).astype('float32').toarray()
        self._X = X; self._meta.append({'doc_id':doc_id,'text':text,'fibo_class_uri':fibo_class_uri})
        with open(self.vectorizer_path,'wb') as f: pickle.dump(self._vectorizer,f)
        np.save(self.vecs_path, self._X.astype('float32')); self.meta_path.write_text(json.dumps(self._meta, indent=2))
        self._build()
    def update_label(self, doc_id: str, fibo_class_uri: str):
        for m in self._meta:
            if m['doc_id']==doc_id: m['fibo_class_uri']=fibo_class_uri
        self.meta_path.write_text(json.dumps(self._meta, indent=2))
    def search(self, text: str, topk: int=8):
        if self._X is None or (self._faiss is None and self._nn is None): return []
        q = self._vectorizer.transform([text]).astype('float32').toarray()
        if HAVE_FAISS:
            D,I = self._faiss.search(self._normalize(q), min(topk,len(self._meta)))
            sims = D[0].tolist(); idxs = I[0].tolist()
        else:
            dists, idxs = self._nn.kneighbors(q, n_neighbors=min(topk,len(self._meta)))
            sims = [1-d for d in dists[0]]
        out=[]; 
        for sim,idx in zip(sims,idxs):
            if idx<0 or idx>=len(self._meta): continue
            m=self._meta[idx]; out.append({'doc_id':m['doc_id'],'sim':float(sim),'fibo_class_uri':m.get('fibo_class_uri')})
        return out
    def suggest_class(self, doc_id: str, topk_neighbors: int=8):
        m = next((m for m in self._meta if m['doc_id']==doc_id), None)
        if not m: return {'suggestion':None,'votes':{}}
        nbrs = self.search(m['text'], topk_neighbors)
        votes={}
        for n in nbrs:
            uri = n.get('fibo_class_uri')
            if not uri: continue
            votes[uri]=votes.get(uri,0.0)+float(n['sim'])
        if not votes: return {'suggestion':None,'votes':{}}
        suggestion = max(votes.items(), key=lambda x:x[1])[0]
        return {'suggestion':suggestion,'votes':votes}
