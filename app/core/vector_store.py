from __future__ import annotations
import os, json, joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data/vectors"
os.makedirs(DATA_DIR, exist_ok=True)

def _save(name: str, vec: TfidfVectorizer, X):
    joblib.dump(vec, os.path.join(DATA_DIR, f"{name}.vectorizer.joblib"))
    np.save(os.path.join(DATA_DIR, f"{name}.npy"), X.astype("float32"))

def _load(name: str):
    vec = joblib.load(os.path.join(DATA_DIR, f"{name}.vectorizer.joblib"))
    X = np.load(os.path.join(DATA_DIR, f"{name}.npy"))
    return vec, X

def build_from_texts(name: str, texts: list[str]):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(texts).astype("float32").toarray()
    _save(name, vec, X)
    return {"name": name, "size": len(texts), "dim": X.shape[1]}

def search(name: str, query: str, top_k=5):
    vec, X = _load(name)
    q = vec.transform([query]).astype("float32").toarray()[0]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    qn = q / (np.linalg.norm(q) + 1e-8)
    sims = (Xn @ qn)
    idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in idx]
