from __future__ import annotations
from typing import Dict, Tuple, Optional
import re
import numpy as np

from .models import OCRIndex

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9$%.:/\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1: a = a.reshape(1, -1)
    if b.ndim == 1: b = b.reshape(1, -1)
    na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-8
    nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-8
    sim = (a @ b.T) / (na * nb.T)
    return float(sim.max())

def find_best_line(ocr: OCRIndex, value: str) -> Optional[Tuple[int,int,int,float]]:
    """Return (page, block, line, score) for best matching line text."""
    target = _normalize(value)
    if not target: return None
    # Very simple TF count vector per line
    vocab = {}
    def vec(s: str):
        for tok in s.split():
            vocab.setdefault(tok, len(vocab))
        v = np.zeros(len(vocab), dtype=float)
        for tok in s.split():
            i = vocab[tok]; v[i]+=1.0
        return v

    # Build line vectors
    entries = []
    vecs = []
    for p_i, page in enumerate(ocr.pages, start=1):
        for b_i, block in enumerate(page.blocks, start=0):
            for l_i, line in enumerate(block.lines, start=0):
                t = _normalize(line.text)
                entries.append((p_i,b_i,l_i))
                vecs.append(vec(t))
    if not vecs:
        return None
    # Rebuild vectors now that vocab is final
    def vec_final(s: str):
        v = np.zeros(len(vocab), dtype=float)
        for tok in s.split():
            if tok in vocab: v[vocab[tok]] += 1.0
        return v
    X = np.vstack([vec_final(_normalize(line.text))
                    for page in ocr.pages
                    for block in page.blocks
                    for line in block.lines])
    q = vec_final(target)
    # cosine
    sims = (X @ q) / ((np.linalg.norm(X, axis=1)+1e-8) * (np.linalg.norm(q)+1e-8))
    j = int(np.argmax(sims))
    score = float(sims[j])
    return (*entries[j], score)
