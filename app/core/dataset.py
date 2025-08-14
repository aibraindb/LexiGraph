from __future__ import annotations
import os, os.path as p, json, uuid
from typing import Optional

DATASET_DIR = "data/dataset"
LABELED_DIR = "data/labeled"

def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(LABELED_DIR, exist_ok=True)

def write_text_copy(root: str, variant_id: str, text: str) -> str:
    os.makedirs(p.join(root, variant_id), exist_ok=True)
    fname = f"{uuid.uuid4().hex}.txt"
    fpath = p.join(root, variant_id, fname)
    with open(fpath, "w") as f:
        f.write(text)
    return fpath

def append_metadata(variant_id: str, source_name: str, chars: int, meta_extra: Optional[dict] = None):
    meta_path = p.join(DATASET_DIR, "metadata.jsonl")
    rec = {"variant_id": variant_id, "source_name": source_name, "chars": chars}
    if meta_extra: rec.update(meta_extra)
    with open(meta_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
