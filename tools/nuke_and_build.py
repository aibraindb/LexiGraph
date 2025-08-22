import shutil, os, glob, subprocess, sys
keep = ["data/fibo_full.ttl"]
for d in ["data/sessions","data/vectors","data/ocr"]:
    if os.path.isdir(d):
        shutil.rmtree(d)
for k in keep:
    if os.path.exists(k):
        print("Keeping", k)
os.makedirs("data/sessions", exist_ok=True)
os.makedirs("data/vectors", exist_ok=True)
os.makedirs("data/ocr", exist_ok=True)
print("Cleaned. Now build as needed:")
print("  python tools/build_fibo_index.py --ttl data/fibo_full.ttl")
