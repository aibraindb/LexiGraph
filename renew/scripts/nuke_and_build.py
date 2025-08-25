import shutil, json
from pathlib import Path

ROOT = Path(".")
for p in ["data/ocr_cache"]:
    d = ROOT / p
    if d.exists():
        shutil.rmtree(d)
        print("Removed", d)
    d.mkdir(parents=True, exist_ok=True)
    print("Created", d)

print("Done.")
