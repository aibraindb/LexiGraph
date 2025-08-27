
import os, json, hashlib, time, requests
BASE=os.path.dirname(os.path.dirname(__file__)); OUT=os.path.join(BASE,"data","samples","wells")
MAN=os.path.join(BASE,"scripts","wf_manifest.json"); HASH=os.path.join(OUT,"_hashes.json")
os.makedirs(OUT, exist_ok=True)
def sha256_file(p):
    import hashlib
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for ch in iter(lambda: f.read(1<<20), b""): h.update(ch)
    return h.hexdigest()
def get(name,url):
    fn=f"{name}.pdf"; out=os.path.join(OUT,fn); print("-> GET", url)
    r=requests.get(url,timeout=60,allow_redirects=True); r.raise_for_status()
    with open(out,"wb") as f: f.write(r.content)
    h=sha256_file(out); print("   saved",fn,"sha256",h[:16],"...","size",len(r.content)//1024,"KB")
    return fn,h,len(r.content)
if __name__=="__main__":
    mani=json.load(open(MAN)); hashes={}
    for it in mani["docs"]:
        try:
            fn,h,sz=get(it["name"], it["url"]); hashes[fn]={"sha256":h,"url":it["url"],"bytes":sz,"ts":int(time.time())}
        except Exception as e:
            print("[error]", it["name"], e)
    json.dump(hashes, open(HASH,"w"), indent=2); print("Done ->", HASH)
