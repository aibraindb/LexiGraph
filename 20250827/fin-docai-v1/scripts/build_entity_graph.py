
import os, re, json, argparse, pdfplumber, networkx as nx
try:
    from pyvis.network import Network
except Exception:
    Network=None
AMOUNT=re.compile(r"\$?[-+]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?")
DATE=re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ACCT=re.compile(r"\b(?:Acct|Account|A/C)[:#\s]*[A-Za-z0-9-]{4,}\b", re.I)
ORG_HINT=re.compile(r"\b(Wells\s*Fargo|Advisors|Auto|Dealer|Customer|Merchant|Bank|LLC|Inc\.|Corporation|Corp\.)\b", re.I)
def extract_text(path):
    out=[]; 
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages: out.append(p.extract_text() or "")
    return "\n".join(out)
def ents(text):
    res=[]; 
    for m in AMOUNT.finditer(text): res.append(("Amount",m.group()))
    for m in DATE.finditer(text): res.append(("Date",m.group()))
    for m in ACCT.finditer(text): res.append(("AccountRef",m.group()))
    for m in ORG_HINT.finditer(text): res.append(("Org",m.group().strip()))
    return res
def build(files):
    G=nx.Graph()
    for f in files: G.add_node(f, kind="document")
    for f in files:
        for (t,v) in ents(extract_text(f)):
            en=f"{t}:{v}"
            if not G.has_node(en): G.add_node(en, kind="entity", etype=t)
            G.add_edge(f,en,weight=3 if t in ("AccountRef","Org") else 1)
    return G
def mst_docs(G):
    docs=[n for n,d in G.nodes(data=True) if d.get("kind")=="document"]; H=nx.Graph()
    for i,a in enumerate(docs):
        for b in docs[i+1:]:
            shared=set(G.neighbors(a)) & set(G.neighbors(b))
            if shared:
                w=sum(max(G[a][e].get("weight",1), G[b][e].get("weight",1)) for e in shared)
                H.add_edge(a,b,weight=w, distance=1.0/max(w,1))
    if H.number_of_edges()==0: return nx.Graph(), []
    T=nx.minimum_spanning_tree(H, weight="distance"); return T, list(T.edges(data=True))
def export_html(G, path):
    if Network is None: return False
    net=Network(height="700px", width="100%", bgcolor="#fff", font_color="#222")
    import os as _os
    for n,d in G.nodes(data=True):
        if d.get("kind")=="document": net.add_node(n,label=_os.path.basename(n),shape="box",color="#6aa9ff")
        else:
            lbl=n.split(":",1)[-1][:32]; color="#ffd166" if d.get("etype") in ("AccountRef","Org") else "#06d6a0"
            net.add_node(n,label=lbl,title=d.get("etype"),color=color)
    for a,b,d in G.edges(data=True): net.add_edge(a,b,value=d.get("weight",1))
    net.show(path); return True
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--folder", default=os.path.join("data","samples","wells"))
    ap.add_argument("--out", default=os.path.join("data","labels","entity_graph")); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files=[os.path.join(args.folder, fn) for fn in os.listdir(args.folder) if fn.lower().endswith(".pdf")]
    G=build(files); T, mst= mst_docs(G)
    data={"nodes":[{"id":n, **d} for n,d in G.nodes(data=True)], "edges":[{"a":a,"b":b, **d} for a,b,d in G.edges(data=True)], "mst_edges":[{"a":a,"b":b, **d} for a,b,d in mst]}
    jp=os.path.join(args.out,"entity_graph.json"); open(jp,"w").write(json.dumps(data, indent=2))
    hp=os.path.join(args.out,"entity_graph.html"); ok=export_html(G, hp)
    print("Wrote", jp); print("Wrote", hp if ok else "(install pyvis for HTML)")
