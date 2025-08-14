#!/usr/bin/env python3
import os, os.path as p, re, glob, collections, argparse, json

def toks(s): return re.sub(r"[^A-Za-z0-9]+"," ",s).lower().split()

def ngrams(tokens, n=1, m=3):
    bag=[]
    for k in range(n,m+1):
        bag += [" ".join(tokens[i:i+k]) for i in range(0,max(0,len(tokens)-k+1))]
    return bag

def load_variant_texts(root, vid):
    texts=[]
    d = p.join(root, vid)
    for f in glob.glob(p.join(d, "*.txt")):
        try: texts.append(open(f).read())
        except: pass
    return texts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="lexigraph/data/dataset")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--other", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    pos_docs = load_variant_texts(args.root, args.variant)
    neg_docs = []
    for o in args.other:
        neg_docs += load_variant_texts(args.root, o)

    pos_counts = collections.Counter()
    neg_counts = collections.Counter()
    for t in pos_docs: pos_counts.update(ngrams(toks(t),1,3))
    for t in neg_docs: neg_counts.update(ngrams(toks(t),1,3))

    candidates = []
    for g,cnt in pos_counts.items():
        if cnt < 2: 
            continue
        adv = cnt / (1 + neg_counts.get(g, 0))
        if adv >= 3:
            candidates.append((adv, cnt, g))
    candidates.sort(reverse=True)
    top = [g for _,_,g in candidates[:args.top]]

    inv = []
    for g,cnt in neg_counts.items():
        if cnt >= 3 and pos_counts.get(g,0) == 0:
            inv.append((cnt, g))
    inv.sort(reverse=True)

    print(json.dumps({
        "anchors_any_of": top[:12],
        "negative_anchors": [g for _,g in inv[:12]]
    }, indent=2))
