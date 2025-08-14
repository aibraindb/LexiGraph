#!/usr/bin/env python3
import argparse, urllib.request, os

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True, help="URL to TTL file")
parser.add_argument("--out", default="lexigraph/data/fibo_full.ttl")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
print("Downloading", args.url, "→", args.out)
urllib.request.urlretrieve(args.url, args.out)
print("Done.")
