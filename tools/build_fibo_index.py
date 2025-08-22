import argparse
from app.core.fibo_index import build_fibo_vectors

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ttl", required=True, help="Path to fibo_full.ttl")
    p.add_argument("--name", default="fibo", help="Vector store name")
    args = p.parse_args()
    info = build_fibo_vectors(args.ttl, name=args.name)
    print(info)
