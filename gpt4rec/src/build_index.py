import argparse, os, pickle, pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

def tokenize(text):
    return re.findall(r"\w+", str(text).lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    cat = pd.read_csv(os.path.join(args.data_dir, "catalog.csv"))
    titles = cat["title"].fillna("").tolist()
    tokenized = [tokenize(t) for t in titles]
    bm25 = BM25Okapi(tokenized)

    obj = {
        "bm25": bm25,
        "catalog": cat,
    }
    with open(args.out_path, "wb") as f:
        pickle.dump(obj, f)

    print("BM25 index written to", args.out_path)

if __name__ == "__main__":
    main()
