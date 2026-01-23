import argparse, gzip, json, os, random, pandas as pd
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def iter_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_path", required=True, help="Path to Amazon 5-core JSON.gz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    cfg = load_config()
    max_seq_len = cfg["dataset"]["max_seq_len"]
    min_u = cfg["dataset"]["min_user_interactions"]
    split_fracs = cfg["dataset"]["splits"]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Reading {args.raw_path} ...")
    users = defaultdict(list)
    items_meta = {}

    for rec in tqdm(iter_json_gz(args.raw_path)):
        # Amazon format keys vary; handle common ones
        uid = rec.get("reviewerID") or rec.get("user_id")
        iid = rec.get("asin") or rec.get("item_id")
        title = rec.get("title") or rec.get("summary") or rec.get("item_title")
        cate = None
        brand = rec.get("brand")
        if "category" in rec:
            cate = rec["category"]
        elif "categories" in rec and isinstance(rec["categories"], list) and rec["categories"]:
            # categories is nested list sometimes
            cate = rec["categories"][0][-1] if rec["categories"][0] else None
        if not (uid and iid and title):
            continue
        users[uid].append((iid, title))
        if iid not in items_meta:
            items_meta[iid] = {"item_id": iid, "title": title, "category": cate, "brand": brand}

    # dedup by order, truncate
    seqs = {}
    for u, lst in users.items():
        seen, seq = set(), []
        for iid, title in lst:
            if iid in seen: continue
            seen.add(iid); seq.append(iid)
        if len(seq) >= min_u:
            seqs[u] = seq[:max_seq_len]

    # train/val/test per user by proportion on sequence
    triples = []
    for u, seq in seqs.items():
        n = len(seq)
        t_idx = int(n * split_fracs[0])
        v_idx = t_idx + int(n * split_fracs[1])
        # At least 3 elements guard
        if n < 3: continue
        triples.append((u, seq[:t_idx], seq[t_idx:v_idx], seq[v_idx:]))

    # write catalog
    cat_df = pd.DataFrame(list(items_meta.values()))
    cat_df.to_csv(os.path.join(args.out_dir, "catalog.csv"), index=False)

    # write user_sequences.jsonl and LM txts
    prompt_pre = cfg["prompt"]["preamble"].strip()
    prompt_post = cfg["prompt"]["postamble"].strip()

    train_lines, val_lines, test_lines = [], [], []
    seq_out = open(os.path.join(args.out_dir, "user_sequences.jsonl"), "w", encoding="utf-8")
    for u, tr, va, te in triples:
        json.dump({"user_id": u, "items": tr + va + te, "split": "all"}, seq_out); seq_out.write("\n")
        # For LM training: use (prefix -> next_title) pairs from within train portion
        # Map item ids to titles
        def to_titles(ids): return [items_meta[i]["title"] for i in ids if i in items_meta and items_meta[i]["title"]]
        # create rolling windows in train portion
        titles = to_titles(tr)
        for t in range(1, len(titles)):
            hist = titles[:t]
            target = titles[t]
            line = f"{prompt_pre}\n" + " ".join(f"{h}." for h in hist) + f"\n\n{prompt_post}\n{target}"
            train_lines.append(line)
        # validation/test target: the next item after full (train+val) / full (train+val+test[:-1])
        if va:
            hist = to_titles(tr + va)
            if len(hist) >= 1:
                val_lines.append(f"{prompt_pre}\n" + " ".join(f"{h}." for h in hist) + f"\n\n{prompt_post}\n")
        if te:
            hist = to_titles(tr + va + te[:-1]) if len(te) > 1 else to_titles(tr + va)
            if len(hist) >= 1:
                test_lines.append(f"{prompt_pre}\n" + " ".join(f"{h}." for h in hist) + f"\n\n{prompt_post}\n")

    seq_out.close()

    for name, lines in [("train.txt", train_lines), ("val.txt", val_lines), ("test.txt", test_lines)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line.strip() + "\n")

    print("Done. Files written to", args.out_dir)

if __name__ == "__main__":
    main()
