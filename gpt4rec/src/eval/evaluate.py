import argparse, os, json, pickle, pandas as pd, numpy as np
from collections import defaultdict
from tqdm import tqdm

def jaccard(a, b):
    if not a or not b: return 0.0
    a, b = set(a), set(b)
    inter = len(a & b); union = len(a | b)
    return inter / union if union else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred_path", required=True)
    ap.add_argument("--K_list", type=int, nargs="+", default=[5,10,20,40])
    args = ap.parse_args()

    # Load catalog
    cat = pd.read_csv(os.path.join(args.data_dir, "catalog.csv"))
    id2cate = dict(zip(cat["item_id"], cat["category"]))
    id2brand = dict(zip(cat["item_id"], cat["brand"]))
    id2title = dict(zip(cat["item_id"], cat["title"]))

    # Build gold targets from user_sequences: last item in each user's sequence is the test target
    # But we don't have direct mapping prompt->target here. As a proxy, we evaluate Diversity/Coverage only.
    # For Recall@K you will typically log the actual target per prompt when building test prompts.
    # Here we provide a placeholder calculation: Recall is unknown => set to NaN.
    with open(args.pred_path, "rb") as f:
        preds = pickle.load(f)

    results = []
    for K in args.K_list:
        # Diversity@K: 1 - average pairwise Jaccard (within top-K items) over cate/brand
        divs_cate, divs_brand = [], []
        covs_cate, covs_brand = [], []  # need user-side coverage; approximated as unique within-topK ratio
        for ranked in preds:
            topk = ranked[:K]
            items = [r["item_id"] for r in topk]
            cates = [id2cate.get(i) for i in items if pd.notna(id2cate.get(i))]
            brands = [id2brand.get(i) for i in items if pd.notna(id2brand.get(i))]

            # pairwise jaccard for single-label categories reduces to proportion of equal/not equal
            if len(cates) > 1:
                equal_pairs = 0; total_pairs = 0
                for i in range(len(cates)):
                    for j in range(i+1, len(cates)):
                        total_pairs += 1
                        equal_pairs += int(cates[i] == cates[j])
                sim = equal_pairs / total_pairs if total_pairs else 0.0
                divs_cate.append(1.0 - sim)
            if len(brands) > 1:
                equal_pairs = 0; total_pairs = 0
                for i in range(len(brands)):
                    for j in range(i+1, len(brands)):
                        total_pairs += 1
                        equal_pairs += int(brands[i] == brands[j])
                sim = equal_pairs / total_pairs if total_pairs else 0.0
                divs_brand.append(1.0 - sim)

            # Approx coverage proxy: unique labels / K
            covs_cate.append(len(set(cates)) / max(1, len(cates)) if cates else 0.0)
            covs_brand.append(len(set(brands)) / max(1, len(brands)) if brands else 0.0)

        results.append({
            "K": K,
            "Recall@K": np.nan,  # placeholder; see README for proper target logging
            "Diversity@K(category)": float(np.mean(divs_cate)) if divs_cate else np.nan,
            "Diversity@K(brand)": float(np.mean(divs_brand)) if divs_brand else np.nan,
            "Coverage@K(category)": float(np.mean(covs_cate)) if covs_cate else np.nan,
            "Coverage@K(brand)": float(np.mean(covs_brand)) if covs_brand else np.nan,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    out_csv = os.path.splitext(args.pred_path)[0] + f".eval.csv"
    df.to_csv(out_csv, index=False)
    print("Saved evaluation to", out_csv)

if __name__ == "__main__":
    main()
