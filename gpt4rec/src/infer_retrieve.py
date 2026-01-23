import argparse, os, pickle, pandas as pd, yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import re
import json
import math

def tokenize_simple(text):
    return re.findall(r"\w+", str(text).lower())

def load_config():
    import pathlib
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def generate_queries(prompt, model, tokenizer, num_queries=20, beam=20, max_new_tokens=24, temperature=0.9, top_p=0.95):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **input_ids,
            num_beams=beam,
            num_return_sequences=num_queries,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    # Extract only the generated suffix after the prompt:
    base = tokenizer.decode(input_ids["input_ids"][0], skip_special_tokens=True)
    queries = []
    for t in texts:
        q = t[len(base):].strip()
        q = q.split("\n")[0].strip()  # stop at newline if any
        if len(q) > 0:
            queries.append(q)
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for q in queries:
        if q not in seen:
            uniq.append(q); seen.add(q)
    return uniq

def rank_bm25(bm25, catalog_df, query, topn=100):
    toks = tokenize_simple(query)
    scores = bm25.get_scores(toks)
    idx = np.argsort(-scores)[:topn]
    return catalog_df.iloc[idx].assign(score=scores[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--lm_dir", required=True)
    ap.add_argument("--bm25_path", required=True)
    ap.add_argument("--num_queries", type=int, default=None)
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    cfg = load_config()
    nq = args.num_queries or cfg["inference"]["num_queries"]
    beam = args.beam or cfg["inference"]["beam_size"]
    K = args.K or cfg["retrieval"]["K"]
    max_new = cfg["inference"]["max_gen_tokens"]

    # Load LM
    tokenizer = AutoTokenizer.from_pretrained(args.lm_dir)
    model = AutoModelForCausalLM.from_pretrained(args.lm_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()

    # Load BM25
    with open(args.bm25_path, "rb") as f:
        obj = pickle.load(f)
    bm25 = obj["bm25"]; catalog = obj["catalog"]

    # Load test prompts
    test_path = os.path.join(args.data_dir, "test.txt")
    with open(test_path, "r", encoding="utf-8") as f:
        test_prompts = [line.strip() for line in f if line.strip()]

    merged_preds = []
    step = max(1, K // max(1, nq))
    for prompt in tqdm(test_prompts, desc="infer+retrieve"):
        queries = generate_queries(
            prompt, model, tokenizer,
            num_queries=nq, beam=beam, max_new_tokens=max_new,
            temperature=cfg["inference"]["temperature"], top_p=cfg["inference"]["top_p"]
        )
        # Search each query, gather K/m non-duplicates
        pool = []
        for q in queries:
            df = rank_bm25(bm25, catalog, q, topn=max(step*5, 50))
            pool.append((q, df))

        taken_ids = set()
        ranked = []
        # Take K/m per query in order
        for q, df in pool:
            count = 0
            for _, row in df.iterrows():
                iid = row["item_id"]
                if iid in taken_ids: continue
                ranked.append({"query": q, "item_id": iid, "score": row["score"]})
                taken_ids.add(iid)
                count += 1
                if count >= step: break
            if len(ranked) >= K:
                break

        merged_preds.append(ranked[:K])

    import pickle as pkl
    with open(args.out_path, "wb") as f:
        pkl.dump(merged_preds, f)
    print("Predictions written to", args.out_path)

if __name__ == "__main__":
    main()
