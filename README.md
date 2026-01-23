# GPT4Rec (Repro) — Minimal Skeleton

A **minimal, training-first** scaffold to reproduce the GPT4Rec pipeline:

1) **Query Generation**: Fine-tune GPT‑2 on formatted user sequences to generate **multi‑query** strings.
2) **Retrieval**: Build a BM25 index over item titles; search each generated query.
3) **Ranking Merge**: Combine per‑query results (K/m non‑duplicates) to balance relevance & diversity.
4) **Eval**: Recall@K, Diversity@K (Jaccard on cate/brand), Coverage@K.

> This skeleton follows the paper's setup and keeps the code compact. Bring your own datasets (Amazon 5‑core Beauty/Electronics).

## Quickstart

```bash
# 새 환경 만들기
conda create -n sdbs4rec python=3.10 -y
conda activate sdbs4rec

# PyTorch (CUDA 12.1)
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# 1) Convert Amazon 5-core JSON to sequences + catalog
python gpt4rec/src/data_prep.py --raw_path /path/to/Beauty_5.json.gz --out_dir gpt4rec/data/beauty

# 2) Fine-tune GPT-2 (causal LM) on formatted prompts
python gpt4rec/src/train_lm.py --data_dir gpt4rec/data/beauty --out_dir gpt4rec/outputs/beauty_gpt2

# 3) Build BM25 index over item titles
python gpt4rec/src/build_index.py --data_dir gpt4rec/data/beauty --out_path gpt4rec/outputs/beauty_bm25.pkl

# 4) Generate multi-queries & retrieve top‑K
python gpt4rec/src/infer_retrieve.py --data_dir gpt4rec/data/beauty   --lm_dir gpt4rec/outputs/beauty_gpt2 --bm25_path gpt4rec/outputs/beauty_bm25.pkl   --beam 20 --num_queries 20 --K 40 --out_path gpt4rec/outputs/beauty_preds.pkl

# 5) Evaluate
python gpt4rec/src/eval/evaluate.py --data_dir gpt4rec/data/beauty   --pred_path gpt4rec/outputs/beauty_preds.pkl --K_list 5 10 20 40
```

## Data format

After `data_prep.py`, you will have:

- `catalog.csv` with columns: `item_id,title,category,brand`
- `user_sequences.jsonl`: per line `{"user_id": "...","items": ["i1","i2",...], "split": "train|val|test"}`
- `train.txt`, `val.txt`, `test.txt`: **prompted** training strings used for LM.

## Prompt (per paper)

```
Previously, the customer has bought:
<TITLE 1>. <TITLE 2> ...

In the future, the customer wants to buy
```

During fine‑tuning, the **target** is the *title of the next item* appended after the prompt. At inference, we stop before appending target and let GPT‑2 **generate queries**.

## Notes

- Defaults roughly match the paper: GPT‑2 (117M), 20 epochs, lr=1e‑4, warmup=2000, AdamW w/ weight decay.
- BM25 uses `rank_bm25.BM25Okapi`; tune `k1` and `b` via grid search in `build_index.py` if desired.
- Merge policy: take `K/m` items per query in descending BM25 score order, skipping duplicates.

## License
For research purposes only.

---

토이데이터

# 1) BM25 인덱스 (토이 카탈로그)
python gpt4rec/src/build_index.py \
  --data_dir gpt4rec/data/toy \
  --out_path gpt4rec/outputs/toy_bm25.pkl

# 2) (선택) 빠른 파인튜닝: 에폭 1~2로만 가볍게
python gpt4rec/src/train_lm.py \
  --data_dir gpt4rec/data/toy \
  --out_dir gpt4rec/outputs/toy_gpt2

# 3) 멀티쿼리 생성 + 검색 + 병합
python gpt4rec/src/infer_retrieve.py \
  --data_dir gpt4rec/data/toy \
  --lm_dir gpt4rec/outputs/toy_gpt2 \
  --bm25_path gpt4rec/outputs/toy_bm25.pkl \
  --beam 10 --num_queries 10 --K 10 \
  --out_path gpt4rec/outputs/toy_preds.pkl

# 4) 평가 (Diversity/Coverage 지표 확인)
python gpt4rec/src/eval/evaluate.py \
  --data_dir gpt4rec/data/toy \
  --pred_path gpt4rec/outputs/toy_preds.pkl \
  --K_list 5 10