import argparse, os, yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def load_config():
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_name", default=None, help="Override model name")
    args = ap.parse_args()

    cfg = load_config()
    model_name = args.model_name or cfg["training"]["model_name"]
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {
        "train": os.path.join(args.data_dir, "train.txt"),
        "validation": os.path.join(args.data_dir, "val.txt"),
    }
    raw = load_dataset("text", data_files=data_files)

    def tok_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512)

    tok = raw.map(tok_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(model_name)

    tr_cfg = cfg["training"]
    args_tr = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tr_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        num_train_epochs=tr_cfg["epochs"],
        learning_rate=float(tr_cfg["lr"]),
        weight_decay=float(tr_cfg["weight_decay"]),
        warmup_steps=tr_cfg["warmup_steps"],
        fp16=tr_cfg.get("fp16", False),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tok["train"],
        eval_dataset=tok["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("Saved fine-tuned LM to", args.out_dir)

if __name__ == "__main__":
    main()
