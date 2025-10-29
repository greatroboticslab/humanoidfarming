#!/usr/bin/env python3
import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def render_chat(tokenizer, example):
    system = example.get("system", "")
    msgs = example["messages"]
    conv = []
    if system:
        conv.append({"role":"system","content":system})
    for m in msgs:
        parts = []
        for blk in m.get("content", []):
            if blk.get("type") == "text":
                parts.append(blk.get("text",""))
        conv.append({"role": m["role"], "content": "\n".join(parts)})
    try:
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    except Exception:
        prompt = ""
        for m in conv:
            tag = "User" if m["role"]=="user" else ("Assistant" if m["role"]=="assistant" else "System")
            prompt += f"{tag}: {m['content']}\n"
    return {"text": prompt}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/train_blip.jsonl")
    ap.add_argument("--out", default="outputs/lora-qwen3b-blipchat")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--quant", choices=["4bit","8bit","fp16"], default="4bit",
                    help="Weight loading mode: 4bit/8bit use bitsandbytes; fp16 needs no bnb.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: render_chat(tok, ex), remove_columns=ds.column_names)

    # Load base model per quant choice
    if args.quant in ("4bit", "8bit"):
        load_kwargs = dict(device_map="auto", trust_remote_code=True)
        if args.quant == "4bit":
            load_kwargs["load_in_4bit"] = True
        else:
            load_kwargs["load_in_8bit"] = True
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model = prepare_model_for_kbit_training(model)
    else:
        # fp16 path: no bitsandbytes required
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
        )

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)

    def collate(batch):
        texts = [b["text"] for b in batch]
        toks = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        toks["labels"] = toks["input_ids"].clone()
        return toks

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saved LoRA to", args.out)

if __name__ == "__main__":
    main()
