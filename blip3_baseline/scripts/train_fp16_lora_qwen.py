import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch, torch.nn as nn

QWEN_LORA_TARGETS = ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]

def safe_chat_template(tok, msgs, add_generation_prompt):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    lines=[]
    for m in msgs:
        role=m.get("role","user").title()
        c=m.get("content","")
        if isinstance(c,list):
            parts=[]
            for blk in c:
                if isinstance(blk,dict) and blk.get("type")=="text": parts.append(blk.get("text",""))
                elif isinstance(blk,str): parts.append(blk)
            c="\n".join(parts)
        lines.append(f"{role}: {c}")
    if add_generation_prompt: lines.append("Assistant:")
    return "\n".join(lines)

def split_conv(tokenizer, ex):
    sys = ex.get("system","")
    msgs = ex.get("messages", [])
    conv=[]
    if sys: conv.append({"role":"system","content":sys})
    for m in msgs:
        parts=[]
        for blk in m.get("content",[]) or []:
            if isinstance(blk, dict) and blk.get("type")=="text": parts.append(blk.get("text",""))
            elif isinstance(blk, str): parts.append(blk)
        conv.append({"role":m.get("role","user"),"content":"\n".join(parts)})
    prompt_msgs = conv[:-1] if conv and conv[-1]["role"]=="assistant" else conv
    prompt = safe_chat_template(tokenizer, prompt_msgs, True)
    full   = safe_chat_template(tokenizer, conv, False)
    return {"prompt_only": prompt, "text_full": full}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data",  default="data/train_blip_aug_2.jsonl")
    ap.add_argument("--out",   default="outputs/lora-qwen3b-aug-fp16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1536)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: split_conv(tok, ex), remove_columns=ds.column_names)

    # Base in FP16 for memory
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    model.config.use_cache = False

    # LoRA attach
    lora = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=QWEN_LORA_TARGETS
    )
    model = get_peft_model(model, lora)

    # >>> Cast only trainable (LoRA) params to FP32 to avoid GradScaler issues
    for n,p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # Sanity: some params must be trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}% trainable)")
    if trainable == 0:
        seen=set()
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                seen.add(n.split(".")[-1])
        print("[ERROR] No trainable params. Check QWEN_LORA_TARGETS. Linear names hint:", sorted(seen)[:50])
        raise SystemExit(1)

    def collate(batch):
        p = [b["prompt_only"] for b in batch]
        f = [b["text_full"]    for b in batch]
        tp = tok(p, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        tf = tok(f, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        input_ids, attn = tf["input_ids"], tf["attention_mask"]
        labels = input_ids.clone()
        pad = tok.pad_token_id or 0
        for i in range(len(batch)):
            prompt_len = (tp["input_ids"][i] != pad).sum().item()
            labels[i,:prompt_len] = -100
        return {"input_ids":input_ids, "attention_mask":attn, "labels":labels}

    # Disable AMP (no GradScaler), keep base FP16 + LoRA FP32
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_steps=200, save_total_limit=2,
        fp16=False, bf16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print("Saved LoRA to", args.out)

if __name__ == "__main__":
    main()
