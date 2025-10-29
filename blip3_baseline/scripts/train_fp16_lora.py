import os, argparse, re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch, torch.nn as nn

def safe_chat_template(tok, msgs, add_generation_prompt):
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
    lines=[]; 
    for m in msgs:
        role=m.get("role","user").title()
        c=m.get("content","")
        if isinstance(c,list):
            parts=[]
            for blk in c:
                if isinstance(blk,dict) and blk.get("type")=="text":
                    parts.append(blk.get("text",""))
                elif isinstance(blk,str):
                    parts.append(blk)
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
            if isinstance(blk, dict) and blk.get("type")=="text":
                parts.append(blk.get("text",""))
            elif isinstance(blk, str):
                parts.append(blk)
        conv.append({"role":m.get("role","user"),"content":"\n".join(parts)})
    prompt_msgs = conv[:-1] if conv and conv[-1]["role"]=="assistant" else conv
    prompt = safe_chat_template(tokenizer, prompt_msgs, add_generation_prompt=True)
    full   = safe_chat_template(tokenizer, conv,         add_generation_prompt=False)
    return {"prompt_only": prompt, "text_full": full}

# ---- NEW: auto-detect target module names for LoRA ----
PREFERRED_PATTERNS = [
    "q_proj","k_proj","v_proj","o_proj",          # LLaMA-style
    "gate_proj","up_proj","down_proj",            # MLP (LLaMA)
    "c_attn","c_proj",                            # GPT-NeoX/OPT-like
    "qkv_proj","out_proj",                        # fused qkv patterns
    "W_pack","o_attn"                             # some Qwen variants
]

def find_lora_targets(model):
    counts = {}
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = n.split(".")[-1]
            for pat in PREFERRED_PATTERNS:
                if last.endswith(pat) or pat in last:
                    names.add(last)
                    counts[last] = counts.get(last,0)+1
    if names:
        return sorted(names)
    # fallback: ALL linear layer names except lm_head/out proj heads & embeddings
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = n.split(".")[-1]
            if "lm_head" in n or "embed" in n: 
                continue
            names.add(last)
    return sorted(names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data",  default="data/train_blip_aug.jsonl")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16
    )
    model.config.use_cache = False

    target_modules = find_lora_targets(model)
    print("[LoRA] target_modules =", target_modules)

    lora = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # must show >0 trainable params

    def collate(batch):
        p_list = [b["prompt_only"] for b in batch]
        f_list = [b["text_full"]    for b in batch]
        tp = tok(p_list, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        tf = tok(f_list,   return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        input_ids = tf["input_ids"]; attn_mask = tf["attention_mask"]
        labels = input_ids.clone()
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        for i in range(len(batch)):
            prompt_len = (tp["input_ids"][i] != pad_id).sum().item()
            labels[i,:prompt_len] = -100
        return {"input_ids":input_ids, "attention_mask":attn_mask, "labels":labels}

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=25, save_steps=500, save_total_limit=2,
        fp16=True, bf16=False, gradient_checkpointing=True,
        dataloader_num_workers=3, report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print("Saved LoRA to", args.out)

if __name__ == "__main__":
    main()
