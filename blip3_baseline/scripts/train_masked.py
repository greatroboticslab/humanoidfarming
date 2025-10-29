import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

def split_conv(tokenizer, example):
    system = example.get("system","")
    msgs = example.get("messages",[])
    conv=[]
    if system:
        conv.append({"role":"system","content":system})
    for m in msgs:
        parts=[]
        for blk in m.get("content",[]) or []:
            if isinstance(blk, dict) and blk.get("type")=="text":
                parts.append(blk.get("text",""))
            elif isinstance(blk,str):
                parts.append(blk)
        conv.append({"role":m.get("role","user"),"content":"\n".join(parts)})
    # prompt (up to last assistant) and full convo
    # if last role isn't assistant, fallback to all-1
    if conv and conv[-1]["role"]=="assistant":
        prompt_msgs = conv[:-1]
    else:
        prompt_msgs = conv
    try:
        prompt_only = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_only = "".join([f"{m['role'].title()}: {m['content']}\n" for m in prompt_msgs])
    try:
        full_text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    except Exception:
        full_text = "".join([f"{m['role'].title()}: {m['content']}\n" for m in conv])
    return {"prompt_only":prompt_only, "text_full":full_text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data",  default="data/train_blip_filled.jsonl")
    ap.add_argument("--out",   default="outputs/lora-qwen3b-blipchat")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr",    type=float, default=5e-5)
    ap.add_argument("--epochs",type=int, default=2)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--quant", choices=["4bit","8bit","fp16"], default="4bit")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda ex: split_conv(tok, ex), remove_columns=ds.column_names)

    # load base model
    if args.quant in ("4bit","8bit"):
        load_kwargs=dict(device_map="auto", trust_remote_code=True)
        if args.quant=="4bit":
            load_kwargs["load_in_4bit"]=True
        else:
            load_kwargs["load_in_8bit"]=True
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora)

    def collate(batch):
        p_list = [b["prompt_only"] for b in batch]
        f_list = [b["text_full"]    for b in batch]
        toks_p = tok(p_list, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        toks_f = tok(f_list, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)

        input_ids = toks_f["input_ids"]
        attn_mask = toks_f["attention_mask"]
        labels = input_ids.clone()

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        # compute prompt lengths per item
        for i in range(len(batch)):
            p = toks_p["input_ids"][i]
            prompt_len = (p != pad_id).sum().item()
            labels[i,:prompt_len] = -100  # mask non-assistant
        return {"input_ids":input_ids, "attention_mask":attn_mask, "labels":labels}

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
        fp16=True, bf16=False,
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
