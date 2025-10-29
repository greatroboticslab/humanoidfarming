#!/bin/bash -l
#SBATCH -J blip_fix_train_eval
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/blip_fix_train_eval_%j.out
#SBATCH -e logs/blip_fix_train_eval_%j.err
set -eo pipefail

# ---- Paths & model ----
PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
HF_CACHE=$PROJECT/hf_cache
MODEL_ID=Qwen/Qwen2.5-3B-Instruct

# ---- Env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o
export HF_HOME=$HF_CACHE HUGGINGFACE_HUB_CACHE=$HF_CACHE XDG_CACHE_HOME=$HF_CACHE
unset TRANSFORMERS_CACHE || true
mkdir -p "$WORKDIR"/{logs,results,outputs,scripts}
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
cd "$WORKDIR"

echo "=== RUNTIME ==="; date; hostname; nvidia-smi || true
python - <<'PY'
import torch; print("CUDA avail:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
echo "==============="

# =========================
# A) BASE batch inference
# =========================
# Produces strong step lists for each sample.
echo "[A] Base inference -> results/base_for_fill"
python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --out_dir results/base_for_fill \
  --max_new_tokens 400 \
  --temperature 0.3

# ===========================================
# B) Build FILLED dataset from base outputs
# ===========================================
# Creates data/train_blip_filled.jsonl with real assistant steps.

cat > scripts/build_filled_dataset.py <<'PY'
import json, os, re
from pathlib import Path

root = Path(".")
src_jsonl = root/"data/train_blip.jsonl"
res_dir   = root/"results/base_for_fill"
dst_jsonl = root/"data/train_blip_filled.jsonl"

def extract_steps_from_txt(txt: str) -> str:
    # remove header lines up to dashed separator
    txt = txt.replace("\r", "")
    parts = txt.split("\n")
    try:
        sep_idx = next(i for i,l in enumerate(parts) if re.match(r"^-{5,}$", l.strip()))
        body = "\n".join(parts[sep_idx+1:]).strip()
    except StopIteration:
        # fallback: keep everything
        body = "\n".join(parts).strip()
    return body

def main():
    n_in, n_ok, n_miss = 0,0,0
    with open(src_jsonl, "r", encoding="utf-8", errors="ignore") as fi, \
         open(dst_jsonl, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): continue
            rec = json.loads(line)
            sid = rec.get("id") or ""
            step_file = res_dir / f"{sid}.txt"
            if not step_file.exists():
                # also support sample_000123 style names
                step_file = res_dir / f"{sid if sid.endswith('.txt') else sid}.txt"
                if not step_file.exists():
                    # last try: match the numeric suffix
                    m = re.search(r"(\d{6})", sid)
                    if m:
                        step_file = res_dir / f"sample_{m.group(1)}.txt"
            n_in += 1
            if step_file.exists():
                body = extract_steps_from_txt(step_file.read_text(encoding="utf-8", errors="ignore"))
                # Replace assistant content with real steps
                msgs = rec.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        c = m.get("content", [])
                        if isinstance(c, list) and c:
                            # overwrite first text block or append
                            found = False
                            for blk in c:
                                if isinstance(blk, dict) and blk.get("type")=="text":
                                    blk["text"] = body
                                    found = True
                                    break
                            if not found:
                                c.append({"type":"text","text":body})
                        elif isinstance(c, str):
                            m["content"] = body
                        else:
                            m["content"] = [{"type":"text","text":body}]
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
                n_ok += 1
            else:
                # keep original if missing (still useful)
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
                n_miss += 1
    print(f"Filled dataset saved to: {dst_jsonl}")
    print(f"Total: {n_in}, replaced: {n_ok}, missing: {n_miss}")

if __name__ == "__main__":
    main()
PY

echo "[B] Build filled dataset"
python -u scripts/build_filled_dataset.py

# =========================================================
# C) Train with masked loss (only assistant tokens learn)
# =========================================================

cat > scripts/train_masked.py <<'PY'
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
PY

echo "[C] Train masked LoRA -> outputs/lora-qwen3b-blipchat"
python -u scripts/train_masked.py \
  --model "$MODEL_ID" \
  --data  data/train_blip_filled.jsonl \
  --out   outputs/lora-qwen3b-blipchat \
  --batch 2 --grad_accum 8 --lr 5e-5 --epochs 2 --max_len 2048 --quant 4bit

# Sanity: show adapter_config base model (optional)
python - <<'PY'
import json, os
p="outputs/lora-qwen3b-blipchat/adapter_config.json"
print("[adapter_config exists?]", os.path.exists(p))
if os.path.exists(p):
    print(json.load(open(p)) .get("base_model_name_or_path"))
PY

# ===============================================
# D) Batch inference: base vs trained (separate)
# ===============================================
echo "[D1] Base batch -> results/base_run_new"
python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --out_dir results/base_run_new \
  --max_new_tokens 400 \
  --temperature 0.3

echo "[D2] Trained (LoRA) batch -> results/trained_run_new"
python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --peft_path outputs/lora-qwen3b-blipchat \
  --out_dir results/trained_run_new \
  --max_new_tokens 400 \
  --temperature 0.3

echo "✅ ALL DONE"; date

