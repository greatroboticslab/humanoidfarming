#!/usr/bin/env python3
import os
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Set

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ----------------------------
# Dataset: JSONL with "messages"
# ----------------------------

class ChatJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                msgs = obj.get("messages")
                if isinstance(msgs, list) and msgs:
                    self.samples.append(msgs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        msgs = self.samples[idx]

        prompt_msgs = []
        assistant_text = None
        for m in msgs:
            if m.get("role") == "assistant":
                assistant_text = (m.get("content") or "")
                break
            prompt_msgs.append(m)

        if assistant_text is None:
            assistant_text = ""

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = self.tokenizer.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": assistant_text}],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = prompt_enc["input_ids"].shape[1]
        labels[:prompt_len] = -100  # only train on assistant tokens

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ----------------------------
# Collator (pad to max in batch)
# ----------------------------

@dataclass
class SimpleCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab
        batch["labels"] = padded_labels
        return batch


# ----------------------------
# LoRA helpers
# ----------------------------

def find_linear_module_names(model) -> Set[str]:
    """
    Return the last component names of torch.nn.Linear modules.
    Example: 'q_proj', 'k_proj', ...
    """
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.add(name.split(".")[-1])
    return names


def print_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    print(f"[PARAMS] trainable={trainable:,} / total={total:,} ({pct:.4f}%)")
    return trainable


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: CUDA not available inside job.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force single GPU load (avoid “multiple devices” surprises)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=None,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda")

    model.config.use_cache = False
    model.train()

    # Discover whether the expected Qwen projection names exist
    linear_names = find_linear_module_names(model)
    preferred = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    target_modules = [n for n in preferred if n in linear_names]

    if not target_modules:
        # Fallback: train ALL Linear modules (heavier, but ensures trainable params exist)
        # In practice, Qwen should have the preferred names; this is just a safety net.
        print("[WARN] None of the preferred target_modules found. Falling back to all Linear module suffixes.")
        target_modules = sorted(list(linear_names))

    print(f"[INFO] Using LoRA target_modules: {target_modules}")

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)

    # Enable checkpointing properly for PEFT
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainable = print_trainable_params(model)
    if trainable == 0:
        raise SystemExit("FATAL: No trainable parameters after applying LoRA. target_modules likely mismatched.")

    train_ds = ChatJsonlDataset(args.train_jsonl, tokenizer, args.max_seq_len)
    collator = SimpleCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        report_to=[],
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"[OK] saved LoRA adapter to {args.out_dir}")


if __name__ == "__main__":
    main()
