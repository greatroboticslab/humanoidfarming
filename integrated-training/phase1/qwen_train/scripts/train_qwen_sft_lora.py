import os
import json
import inspect
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME  = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DATA_PATH   = os.environ.get("DATA_PATH", "qwen_train/data/train.jsonl")
OUT_DIR     = os.environ.get("OUT_DIR", "qwen_train/ckpts/qwen2p5-7b-lora")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))

def format_example(ex):
    prompt = (ex.get("prompt") or "").strip()
    response = (ex.get("response") or "").strip()
    text = (
        "### Instruction\n"
        f"{prompt}\n\n"
        "### Response\n"
        f"{response}\n"
    )
    return {"text": text}

def main():
    # --- dataset ---
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(format_example, remove_columns=ds.column_names)

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # --- quant + model ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
    )

    # --- LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # --- training args ---
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=1.0,
    )

    # --- formatting for TRL ---
    def formatting_func(example):
        return example["text"]

    # --- build kwargs that match YOUR installed TRL SFTTrainer ---
    sig = inspect.signature(SFTTrainer.__init__)
    supported = set(sig.parameters.keys())

    candidate_kwargs = {
        # tokenizer handling (different TRL versions use different names)
        "processing_class": tokenizer,
        "tokenizer": tokenizer,

        # dataset text handling (some versions use this, some don't)
        "dataset_text_field": "text",
        "formatting_func": formatting_func,

        # length handling (different TRL versions use different names)
        "max_seq_length": MAX_SEQ_LEN,
        "max_length": MAX_SEQ_LEN,

        # packing flag (name is usually the same)
        "packing": False,
    }

    trainer_kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    print("[INFO] SFTTrainer supports:", sorted(list(supported)))
    print("[INFO] Passing kwargs:", json.dumps(sorted(list(trainer_kwargs.keys())), indent=2))

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora_config,
        **trainer_kwargs
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] Saved LoRA adapter to: {OUT_DIR}")

if __name__ == "__main__":
    main()
