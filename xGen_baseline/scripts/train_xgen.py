#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_NAME = "Salesforce/xgen-mm-phi3-mini-instruct-singleimg-r-v1.5"

DATA_JSONL = "VideoProcessing/XGEN_baseline/results/training_data/blip3_frames.jsonl"

OUTPUT_DIR = "VideoProcessing/XGEN_baseline/results/blip3_full_finetune_clean"

MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 1
NUM_EPOCHS = 1
LR = 2e-5

# use more later; for now keep small to test
MAX_SAMPLES: Optional[int] = 20


# ---------------------------------------------------------
# Model + tokenizer + image processor (with patch)
# ---------------------------------------------------------

def load_model_and_processors():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
        legacy=False,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    tokenizer = model.update_special_tokens(tokenizer)
    tokenizer.padding_side = "left"

    # patch vision_tokenizer to make vision_attn_masks optional
    if hasattr(model, "vlm") and hasattr(model.vlm, "vision_tokenizer"):
        vt = model.vlm.vision_tokenizer
        orig_forward = vt.forward

        def patched_forward(self, x, vision_attn_masks=None):
            if vision_attn_masks is None:
                b, T, F, v = x.shape[:4]
                vision_attn_masks_local = torch.ones(
                    (b, v), dtype=torch.bool, device=x.device
                )
            else:
                vision_attn_masks_local = vision_attn_masks
            return orig_forward(x, vision_attn_masks_local)

        vt.forward = patched_forward.__get__(vt, vt.__class__)

    model.to(device)
    return tokenizer, image_processor, model


# ---------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------

def load_jsonl_dataset(path: str, max_samples: Optional[int] = None):
    ds = load_dataset("json", data_files={"train": path})["train"]

    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
        print(f"[INFO] Using only {max_samples} samples")

    return ds


# ---------------------------------------------------------
# Example preparation (image + text, labels only on answer)
# ---------------------------------------------------------

def make_example(tokenizer, image_processor, example):
    img_path = example["image_path"]
    input_text = example["input_text"]
    answer_text = example["answer_text"]

    # Load image
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {img_path}: {e}")
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))

    # Prompt: keep text context but avoid over-long boilerplate
    # Still uses both image + text
    prompt = (
        "You are an assistant that explains farming videos.\n\n"
        + input_text
        + "\n\nAnswer: "
    )

    full_text = prompt + answer_text

    # Encode image
    image_inputs = image_processor(
        [image],
        return_tensors="pt",
    )

    # ---- TEXT ENCODING WITH PROMPT MASKING ----
    # 1) tokenize full prompt + answer (for inputs)
    text_inputs = tokenizer(
        [full_text],
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )

    # 2) tokenize ONLY the prompt (no answer) to know where to mask
    prompt_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Build labels same as input_ids, but ignore prompt tokens
    labels = text_inputs["input_ids"].clone()
    # mask out the prompt part
    labels[:, :prompt_len] = -100  # -100 is ignored by CrossEntropyLoss in HF

    # Merge dicts
    model_inputs = {**image_inputs, **text_inputs}
    model_inputs["labels"] = labels

    # Remove batch dim
    model_inputs = {k: v[0] for k, v in model_inputs.items()}
    return model_inputs


class VideoQADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_processor):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        example = self.ds[idx]
        return make_example(self.tokenizer, self.image_processor, example)


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def build_vision_x(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Ensure vision_x has shape (B, T_img, F, C, H, W).
    In your case, processor already returns 6D [B, 1, 1, C, H, W].
    """
    if pixel_values.ndim == 6:
        return pixel_values
    elif pixel_values.ndim == 4:
        return pixel_values.unsqueeze(1).unsqueeze(2)
    elif pixel_values.ndim == 5:
        return pixel_values.unsqueeze(1)
    elif pixel_values.ndim == 3:
        return pixel_values.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected pixel_values.ndim={pixel_values.ndim}, shape={pixel_values.shape}")


# ---------------------------------------------------------
# Training loop (image + text, labels only on answer)
# ---------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, image_processor, model = load_model_and_processors()
    model.train()

    train_ds_raw = load_jsonl_dataset(DATA_JSONL, max_samples=MAX_SAMPLES)
    print(f"[INFO] Raw HF dataset size: {len(train_ds_raw)}")

    train_ds = VideoQADataset(train_ds_raw, tokenizer, image_processor)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        optimizer.zero_grad()

        for step, batch in enumerate(train_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            vision_x = build_vision_x(pixel_values)

            if global_step == 0 and step == 0:
                print(f"[DEBUG] pixel_values.shape = {pixel_values.shape}")
                print(f"[DEBUG] vision_x.shape = {vision_x.shape}")

            outputs = model.vlm(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            print(f"Step {global_step}: loss = {loss.item():.4f}")

        print(f"End of epoch {epoch+1}: last loss = {loss.item():.4f}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    image_processor.save_pretrained(OUTPUT_DIR)
    print(f"[OK] Finished training. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
