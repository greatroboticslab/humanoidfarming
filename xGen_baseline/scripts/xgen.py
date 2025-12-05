#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModelForVision2Seq,
)

MODEL_DIR = "/VideoProcessing/XGEN_baseline/results/blip3_full_finetune_clean"
DATA_JSONL = "/VideoProcessing/XGEN_baseline/results/training_data/blip3_frames.jsonl"
OUTPUT_DIR = "VideoProcessing/XGEN_baseline/results/xgen"

device = "cuda" if torch.cuda.is_available() else "cpu"

# limit number of videos for testing; set to None for all
MAX_VIDEOS = 1   # change to None when you want all videos


def load_everything():
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        use_fast=True,
        legacy=False,
    )

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )

    tokenizer = model.update_special_tokens(tokenizer)
    tokenizer.padding_side = "left"

    if hasattr(model, "vlm") and hasattr(model.vlm, "vision_tokenizer"):
        vt = model.vlm.vision_tokenizer
        original_forward = vt.forward

        def patched_forward(self, x, vision_attn_masks=None):
            if vision_attn_masks is None:
                b, T, F, v = x.shape[:4]
                mask = torch.ones((b, v), dtype=torch.bool, device=x.device)
            else:
                mask = vision_attn_masks
            return original_forward(x, mask)

        vt.forward = patched_forward.__get__(vt, vt.__class__)

    model.to(device)
    model.eval()
    return tokenizer, image_processor, model


def build_vision_x(pixel_values: torch.Tensor) -> torch.Tensor:
    if pixel_values.ndim == 6:
        return pixel_values
    elif pixel_values.ndim == 4:
        return pixel_values.unsqueeze(1).unsqueeze(2)
    elif pixel_values.ndim == 5:
        return pixel_values.unsqueeze(2)
    elif pixel_values.ndim == 3:
        return pixel_values.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected pixel_values.ndim={pixel_values.ndim}, shape={pixel_values.shape}")


def greedy_generate(tokenizer, image_processor, model, image_path, input_text, max_new_tokens=96):
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {image_path}: {e}")
        raw_image = Image.new("RGB", (224, 224), color=(0, 0, 0))

    image_inputs = image_processor([raw_image], return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)
    vision_x = build_vision_x(pixel_values)

    prompt = (
        "You are an assistant that explains farming videos.\n\n"
        + input_text
        + "\n\nAnswer: "
    )

    text_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    generated = input_ids

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)
    try:
        end_tok = tokenizer.convert_tokens_to_ids("<|end|>")
        if end_tok is not None and end_tok not in eos_ids:
            eos_ids.add(end_tok)
    except Exception:
        pass

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model.vlm(
                vision_x=vision_x,
                lang_x=generated,
                attention_mask=attention_mask,
            )
            next_token = outputs.logits[:, -1].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.ones_like(generated, device=device)

            if eos_ids and next_token.item() in eos_ids:
                break

    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        clean = decoded.split("Answer:", 1)[-1].strip()
    else:
        clean = decoded.strip()

    clean = clean.replace("\n", " ").strip()
    return clean


def pretty_frame_block(record):
    block = []
    block.append("------------------------------------------------------------")
    block.append(
        f"Task {record['task_index']} | Subtask {record['subtask_index']} | "
        f"Frame: {record['frame_name']}"
    )
    block.append("")
    block.append("Task / Question:")
    block.append(record["clean_question"])
    block.append("")
    block.append("Ground Truth:")
    block.append(record["answer_text"])
    block.append("")
    block.append("Model Prediction:")
    block.append(record["predicted_text"])
    block.append("------------------------------------------------------------")
    return "\n".join(block)


def sanitize_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))


def main():
    tokenizer, image_processor, model = load_everything()

    ds = load_dataset("json", data_files={"train": DATA_JSONL})["train"]
    print(f"[INFO] Loaded {len(ds)} total frames")

    videos = {}
    for ex in ds:
        vid = ex.get("video_index", "unknown")
        videos.setdefault(vid, []).append(ex)

    all_video_ids = sorted(videos.keys())
    if MAX_VIDEOS is not None:
        all_video_ids = all_video_ids[:MAX_VIDEOS]
        print(f"[INFO] Limiting to first {MAX_VIDEOS} videos")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    for vi, vid in enumerate(all_video_ids, start=1):
        frames = videos[vid]
        frames.sort(
            key=lambda ex: (
                ex.get("task_index", 0),
                ex.get("subtask_index", 0),
                ex.get("frame_index", 0),
            )
        )

        safe_vid = sanitize_filename(str(vid))
        out_path = out_root / f"{safe_vid}.txt"

        print(
            f"[INFO] [{vi}/{len(all_video_ids)}] Processing video {vid} "
            f"with {len(frames)} frames -> {out_path}"
        )

        with out_path.open("w") as f_out:
            f_out.write(f"==================== VIDEO {vid} ====================\n\n")
            for i, ex in enumerate(frames):
                input_text = ex["input_text"]

                task_text = (
                    ex.get("task_text")
                    or ex.get("task")
                    or ex.get("task_description")
                )

                if task_text:
                    clean_question = task_text.strip()
                else:
                    clean_question = input_text.split("Question:", 1)[-1].strip()

                frame_name = Path(ex["image_path"]).name

                try:
                    pred = greedy_generate(
                        tokenizer,
                        image_processor,
                        model,
                        ex["image_path"],
                        input_text,
                    )
                except Exception as e:
                    print(
                        f"[WARN]   failed on frame idx={i}, "
                        f"image={ex['image_path']}: {e}"
                    )
                    pred = "<error>"

                rec = {
                    "task_index": ex.get("task_index"),
                    "subtask_index": ex.get("subtask_index"),
                    "frame_name": frame_name,
                    "clean_question": clean_question,
                    "answer_text": ex["answer_text"],
                    "predicted_text": pred,
                }

                f_out.write(pretty_frame_block(rec) + "\n\n")

        print(f"[INFO]   Saved video report to {out_path}")

    print(f"[OK] Done. Per-video reports in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
