#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from typing import List

from PIL import Image
import imagehash
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# ============================================================
# CONFIG
# ============================================================

# Directory with extracted frames (each subdir = one video_index)
FRAME_ROOT = Path(
    "/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/frame_extractions"
)

# Where to write one JSON per video with frame captions
JSON_OUT_DIR = Path(
    "/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/frame_captions"
)

# Qwen vision-language model
QWEN_VL_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

# Caption behavior
CAPTION_PROMPT = "Describe this video frame in one short, clear sentence."
MAX_NEW_TOKENS = 64

# If True, duplicate images are deleted from disk; if False, they’re just skipped
DELETE_DUPLICATES = False

JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Qwen-VL: load model & processor
# ============================================================

def load_qwen_vl():
    print(f"[INFO] Loading Qwen VL model: {QWEN_VL_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForVision2Seq.from_pretrained(
        QWEN_VL_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL, trust_remote_code=True)

    return processor, model


# ============================================================
# Helpers
# ============================================================

frame_name_re = re.compile(r"task(\d+)_sub(\d+)_f(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_frame_name(path: Path):
    """
    Parse task/subtask/frame index from a filename like:
    task00_sub01_f03.jpg -> (0, 1, 3)
    """
    m = frame_name_re.match(path.name)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def compute_hash(img_path: Path):
    """
    Compute a perceptual hash for an image (used for deduplication).
    """
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            return imagehash.phash(im)
    except Exception as e:
        print(f"[WARN] Failed to hash {img_path}: {e}")
        return None


def dedupe_frames_in_video_dir(video_dir: Path) -> List[Path]:
    """
    Return a list of unique frame paths in this video_dir.
    Optionally delete duplicate files on disk.
    """
    print(f"[INFO] De-duplicating frames in {video_dir.name}")
    unique_paths: List[Path] = []
    seen_hashes = {}

    for img_path in sorted(video_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        h = compute_hash(img_path)
        if h is None:
            continue

        if h in seen_hashes:
            print(f"[DUP] {img_path} is duplicate of {seen_hashes[h]}")
            if DELETE_DUPLICATES:
                try:
                    img_path.unlink()
                    print(f"     Deleted {img_path}")
                except Exception as e:
                    print(f"[WARN] Could not delete {img_path}: {e}")
            # skip adding this one
        else:
            seen_hashes[h] = img_path
            unique_paths.append(img_path)

    print(f"[INFO] {len(unique_paths)} unique frames kept in {video_dir.name}")
    return unique_paths


def clean_caption_text(raw: str) -> str:
    """
    Remove the system/user/assistant chat boilerplate from Qwen's output.

    Example input:
    system
    You are a helpful assistant.
    user
    Describe this video frame...
    assistant
    A person is...

    We only want: "A person is..."
    """
    text = raw.strip()

    marker = "\nassistant\n"
    if marker in text:
        text = text.split(marker, 1)[1].strip()

    # Extra cleanup: drop lines that are exactly these boilerplate bits
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in {
            "system",
            "user",
            "assistant",
            "You are a helpful assistant.",
            CAPTION_PROMPT,
        }:
            continue
        lines.append(stripped)

    cleaned = " ".join(lines).strip()

    return cleaned if cleaned else raw.strip()


def caption_image(processor, model, img_path: Path) -> str:
    """
    Use Qwen2-VL with the proper chat template so image tokens match image features,
    then clean the caption to remove system/user/assistant boilerplate.
    """
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {img_path} for captioning: {e}")
        return ""

    # Qwen2-VL expects a chat-style message with an image object in the content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]

    # Build text with image placeholder tokens
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    raw_caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    caption = clean_caption_text(raw_caption)
    return caption


def process_video_frames(video_dir: Path, processor, model):
    """
    - De-duplicate frames in this <video_index> directory
    - Caption each unique frame
    - Save one JSON with all frame metadata + captions
    """
    video_index = video_dir.name
    unique_frames = dedupe_frames_in_video_dir(video_dir)

    if not unique_frames:
        print(f"[WARN] No usable frames in {video_index}, skipping JSON.")
        return

    frames_data = []

    for img_path in unique_frames:
        task_idx, sub_idx, frame_idx = parse_frame_name(img_path)
        if task_idx is None:
            print(f"[WARN] Could not parse task/sub/frame from {img_path.name}, setting indexes to -1.")
            task_idx = sub_idx = frame_idx = -1

        print(f"[CAPTION] {video_index} -> {img_path.name}")
        caption = caption_image(processor, model, img_path)

        frame_entry = {
            "video_index": video_index,
            "image_file": img_path.name,
            "relative_path": str(img_path.relative_to(FRAME_ROOT)),
            "task_index": task_idx,
            "subtask_index": sub_idx,
            "frame_index": frame_idx,
            "caption": caption,
        }
        frames_data.append(frame_entry)

    out_path = JSON_OUT_DIR / f"{video_index}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_index": video_index,
                    "frames": frames_data,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[OK] Wrote JSON -> {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}")


# ============================================================
# Main
# ============================================================

def main():
    processor, model = load_qwen_vl()

    if not FRAME_ROOT.exists():
        print(f"[ERROR] FRAME_ROOT does not exist: {FRAME_ROOT}")
        return

    video_dirs = [p for p in FRAME_ROOT.iterdir() if p.is_dir()]
    if not video_dirs:
        print(f"[WARN] No video directories found under {FRAME_ROOT}")
        return

    print(f"[INFO] Found {len(video_dirs)} video directories.")

    for vdir in sorted(video_dirs):
        process_video_frames(vdir, processor, model)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
