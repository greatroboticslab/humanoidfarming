#!/usr/bin/env python3

import torch

# ------------------------------------------------------------------
# Workaround for older torch + newer torchvision mismatch:
# Some torchvision versions expect torch.library.register_fake.
# If it's missing, define a harmless no-op so imports don't crash.
# ------------------------------------------------------------------
if not hasattr(torch, "library") or not hasattr(torch.library, "register_fake"):
    import types

    if not hasattr(torch, "library"):
        torch.library = types.SimpleNamespace()

    def _dummy_register_fake(*args, **kwargs):
        def inner(fn):
            return fn
        return inner

    torch.library.register_fake = _dummy_register_fake
# ------------------------------------------------------------------

import os
import re
import json
from pathlib import Path
from typing import List

from PIL import Image
import imagehash
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ============================================================
# CONFIG
# ============================================================

# Directory with extracted frames (each subdir = one video_index)
FRAME_ROOT = Path(os.environ.get("FRAME_ROOT", "results/frame_extractions"))

# Where to write one JSON per video with tasks + subtasks + frames + captions
JSON_OUT_DIR = Path(os.environ.get("JSON_OUT_DIR", "results/unified_pipeline/frame_captions"))

# Where the tasks-with-timestamps JSONs live
TASK_JSON_DIR = Path(os.environ.get("TASK_JSON_DIR", "results/unified_pipeline/pipeline2_guided_tasks"))

# InstructBLIP model to use for captioning
INSTRUCTBLIP_MODEL = "Salesforce/instructblip-flan-t5-xl"

# Caption behavior
CAPTION_PROMPT = "Describe this video frame in one short, clear sentence."
MAX_NEW_TOKENS = 32   # smaller to reduce memory

# If True, duplicate images are deleted from disk; if False, they’re just skipped
DELETE_DUPLICATES = False

JSON_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# InstructBLIP: load model & processor
# ============================================================

def load_instructblip():
    print(f"[INFO] Loading InstructBLIP model: {INSTRUCTBLIP_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InstructBlipForConditionalGeneration.from_pretrained(
        INSTRUCTBLIP_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    processor = InstructBlipProcessor.from_pretrained(INSTRUCTBLIP_MODEL)

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


def caption_image(processor, model, img_path: Path) -> str:
    """
    Use InstructBLIP to generate a short caption for a frame.
    Includes a safety catch for CUDA OOM.
    """
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {img_path} for captioning: {e}")
        return ""

    try:
        device = model.device
        inputs = processor(
            images=image,
            text=CAPTION_PROMPT,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

        # InstructBLIPProcessor has a tokenizer attached
        raw_caption = processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        caption = raw_caption.strip()
    except torch.cuda.OutOfMemoryError:
        print(f"[ERROR] CUDA OOM while captioning {img_path}. Skipping this frame.")
        caption = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[ERROR] Unexpected error while captioning {img_path}: {e}")
        caption = ""

    # Clean up any residual GPU cache between frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return caption


def attach_frame_to_tasks(
    tasks: List[dict],
    task_idx: int,
    sub_idx: int,
    frame_entry: dict,
):
    """
    Attach a frame_entry dict to tasks[task_idx]['subtasks'][sub_idx]['frames'].
    If indices are out of range or malformed, the frame is skipped.
    """
    if task_idx is None or sub_idx is None:
        return

    if task_idx < 0 or sub_idx < 0:
        return

    if task_idx >= len(tasks):
        print(f"[WARN] task_idx {task_idx} out of range for tasks (len={len(tasks)}). Skipping frame.")
        return

    task = tasks[task_idx]
    subtasks = task.get("subtasks", [])
    if sub_idx >= len(subtasks):
        print(f"[WARN] subtask_index {sub_idx} out of range for task {task_idx} (len={len(subtasks)}). Skipping frame.")
        return

    subtask = subtasks[sub_idx]
    # ensure "frames" list exists
    if "frames" not in subtask or subtask["frames"] is None:
        subtask["frames"] = []

    subtask["frames"].append(frame_entry)


def process_video_frames(video_dir: Path, processor, model):
    """
    For a given video directory:
    - Load its tasks_with_timestamps JSON
    - De-duplicate frames
    - Caption each unique frame
    - Attach frames under the right subtask (by task/sub indices from filename)
    - Save one JSON with structure: tasks -> subtasks -> frames
    """
    video_index = video_dir.name

    # Load the tasks-with-timestamps JSON
    task_json_path = TASK_JSON_DIR / f"{video_index}.json"
    if not task_json_path.exists():
        print(f"[WARN] No task JSON found for {video_index} at {task_json_path}, skipping.")
        return

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_info = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to read task JSON {task_json_path}: {e}")
        return

    # Extract tasks and other metadata
    tasks = task_info.get("tasks", [])
    if not tasks:
        print(f"[WARN] No tasks found in {task_json_path}, skipping.")
        return

    # Make sure each subtask has a frames list
    for t in tasks:
        for s in t.get("subtasks", []) or []:
            if "frames" not in s or s["frames"] is None:
                s["frames"] = []

    unique_frames = dedupe_frames_in_video_dir(video_dir)

    if not unique_frames:
        print(f"[WARN] No usable frames in {video_index}, skipping JSON.")
        return

    for img_path in unique_frames:
        task_idx, sub_idx, frame_idx = parse_frame_name(img_path)
        if task_idx is None:
            print(f"[WARN] Could not parse task/sub/frame from {img_path.name}, skipping.")
            continue

        print(f"[CAPTION] {video_index} -> {img_path.name}")
        caption = caption_image(processor, model, img_path)

        frame_entry = {
            "image_file": img_path.name,
            "relative_path": str(img_path.relative_to(FRAME_ROOT)),
            "caption": caption,
            "frame_index": frame_idx,
        }

        attach_frame_to_tasks(tasks, task_idx, sub_idx, frame_entry)

    # Build final payload: preserve Pipeline 2 mission/sub-mission metadata
    # and keep the updated task/subtask/frame structure.
    out_payload = {
        "index": task_info.get("index") or video_index,
        "video_index": video_index,
        "title": task_info.get("title"),
        "url": task_info.get("url"),
        "category": task_info.get("category"),
        "relevant": task_info.get("relevant"),
        "reason": task_info.get("reason"),
        "integration_note": task_info.get("integration_note"),
        "mission": task_info.get("mission"),
        "tasks": tasks,
    }

    out_path = JSON_OUT_DIR / f"{video_index}.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote nested JSON -> {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}")


# ============================================================
# Main
# ============================================================

def main():
    processor, model = load_instructblip()

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
