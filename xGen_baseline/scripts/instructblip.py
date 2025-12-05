#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
)

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_NAME = "Salesforce/instructblip-flan-t5-xl"

DATA_JSONL = (
    "/VideoProcessing/XGEN_baseline/results/training_data/blip3_frames.jsonl"
)

OUTPUT_DIR = (
    "/VideoProcessing/XGEN_baseline/results/instructblip"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# limit number of videos for testing; set to None for all
MAX_VIDEOS = 1


# ---------------------------------------------------------
# Load model + processor
# ---------------------------------------------------------

def load_everything():
    print(f"[INFO] Loading model {MODEL_NAME} on {device}...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    print("[INFO] Model loaded.")
    return processor, model


# ---------------------------------------------------------
# Detailed generation for ONE frame
# ---------------------------------------------------------

def generate_detailed_caption(
    processor,
    model,
    image_path: str,
    question_text: str,
    max_new_tokens: int = 128,
) -> str:
    """
    Generate 2–4 detailed, grounded sentences for a single frame.
    """
    image = Image.open(image_path).convert("RGB")

    # Rich, explicit prompt
    prompt = (
        "You are an expert agricultural visual analyst. "
        "You are looking at a single frame from a farming-related promotional video.\n\n"
        "Describe this frame in 2–4 detailed sentences. "
        "Always mention:\n"
        "1) What the main subjects (people, plants, machines, or screens) are doing.\n"
        "2) The setting and important visual details in the background.\n"
        "3) Any visible equipment, tools, graphs, or on-screen text if you can see it.\n\n"
        "Be specific and precise based purely on the image. "
        "Do NOT hallucinate numbers, company names, or technology if they are not clearly visible.\n\n"
        f"Frame request: {question_text}\n\n"
        "Your detailed answer:"
    )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,     # greedy for stability
            repetition_penalty=1.05,
        )

    decoded = processor.tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    # Try to cut off the prompt echo if present
    if "Your detailed answer:" in decoded:
        decoded = decoded.split("Your detailed answer:", 1)[-1].strip()
    elif "Answer:" in decoded:
        decoded = decoded.split("Answer:", 1)[-1].strip()

    # Clean up newlines
    decoded = " ".join(decoded.split())
    return decoded


# ---------------------------------------------------------
# Pretty formatting helper
# ---------------------------------------------------------

def pretty_frame_block(record):
    block = []
    block.append("------------------------------------------------------------")
    block.append(
        f"Task {record['task_index']} | Subtask {record['subtask_index']} | "
        f"Frame: {record['frame_name']}"
    )
    block.append("")
    block.append("Input Question:")
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


# ---------------------------------------------------------
# Main: group by video and write one file per video
# ---------------------------------------------------------

def main():
    processor, model = load_everything()

    ds = load_dataset("json", data_files={"train": DATA_JSONL})["train"]
    print(f"[INFO] Loaded {len(ds)} total frames")

    # group by video_index
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
                # Just the question part for display
                question_line = input_text.split("Question:", 1)[-1].strip()
                frame_name = Path(ex["image_path"]).name

                try:
                    pred = generate_detailed_caption(
                        processor,
                        model,
                        ex["image_path"],
                        question_line,
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
                    "clean_question": question_line,
                    "answer_text": ex["answer_text"],
                    "predicted_text": pred,
                }

                f_out.write(pretty_frame_block(rec) + "\n\n")

        print(f"[INFO]   Saved video report to {out_path}")

    print(f"[OK] Done. Per-video detailed reports in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
