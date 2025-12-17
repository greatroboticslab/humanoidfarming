#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

FRAME_CAPTION_DIR = Path("results/frame_captions")
OUTPUT_DIR = Path("results/subtask_guidance")

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Max frames per subtask to include in prompt (to keep context manageable)
MAX_FRAMES_PER_SUBTASK = 8

# Max new tokens from LLM
MAX_NEW_TOKENS = 384

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------

def load_model_and_tokenizer():
    print(f"[INFO] Loading model {MODEL_NAME} on cuda..." if torch.cuda.is_available() else f"[INFO] Loading model {MODEL_NAME} on cpu...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    print("[INFO] Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------
# PROMPT BUILDING
# ---------------------------------------------------------------------

def summarize_frames_for_prompt(frames: List[Dict[str, Any]]) -> str:
    """
    Build a compact text description of the frames for the prompt.
    Uses at most MAX_FRAMES_PER_SUBTASK frames to avoid over-long prompts.
    """
    if not frames:
        return "No frames available for this subtask."

    # Sort by frame_index if present
    frames_sorted = sorted(frames, key=lambda f: f.get("frame_index", 0))
    frames_trimmed = frames_sorted[:MAX_FRAMES_PER_SUBTASK]

    lines = []
    for f in frames_trimmed:
        idx = f.get("frame_index", 0)
        cap = f.get("caption", "").strip()
        rel = f.get("relative_path", "")
        lines.append(f"- frame_index={idx}, file={rel}: {cap}")
    return "\n".join(lines)


def build_subtask_prompt(
    video_index: str,
    title: str,
    task_obj: Dict[str, Any],
    subtask_obj: Dict[str, Any],
) -> str:
    """
    Build the natural-language prompt for one subtask.
    We explicitly ask for SECTIONED TEXT and explicitly forbid JSON / braces.
    """

    task_text = task_obj.get("task", "").strip()
    sub_text = subtask_obj.get("text", "").strip()
    start = subtask_obj.get("start", None)
    end = subtask_obj.get("end", None)
    frames = subtask_obj.get("frames") or []
    frame_summary = summarize_frames_for_prompt(frames)

    # Time range string
    if start is not None and end is not None:
        time_range = f"{start:.1f}s–{end:.1f}s"
    else:
        time_range = "unknown time range"

    prompt = f"""You are helping design a *high-level, human-readable* description of what a humanoid robot
should understand and do for a specific **subtask** in a tutorial video.

VIDEO METADATA
--------------
- video_index: {video_index}
- title: {title}
- high_level_task: {task_text}

SUBTASK INFO
------------
- subtask_text: {sub_text}
- subtask_time_range: {time_range}

FRAMES FOR THIS SUBTASK
-----------------------
Each frame has a frame_index and a caption. Use these as your only visual clues.
Do NOT invent completely unrelated objects or scenes; stay consistent with the captions.

{frame_summary}

YOUR JOB
--------
Based on the frames and subtask description, produce a **SECTIONED TEXT DESCRIPTION**
for this subtask that a humanoid robot could ultimately use as a guide.

IMPORTANT FORMAT RULES
----------------------
- DO NOT output JSON.
- DO NOT use curly braces `{{` or `}}`.
- DO NOT use quotation marks around section titles.
- Use exactly the following section headings in this order, each on its own line:
  1. GLOBAL_SUMMARY:
  2. FRAME_BASED_OBSERVATIONS:
  3. INTEGRATED_SCENE_UNDERSTANDING:
  4. PRECONDITIONS_FOR_ROBOT:
  5. SUCCESS_CRITERIA:
  6. ORDERED_ROBOT_ACTION_STEPS:
  7. SUBTASK_STORY:
- Put a blank line after each section title.
- For FRAME_BASED_OBSERVATIONS, use bullet points like:
  - [frames=[0,1]] description...
  - [frames=[2]] description...
- For ORDERED_ROBOT_ACTION_STEPS, use numbered steps like:
  1. [type=navigation, frames=[0,1]] ...
  2. [type=manipulation, frames=[2]] ...
- Only reference frame indices that appear in the frame list above.

CONTENT GUIDELINES
------------------
- GLOBAL_SUMMARY: 1–3 sentences describing the subtask at a high level.
- FRAME_BASED_OBSERVATIONS: 3–8 bullet points linking frames to what is happening.
- INTEGRATED_SCENE_UNDERSTANDING: 1–2 short paragraphs explaining the overall story of this subtask.
- PRECONDITIONS_FOR_ROBOT: 3–6 bullet points listing what must already be true for the robot to succeed.
- SUCCESS_CRITERIA: 3–6 bullet points describing when the subtask is considered successfully completed.
- ORDERED_ROBOT_ACTION_STEPS: 5–10 numbered steps; each step MUST have:
    [type=<navigation|manipulation|perception|communication>, frames=[...]] followed by a short, clear action description.
- SUBTASK_STORY: A short narrative (1–2 paragraphs) that describes the subtask from beginning to end in time order.

Now write ONLY the sections described above, in the required order, with no extra text before or after.
Remember: no JSON, no curly braces.
"""
    return prompt


# ---------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------

def generate_guidance_lines_for_subtask(
    video_index: str,
    title: str,
    task_obj: Dict[str, Any],
    subtask_obj: Dict[str, Any],
    tokenizer,
    model,
) -> Optional[List[str]]:
    """
    Call the LLM to generate sectioned guidance text, then split into lines
    so we can store as a list in JSON (no literal '\n' clutter).
    """
    frames = subtask_obj.get("frames") or []
    if not frames:
        # No frames, nothing to do
        return None

    prompt = build_subtask_prompt(video_index, title, task_obj, subtask_obj)

    messages = [
        {
            "role": "system",
            "content": "You are a precise, concise assistant that follows formatting instructions exactly.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Apply chat template for Qwen-style models
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # deterministic, and avoids top_k/top_p warnings
            )
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA OOM during generation; skipping this subtask.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    # Remove the prompt tokens from the output (keep only the generated tail)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    text = text.strip()

    # Split into lines; store as list in JSON
    lines = [line.rstrip() for line in text.splitlines()]

    # Optionally collapse excessive empty lines (not strictly necessary)
    cleaned_lines: List[str] = []
    empty_streak = 0
    for line in lines:
        if line.strip() == "":
            empty_streak += 1
            # allow at most 2 consecutive blank lines
            if empty_streak <= 2:
                cleaned_lines.append("")
        else:
            empty_streak = 0
            cleaned_lines.append(line)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cleaned_lines if cleaned_lines else None


# ---------------------------------------------------------------------
# PROCESSING ONE VIDEO JSON
# ---------------------------------------------------------------------

def process_video_file(path: Path, tokenizer, model):
    """
    Read one frame-captions JSON, attach guidance_text as a list of lines
    for each subtask that has frames, and write to OUTPUT_DIR.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_index = data.get("video_index") or path.stem
    title = data.get("title", "")

    print(f"[INFO] Processing {video_index}.json")

    tasks = data.get("tasks") or []
    for ti, task in enumerate(tasks):
        subtasks = task.get("subtasks") or []
        for si, sub in enumerate(subtasks):
            frames = sub.get("frames") or []
            if not frames:
                continue

            print(f"  [LLM] video={video_index}, task={ti}, subtask={si}, frames={len(frames)}")
            guidance_lines = generate_guidance_lines_for_subtask(
                video_index=video_index,
                title=title,
                task_obj=task,
                subtask_obj=sub,
                tokenizer=tokenizer,
                model=model,
            )

            if guidance_lines is None:
                continue

            # Store as list-of-lines instead of a single big string
            sub["guidance_text"] = guidance_lines

    # Write updated JSON for this video
    out_path = OUTPUT_DIR / path.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    tokenizer, model = load_model_and_tokenizer()

    # If specific json files are given on the command line, process ONLY those.
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        print(f"[INFO] Running ONLY on {len(selected)} provided files under {FRAME_CAPTION_DIR}")

        for name in selected:
            path = FRAME_CAPTION_DIR / name
            if not path.exists():
                print(f"[WARN] File not found: {path}")
                continue

            try:
                process_video_file(path, tokenizer, model)
            except Exception as e:
                print(f"[ERROR] Failed processing {name}: {e}")

        print("[INFO] Done selected videos.")
        return

    # Otherwise: original behavior — process ALL files
    if not FRAME_CAPTION_DIR.exists():
        print(f"[ERROR] FRAME_CAPTION_DIR does not exist: {FRAME_CAPTION_DIR}")
        return

    files = sorted(p for p in FRAME_CAPTION_DIR.iterdir() if p.is_file() and p.suffix == ".json")
    if not files:
        print(f"[WARN] No JSON files found under {FRAME_CAPTION_DIR}")
        return

    print(f"[INFO] Found {len(files)} frame-caption JSON files.")

    for path in files:
        try:
            process_video_file(path, tokenizer, model)
        except Exception as e:
            print(f"[ERROR] Failed processing {path.name}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
