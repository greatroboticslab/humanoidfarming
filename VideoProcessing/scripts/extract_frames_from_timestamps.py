import os
import json
import subprocess
from pathlib import Path

# === CONFIGURE THESE THREE PATHS ===
JSON_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/timestamps_using_llm")
VIDEO_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/data/Videos_with_speech")
OUTPUT_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/frame_extractions")

# Try these video extensions, in order
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi"]

# Create output dir if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_video_for_index(video_index: str) -> Path | None:
    """
    Given the 'index' from the JSON (usually the base filename),
    try to find a matching video file in VIDEO_DIR.
    """
    for ext in VIDEO_EXTS:
        candidate = VIDEO_DIR / f"{video_index}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_frame_ffmpeg(video_path: Path, timestamp: float, out_path: Path) -> bool:
    """
    Use ffmpeg to extract a single frame at 'timestamp' seconds.
    Returns True on success, False on failure.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-y",
        str(out_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            print(f"[WARN] ffmpeg failed for {video_path} at {timestamp:.3f}s:\n"
                  f"       {result.stderr.splitlines()[-1] if result.stderr else ''}")
            return False
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Load the module or add it to PATH.")
        return False


def pick_timestamp(subtask: dict) -> float | None:
    """
    Pick a representative timestamp for a subtask.
    Prefer the midpoint of [start, end]. If only one is present, use that.
    """
    st = subtask.get("start")
    en = subtask.get("end")

    if st is None and en is None:
        return None
    if st is not None and en is not None:
        if en <= st:
            return float(st)
        return float((st + en) / 2.0)
    if st is not None:
        return float(st)
    if en is not None:
        return float(en)
    return None


def process_json_file(json_path: Path):
    print(f"[INFO] Processing JSON: {json_path.name}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_index = data.get("index") or json_path.stem
    relevant = data.get("relevant", True)

    video_path = find_video_for_index(video_index)
    if video_path is None:
        print(f"[WARN] No video found for index '{video_index}'. Skipping.")
        return

    if not relevant:
        print(f"[INFO] Transcript marked irrelevant: {data.get('reason', '')}. Skipping frame extraction.")
        return

    tasks = data.get("tasks", [])
    if not tasks:
        print("[WARN] No tasks in JSON. Nothing to extract.")
        return

    video_out_dir = OUTPUT_DIR / video_index
    video_out_dir.mkdir(parents=True, exist_ok=True)

    for ti, task in enumerate(tasks):
        subtasks = task.get("subtasks", []) or []

        for si, sub in enumerate(subtasks):
            ts = pick_timestamp(sub)
            if ts is None:
                print(f"[WARN] No timestamp for video {video_index}, task {ti}, subtask {si}. Skipping.")
                continue

            out_name = f"task{ti:02d}_sub{si:02d}.jpg"
            out_path = video_out_dir / out_name

            success = extract_frame_ffmpeg(video_path, ts, out_path)
            if success:
                print(f"[OK] Extracted frame -> {out_path} (t={ts:.2f}s)")
            else:
                print(f"[FAIL] Could not extract frame for {video_index} t={ts:.2f}s")


def main():
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {JSON_DIR}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files.")
    for jp in json_files:
        process_json_file(jp)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
