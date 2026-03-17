#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path

# === CONFIGURE THESE THREE PATHS ===
JSON_DIR = Path("results/tasks_with_timestamps")
VIDEO_DIR = Path("data/Videos_with_speech")
OUTPUT_DIR = Path("results/frame_extractions")

# Try these video extensions, in order
VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".avi"]

# Dynamic frame sampling per subtask
MIN_FRAMES_PER_SUBTASK = 3           # always get at least this many frames
MAX_FRAMES_PER_SUBTASK = 20          # safety cap so it doesn't explode
TARGET_SECONDS_BETWEEN_FRAMES = 5.0  # ~1 frame every 5 seconds

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
            print(
                f"[WARN] ffmpeg failed for {video_path} at {timestamp:.3f}s:\n"
                f"       {result.stderr.splitlines()[-1] if result.stderr else ''}"
            )
            return False
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Load the module or add it to PATH.")
        return False


def pick_timestamps(subtask: dict) -> list[float]:
    """
    Pick representative timestamps for a subtask, in a human-like way:

    - If we have both start and end and a positive duration:
        • At least 3 timestamps: near start, middle, end
        • More timestamps for longer subtasks, about every TARGET_SECONDS_BETWEEN_FRAMES
        • Capped by MAX_FRAMES_PER_SUBTASK

    - If we only have start or only end:
        • Place a few frames around that time, spaced by ~TARGET_SECONDS_BETWEEN_FRAMES.
    """
    st = subtask.get("start")
    en = subtask.get("end")

    if st is not None:
        st = float(st)
    if en is not None:
        en = float(en)

    times: list[float] = []

    # Case 1: both start and end, and positive duration
    if st is not None and en is not None and en > st:
        duration = en - st

        # approximate "one frame every ~TARGET_SECONDS_BETWEEN_FRAMES seconds"
        est_frames = int(duration / TARGET_SECONDS_BETWEEN_FRAMES) + 1

        # ensure at least min, at most max
        n_frames = max(
            MIN_FRAMES_PER_SUBTASK,
            min(MAX_FRAMES_PER_SUBTASK, est_frames),
        )

        if n_frames == 1:
            # super short segment: just middle
            times = [(st + en) / 2.0]
        elif n_frames == 2:
            # start + end
            times = [st, en]
        else:
            # uniformly spaced from start to end
            step = duration / (n_frames - 1)
            times = [st + i * step for i in range(n_frames)]

    # Case 2: both present but weird (end <= start) → collapse around start
    elif st is not None:
        center = st
        n_frames = MIN_FRAMES_PER_SUBTASK
        offsets = [
            (i - (n_frames - 1) / 2.0) * TARGET_SECONDS_BETWEEN_FRAMES
            for i in range(n_frames)
        ]
        times = [max(0.0, center + off) for off in offsets]

    # Case 3: only end is given
    elif en is not None:
        center = en
        n_frames = MIN_FRAMES_PER_SUBTASK
        offsets = [
            (i - (n_frames - 1) / 2.0) * TARGET_SECONDS_BETWEEN_FRAMES
            for i in range(n_frames)
        ]
        times = [max(0.0, center + off) for off in offsets]

    else:
        # no usable timestamps
        return []

    # normalize: ensure unique + sorted floats
    times = sorted({float(t) for t in times})
    return times


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
        print(
            f"[INFO] Transcript marked irrelevant: {data.get('reason', '')}. "
            f"Skipping frame extraction."
        )
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
            timestamps = pick_timestamps(sub)

            if not timestamps:
                print(
                    f"[WARN] No timestamps for video {video_index}, "
                    f"task {ti}, subtask {si}. Skipping."
                )
                continue

            for fi, ts in enumerate(timestamps):
                out_name = f"task{ti:02d}_sub{si:02d}_f{fi:02d}.jpg"
                out_path = video_out_dir / out_name

                success = extract_frame_ffmpeg(video_path, ts, out_path)
                if success:
                    print(f"[OK] Extracted frame -> {out_path} (t={ts:.2f}s)")
                else:
                    print(
                        f"[FAIL] Could not extract frame for "
                        f"{video_index} t={ts:.2f}s"
                    )


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
