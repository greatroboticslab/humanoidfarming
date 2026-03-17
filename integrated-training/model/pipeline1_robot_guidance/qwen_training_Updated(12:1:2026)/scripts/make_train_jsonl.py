#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def join_guidance(g):
    if g is None:
        return None
    if isinstance(g, list):
        return "\n".join([str(x) for x in g]).strip()
    return str(g).strip()

def build_prompt(video):
    # Minimal, stable prompt for training:
    # Inference-time you can feed the same structure.
    return (
        "You are a precise assistant that writes robot-centric guidance with strict section headings.\n"
        "Return EXACTLY these headings in order:\n"
        "GLOBAL_SUMMARY:\n"
        "FRAME_BASED_OBSERVATIONS:\n"
        "INTEGRATED_SCENE_UNDERSTANDING:\n"
        "PRECONDITIONS_FOR_ROBOT:\n"
        "SUCCESS_CRITERIA:\n"
        "ORDERED_ROBOT_ACTION_STEPS:\n"
        "SUBTASK_STORY:\n"
        "\n"
        "Constraints:\n"
        "- Steps must be numbered: k. [type=<navigation|manipulation|perception|communication>, frames=[...]] ...\n"
        "- Use only listed frame_index values (or frames=[all]).\n"
        "- Include at least one verify/check/confirm/read/measure/cross-check.\n"
        "- Keep content grounded in captions.\n"
        "\n"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory with results/subtask_guidance/*.json (or your guidance outputs)")
    ap.add_argument("--out_jsonl", required=True, help="Output train.jsonl path")
    ap.add_argument("--max_frames", type=int, default=8)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {in_dir}")

    n_written = 0
    with out_path.open("w", encoding="utf-8") as w:
        for fp in files:
            video = json.loads(fp.read_text(encoding="utf-8"))
            video_index = video.get("video_index", fp.stem)
            title = video.get("title", "")
            url = video.get("url", "")

            for task in (video.get("tasks") or []):
                task_text = (task.get("task") or "").strip()
                for sub in (task.get("subtasks") or []):
                    sub_text = (sub.get("text") or "").strip()
                    start = sub.get("start")
                    end = sub.get("end")

                    frames = sub.get("frames") or []
                    frames_sorted = sorted(frames, key=lambda f: int(f.get("frame_index", 0)))
                    frames_sorted = frames_sorted[: args.max_frames]

                    frame_lines = []
                    allowed_idxs = []
                    for f in frames_sorted:
                        idx = int(f.get("frame_index", 0))
                        cap = (f.get("caption") or "").strip()
                        rel = (f.get("relative_path") or "").strip()
                        allowed_idxs.append(idx)
                        frame_lines.append(f"- frame_index={idx}, file={rel}: {cap}")

                    guidance = join_guidance(sub.get("guidance_text"))
                    if not guidance:
                        continue
                    if not frame_lines:
                        continue

                    time_range = "unknown"
                    try:
                        if start is not None and end is not None:
                            time_range = f"{float(start):.1f}s–{float(end):.1f}s"
                    except Exception:
                        pass

                    user_prompt = (
                        build_prompt(video)
                        + f"VIDEO\n- video_index: {video_index}\n- title: {title}\n- url: {url}\n\n"
                        + f"TASK: {task_text}\n"
                        + f"SUBTASK: {sub_text}\n"
                        + f"TIME_RANGE: {time_range}\n\n"
                        + "FRAMES (captions):\n"
                        + "\n".join(frame_lines)
                        + "\n\nNow write the guidance.\n"
                    )

                    # Chat format for Qwen instruct
                    record = {
                        "messages": [
                            {"role": "system", "content": "You are a precise assistant that follows formatting instructions exactly."},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": guidance},
                        ]
                    }
                    w.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"[OK] wrote {n_written} examples to {out_path}")

if __name__ == "__main__":
    main()
