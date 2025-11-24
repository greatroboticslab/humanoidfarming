import json
from pathlib import Path

# Base paths
BASE_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing")

JSON_DIR = BASE_DIR / "results" / "timestamps_using_llm"
FRAMES_DIR = BASE_DIR / "results" / "frame_extractions"
OUT_DIR = BASE_DIR / "documents" / "html"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_timestamp(sub):
    """Return representative timestamp (midpoint) for a subtask."""
    st = sub.get("start")
    en = sub.get("end")
    if st is None and en is None:
        return None
    if st is not None and en is not None:
        st = float(st)
        en = float(en)
        if en <= st:
            return st
        return (st + en) / 2.0
    if st is not None:
        return float(st)
    if en is not None:
        return float(en)
    return None


def build_html_for_video(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_index = data.get("index") or json_path.stem
    title = data.get("title", "Untitled Video")
    url = data.get("url", "")
    tasks = data.get("tasks", []) or []

    # Relative path from HTML files to frame_extractions:
    # HTML lives in  .../VideoProcessing/documents/html
    # Frames in      .../VideoProcessing/results/frame_extractions
    # So relative path is ../../results/frame_extractions/...
    rel_frames_root = Path("../../results/frame_extractions")

    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html><head><meta charset='utf-8'>")
    html_parts.append(f"<title>{title}</title>")
    html_parts.append("""<style>
body { font-family: Arial, sans-serif; margin: 24px; }
h1 { font-size: 24px; }
h2 { font-size: 20px; margin-top: 24px; }
h3 { font-size: 16px; margin-top: 18px; }
.subtask { margin: 12px 0 24px 20px; border-left: 3px solid #ccc; padding-left: 10px; }
.subtask img { max-width: 480px; display: block; margin-top: 8px; border: 1px solid #999; }
.meta { color: #555; font-size: 13px; }
</style>""")
    html_parts.append("</head><body>")

    # Header
    html_parts.append(f"<h1>{title}</h1>")
    html_parts.append(f"<p class='meta'><strong>Video ID:</strong> {video_index}</p>")
    if url:
        html_parts.append(f"<p class='meta'><strong>URL:</strong> <a href='{url}'>{url}</a></p>")

    # Tasks & subtasks
    frames_video_dir = FRAMES_DIR / video_index
    for ti, task in enumerate(tasks):
        task_text = task.get("task", "")
        html_parts.append(f"<h2>Task {ti+1}: {task_text}</h2>")
        subtasks = task.get("subtasks", []) or []

        for si, sub in enumerate(subtasks):
            sub_text = sub.get("text", "")
            ts = pick_timestamp(sub)
            ts_str = f"{ts:.2f} s" if ts is not None else "N/A"

            html_parts.append("<div class='subtask'>")
            html_parts.append(f"<h3>Subtask {ti+1}.{si+1}</h3>")
            html_parts.append(f"<p><strong>Description:</strong> {sub_text}</p>")
            html_parts.append(f"<p class='meta'><strong>Timestamp:</strong> {ts_str}</p>")

            img_name = f"task{ti:02d}_sub{si:02d}.jpg"
            img_path_abs = frames_video_dir / img_name
            if img_path_abs.exists():
                img_rel = rel_frames_root / video_index / img_name
                html_parts.append(
                    f"<img src='{img_rel.as_posix()}' alt='{img_name}' />"
                )
            else:
                html_parts.append("<p class='meta'><em>No image available for this subtask.</em></p>")

            html_parts.append("</div>")

    html_parts.append("</body></html>")
    html_content = "\n".join(html_parts)

    out_path = OUT_DIR / f"{video_index}.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[OK] Wrote {out_path}")


def main():
    json_files = sorted(JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"[WARN] No JSON files found in {JSON_DIR}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files.")
    for jp in json_files:
        video_id = jp.stem
        frames_dir = FRAMES_DIR / video_id
        if not frames_dir.exists():
            print(f"[WARN] No frames directory for {video_id} at {frames_dir}, skipping.")
            continue
        build_html_for_video(jp)

    print("[INFO] Done generating HTML reports.")


if __name__ == "__main__":
    main()
