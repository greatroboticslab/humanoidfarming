import json
import sys
from pathlib import Path

def normalize_guidance_text(g):
    """Convert guidance_text (list or string) into clean multiline text."""
    if g is None:
        return ""
    if isinstance(g, list):
        return "\n".join(line.rstrip() for line in g)
    if isinstance(g, str):
        return g
    return str(g)

def process_file(json_path, out_dir):
    with json_path.open() as f:
        data = json.load(f)

    video_index = data.get("video_index", json_path.stem)
    title = data.get("title", "No Title")

    # build txt output path
    out_txt_path = out_dir / f"{video_index}.txt"

    # write txt file
    with out_txt_path.open("w", encoding="utf-8") as out:
        out.write(f"VIDEO: {video_index}\n")
        out.write(f"TITLE: {title}\n")
        out.write("="*70 + "\n\n")

        for ti, task in enumerate(data.get("tasks", [])):
            out.write(f"TASK {ti}: {task.get('task')}\n")
            out.write(f"Time Range: {task.get('start')} → {task.get('end')}\n")
            out.write("-"*70 + "\n")

            for si, sub in enumerate(task.get("subtasks", [])):
                out.write(f"  SUBTASK {si}: {sub.get('text')}\n")
                out.write(f"  Time Range: {sub.get('start')} → {sub.get('end')}\n")
                out.write("  Guidance:\n")

                gtext = normalize_guidance_text(sub.get("guidance_text"))

                for line in gtext.split("\n"):
                    out.write(f"    {line}\n")

                out.write("\n")

            out.write("\n" + "="*70 + "\n\n")

    # update JSON to include reference to txt file
    data["guidance_text_file"] = str(out_txt_path.relative_to(json_path.parent))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Exported: {json_path.name} → {out_txt_path.name}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_export_all_videos.py <folder_with_json>")
        sys.exit(1)

    json_dir = Path(sys.argv[1])
    if not json_dir.is_dir():
        print(f"ERROR: Directory not found: {json_dir}")
        sys.exit(1)

    out_dir = json_dir / "final_guidance_txt"
    out_dir.mkdir(exist_ok=True)

    for json_path in sorted(json_dir.glob("*.json")):
        process_file(json_path, out_dir)

    print("\nDONE: All JSONs processed and TXT files created.")

if __name__ == "__main__":
    main()
