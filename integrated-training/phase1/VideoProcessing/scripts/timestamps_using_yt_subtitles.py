import json, os, re, subprocess, sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import srt

TASKS_DIR = Path("s1_baseline/output/tasks")              # where your task JSONs live
OUT_DIR   = Path("data/transcripts")               # where to save .srt + .json
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: set to your cookies file if YouTube blocks you (leave None otherwise)
COOKIES = os.environ.get("YTDLP_COOKIES") or None  # e.g., export YTDLP_COOKIES=/path/to/cookies.txt

def extract_video_id(url_or_id: str) -> str:
    """
    Accepts either a full YouTube URL or an ID-like string.
    Returns the YouTube video ID if found, else raises.
    """
    # If looks like a URL, parse ?v=
    if url_or_id.startswith("http"):
        q = parse_qs(urlparse(url_or_id).query)
        if "v" in q and q["v"]:
            return q["v"][0]
        # also handle youtu.be short links
        netloc = urlparse(url_or_id).netloc
        if "youtu.be" in netloc:
            return urlparse(url_or_id).path.strip("/")

    # Fallback: assume it's already an ID
    vid = url_or_id.strip()
    if vid:
        return vid
    raise ValueError(f"Could not extract video id from: {url_or_id}")

def srt_path_for(video_id: str) -> Path:
    # yt-dlp will write: data/transcripts/<id>.en.srt
    return OUT_DIR / f"{video_id}.en.srt"

def json_path_for(video_id: str) -> Path:
    return OUT_DIR / f"{video_id}.json"

def download_auto_captions(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--skip-download",
        "--sub-format", "srt",
        "--convert-subs", "srt",
        "-o", str(OUT_DIR / "%(id)s.%(ext)s"),
        url,
    ]
    if COOKIES:
        cmd[1:1] = ["--cookies", COOKIES]  # insert after yt-dlp
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=False)

def srt_to_json(srt_file: Path, out_json: Path):
    with srt_file.open("r", encoding="utf-8") as f:
        srt_data = f.read()
    subs = list(srt.parse(srt_data))
    segments = []
    for i, sub in enumerate(subs):
        segments.append({
            "id": i,
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "text": sub.content.strip()
        })
    obj = {
        "text": " ".join(s.content.strip() for s in subs),
        "segments": segments,
        "language": "en"
    }
    out_json.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"✅ wrote {len(segments)} segments → {out_json}")

def process_task_file(task_json: Path):
    data = json.loads(task_json.read_text(encoding="utf-8"))
    # Prefer URL, else fall back to "index"
    url = data.get("url", "")
    idx  = data.get("index", "")
    video_id = extract_video_id(url or idx)

    # 1) fetch captions (no video download)
    download_auto_captions(video_id)

    # 2) convert SRT → JSON
    srt_file = srt_path_for(video_id)
    if not srt_file.exists():
        print(f"⚠️ no captions found for {video_id} (file missing: {srt_file.name})")
        return
    srt_to_json(srt_file, json_path_for(video_id))

def main():
    files = sorted(TASKS_DIR.glob("*.json"))
    if not files:
        print(f"no task jsons in {TASKS_DIR}")
        sys.exit(1)
    for fp in files:
        print(f"\n=== processing {fp.name} ===")
        try:
            process_task_file(fp)
        except Exception as e:
            print(f"❌ failed {fp.name}: {e}")

if __name__ == "__main__":
    main()

