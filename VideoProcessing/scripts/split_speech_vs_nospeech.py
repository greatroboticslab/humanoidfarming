#!/usr/bin/env python3
import json
from pathlib import Path

PROJECT = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing")
RESULTS = PROJECT / "results"
VIDEOS = PROJECT / "data" / "Videos"

# How much text (chars) counts as "has speech"
THRESHOLD_CHARS = 30  # tweak if needed

with_speech = []
no_speech = []

for json_path in sorted(RESULTS.glob("*.json")):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"!! Failed to read {json_path.name}: {e}")
        continue

    # Prefer segments; fallback to top-level "text"
    seg_text = ""
    if "segments" in data and isinstance(data["segments"], list):
        seg_text = " ".join((s.get("text") or "") for s in data["segments"])
    else:
        seg_text = data.get("text", "") or ""

    # Clean + count
    cleaned = "".join(ch for ch in seg_text if not ch.isspace())
    total = len(cleaned)

    stem = json_path.stem
    mp4_path = VIDEOS / f"{stem}.mp4"

    if total >= THRESHOLD_CHARS:
        with_speech.append(str(mp4_path))
    else:
        no_speech.append(str(mp4_path))

# Write lists
(PROJECT / "videos_with_speech.txt").write_text("\n".join(with_speech) + "\n", encoding="utf-8")
(PROJECT / "videos_no_speech.txt").write_text("\n".join(no_speech) + "\n", encoding="utf-8")

print(f"Videos with speech: {len(with_speech)}")
print(f"Videos with no/very little speech: {len(no_speech)}")
