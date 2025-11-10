#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align tasks & subtasks in s1_baseline/tasks/*.json to caption timestamps in
Transcript_with_timestamp/transcripts/<video_id>.json (or data/transcripts).

Output: s1_baseline/tasks_with_timestamps/<same_name>.json
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from rapidfuzz import fuzz, process
from unidecode import unidecode

# -------- Paths (adjust if needed) --------
TASKS_DIR = Path("s1_baseline/output/tasks")
TRANS_DIRS: List[Path] = Path("transcripts")
OUT_DIR = Path("tasks_with_timestamps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Matching params --------
SUBTASK_MATCH_THRESHOLD = 55.0
TASK_MATCH_THRESHOLD    = 60.0


# ----------------- Utils -----------------
def extract_video_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    s = url_or_id.strip()

    if s.startswith("http"):
        u = urlparse(s)
        q = parse_qs(u.query)
        if "v" in q and q["v"]:
            return q["v"][0]
        if "youtu.be" in u.netloc:
            return u.path.strip("/")
    return s


def clean_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.replace("} \\\\", " ").replace("} \\", " ").replace("}\\", " ")
    s = s.replace("\\\\", " ").replace("\\", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm(s: str) -> str:
    s = unidecode(s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_transcript(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """Return list of segments: [{id,start,end,text,ntext}, ...] or None."""
    jpath: Optional[Path] = None
    for base in TRANS_DIRS:
        p = base / f"{video_id}.json"
        if p.exists():
            jpath = p
            break
    if jpath is None:
        return None

    try:
        data = json.loads(jpath.read_text(encoding="utf-8"))
    except Exception:
        return None

    segs: List[Dict[str, Any]] = []
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        return None
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        txt = clean_text(seg.get("text", ""))
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except Exception:
            continue
        segs.append({
            "id": seg.get("id"),
            "start": start,
            "end": end,
            "text": txt,
            "ntext": norm(txt),
        })
    return segs


def best_match(text: str, segs: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return (segment, score) for best fuzzy match."""
    target = norm(clean_text(text))
    if not target or not segs:
        return None, 0.0

    # Build a simple list of strings (no tuples/Paths).
    choices_text: List[str] = [str(s.get("ntext", "")) for s in segs]
    # RapidFuzz returns (match_text, score, index)
    res = process.extractOne(target, choices_text, scorer=fuzz.token_set_ratio)
    if not res:
        return None, 0.0
    match_text, score, match_index = res
    if match_index is None:
        return None, 0.0
    idx = int(match_index)
    if not (0 <= idx < len(segs)):
        return None, 0.0
    return segs[idx], float(score)


def iter_subtasks(task_obj: Dict[str, Any]) -> List[str]:
    """Return subtask strings from a task object."""
    out: List[str] = []
    subs = task_obj.get("subtasks", [])
    if not isinstance(subs, list):
        return out
    for st in subs:
        if isinstance(st, str):
            out.append(clean_text(st))
        elif isinstance(st, dict):
            out.append(clean_text(st.get("text") or st.get("description") or ""))
        else:
            out.append(clean_text(str(st)))
    return out


# --------------- Core logic ---------------
def align_file(task_path: Path) -> Tuple[int, int]:
    try:
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ {task_path.name}: failed to read/parse ({e})")
        return 0, 0

    video_id = extract_video_id(str(data.get("url") or data.get("index") or ""))
    if not video_id:
        print(f"⚠️  {task_path.name}: no url/index → cannot determine video id")
        return 0, 0

    segs = load_transcript(video_id)
    if not segs:
        exp = str(TRANS_DIRS[0] / f"{video_id}.json")  # string, not Path division in f-string
        print(f"⚠️  {task_path.name}: transcript not found for {video_id} (expected {exp})")
        return 0, 0

    tasks_in = data.get("tasks", [])
    if not isinstance(tasks_in, list):
        print(f"⚠️  {task_path.name}: 'tasks' is not a list")
        return 0, 0

    tasks_out: List[Dict[str, Any]] = []
    matched = 0
    total = 0

    for t in tasks_in:
        if not isinstance(t, dict):
            # skip malformed entries
            continue
        task_name = clean_text(t.get("task", ""))

        # --- align subtasks ---
        subs = iter_subtasks(t)
        aligned_subs: List[Dict[str, Any]] = []
        for sub in subs:
            total += 1
            seg, score = best_match(sub, segs)
            if seg and score >= SUBTASK_MATCH_THRESHOLD:
                aligned_subs.append({
                    "text": sub,
                    "segment_id": seg.get("id"),
                    "match_score": float(score),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                })
                matched += 1
            else:
                aligned_subs.append({
                    "text": sub,
                    "segment_id": None,
                    "match_score": float(score),
                    "start": None,
                    "end": None,
                })

        # --- align task text ---
        task_seg, task_score = (None, 0.0)
        if task_name:
            task_seg, task_score = best_match(task_name, segs)

        # Decide task time window
        task_start = None
        task_end = None
        time_method = "none"

        if task_seg and task_score >= TASK_MATCH_THRESHOLD:
            task_start = float(task_seg.get("start"))
            task_end   = float(task_seg.get("end"))
            time_method = "task_text_match"
        else:
            starts = [float(s["start"]) for s in aligned_subs if s["start"] is not None]
            ends   = [float(s["end"])   for s in aligned_subs if s["end"]   is not None]
            if starts and ends:
                task_start = min(starts)
                task_end   = max(ends)
                time_method = "subtasks_span"

        tasks_out.append({
            "task": task_name,
            "task_start": task_start,
            "task_end": task_end,
            "task_match_score": float(task_score),
            "task_time_method": time_method,
            "subtasks": aligned_subs,
        })

    out_obj = {
        "index": data.get("index"),
        "title": data.get("title"),
        "url": data.get("url"),
        "category": data.get("category"),
        "video_id": video_id,
        "tasks": tasks_out,
    }

    out_path = OUT_DIR / task_path.name
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"✅ {task_path.name}: matched {matched}/{total} subtasks → {out_path}")
    return matched, total


def main() -> None:
    if not TASKS_DIR.exists():
        print(f"no task dir: {str(TASKS_DIR.resolve())}")
        return

    files = sorted(TASKS_DIR.glob("*.json"))
    if not files:
        print(f"no task jsons in {str(TASKS_DIR)}")
        return

    grand_matched = 0
    grand_total = 0
    for fp in files:
        try:
            m, t = align_file(fp)
            grand_matched += m
            grand_total += t
        except Exception as e:
            print(f"❌ {fp.name}: {e}")

    if grand_total > 0:
        rate = 100.0 * grand_matched / grand_total
        print(f"\nOverall: matched {grand_matched}/{grand_total} ({rate:.1f}%)")
    else:
        print("\nNo subtasks found to align.")


if __name__ == "__main__":
    main()
