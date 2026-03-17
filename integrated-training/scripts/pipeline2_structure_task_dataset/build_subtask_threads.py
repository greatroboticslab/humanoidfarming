import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def flatten_subtasks(video: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat = []
    for ti, task in enumerate(video.get("tasks", []) or []):
        for si, sub in enumerate(task.get("subtasks", []) or []):
            entry = {
                "task_index": ti,
                "task_text": task.get("task", ""),
                "sub_index": si,
                "text": sub.get("text", ""),
                "start": sub.get("start", None),
                "end": sub.get("end", None),
                "segment_ids": sub.get("segment_ids", []) or [],
            }
            flat.append(entry)
    return flat


def safe_time_key(s: Dict[str, Any]) -> float:
    # Prefer start time, fall back to first segment_id, else large.
    st = s.get("start", None)
    if isinstance(st, (int, float)):
        return float(st)
    seg_ids = s.get("segment_ids", [])
    if seg_ids:
        return float(seg_ids[0])  # keeps stable order if times missing
    return 1e18


def keyword_overlap(a: str, b: str) -> float:
    # Super simple lexical overlap; good enough to start.
    # You can swap this for embeddings later.
    import re
    ta = set(re.findall(r"[a-z]+", (a or "").lower()))
    tb = set(re.findall(r"[a-z]+", (b or "").lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def group_into_threads(
    subtasks: List[Dict[str, Any]],
    max_gap_seconds: float = 25.0,
    min_overlap: float = 0.06,
    max_thread_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    Groups subtasks into sequential threads using:
      - time gap threshold
      - lightweight semantic continuity via keyword overlap
      - cap thread size to keep LLM prompts stable
    """
    threads: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        starts = [x["start"] for x in cur if isinstance(x.get("start"), (int, float))]
        ends = [x["end"] for x in cur if isinstance(x.get("end"), (int, float))]
        t0 = min(starts) if starts else None
        t1 = max(ends) if ends else None
        threads.append({
            "time_start": t0,
            "time_end": t1,
            "subtasks": cur
        })
        cur = []

    for s in subtasks:
        if not cur:
            cur.append(s)
            continue

        prev = cur[-1]
        prev_end = prev.get("end", None)
        curr_start = s.get("start", None)

        gap_ok = True
        if isinstance(prev_end, (int, float)) and isinstance(curr_start, (int, float)):
            gap = float(curr_start) - float(prev_end)
            gap_ok = gap <= max_gap_seconds

        overlap = keyword_overlap(prev.get("text", ""), s.get("text", ""))

        # Split if gap too large OR semantic continuity too low OR thread too large
        if (not gap_ok) or (overlap < min_overlap) or (len(cur) >= max_thread_size):
            flush()
            cur.append(s)
        else:
            cur.append(s)

    flush()
    return threads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/tasks_with_timestamps")
    ap.add_argument("--output_dir", required=True, help="results/subtask_threads")
    ap.add_argument("--max_gap", type=float, default=25.0)
    ap.add_argument("--min_overlap", type=float, default=0.06)
    ap.add_argument("--max_thread_size", type=int, default=10)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No json files found in {in_dir}")
        return

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))

        # Flatten + sort
        flat = flatten_subtasks(data)
        flat.sort(key=safe_time_key)

        threads = group_into_threads(
            flat,
            max_gap_seconds=args.max_gap,
            min_overlap=args.min_overlap,
            max_thread_size=args.max_thread_size
        )

        out = {
            "index": data.get("index", p.stem),
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "num_threads": len(threads),
            "threads": threads,
        }

        (out_dir / f"{out['index']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[OK] {p.name} -> threads={len(threads)}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
