#!/usr/bin/env python3
"""
Generate "Task Blueprint" JSONs (robot execution to-do list) from thread_logic outputs.

Input : results/thread_logic/*.json
Output: results/task_blueprints/<video_id>.json

Blueprint contains:
- tasks[]: task_index, task_text, subtasks[] (timestamped)
- execution_order[]: globally time-sorted list of subtask refs + timestamps
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default=1e18) -> float:
    try:
        return float(x)
    except Exception:
        return default


def collect_tasks_from_threads(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Thread-logic JSON schema:
      data["threads_with_logic"][...]["subtasks"] -> each subtask has task_index, sub_index, task_text, text, start/end, segment_ids
    """
    task_map: Dict[int, Dict[str, Any]] = {}
    for th in (data.get("threads_with_logic") or []):
        for s in (th.get("subtasks") or []):
            ti = safe_int(s.get("task_index", 0))
            ttext = (s.get("task_text", "") or "").strip()
            task_map.setdefault(ti, {"task_text": ttext, "subtasks": []})
            if not task_map[ti]["task_text"] and ttext:
                task_map[ti]["task_text"] = ttext
            task_map[ti]["subtasks"].append(s)

    # sort subtasks inside each task
    for ti in list(task_map.keys()):
        subs = task_map[ti]["subtasks"]
        subs.sort(key=lambda z: (safe_float(z.get("start")), safe_int(z.get("sub_index", 0))))
        task_map[ti]["subtasks"] = subs
    return task_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/thread_logic")
    ap.add_argument("--output_dir", required=True, help="results/task_blueprints")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No JSON files found in {in_dir}")
        return

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        vid = data.get("index", p.stem)

        task_map = collect_tasks_from_threads(data)
        task_indices = sorted(task_map.keys()) if task_map else [0]

        tasks_out: List[Dict[str, Any]] = []
        execution_order: List[Dict[str, Any]] = []

        for ti in task_indices:
            t = task_map.get(ti, {"task_text": "", "subtasks": []})
            subtasks_out = []
            for s in (t.get("subtasks") or []):
                si = safe_int(s.get("sub_index", 0))
                entry = {
                    "task_index": ti,
                    "task_text": (t.get("task_text") or "").strip(),
                    "sub_index": si,
                    "text": (s.get("text", "") or "").strip(),
                    "start": s.get("start", None),
                    "end": s.get("end", None),
                    "segment_ids": s.get("segment_ids", []) or [],
                }
                subtasks_out.append(entry)

                execution_order.append({
                    "task_index": ti,
                    "sub_index": si,
                    "start": entry["start"],
                    "end": entry["end"],
                    "segment_ids": entry["segment_ids"],
                    "text": entry["text"],
                })

            tasks_out.append({
                "task_index": ti,
                "task_text": (t.get("task_text") or "").strip(),
                "subtasks": subtasks_out,
            })

        # global time order
        def time_key(x):
            st = x.get("start", None)
            if isinstance(st, (int, float)):
                return float(st)
            seg = x.get("segment_ids", [])
            if seg:
                try:
                    return float(seg[0])
                except Exception:
                    pass
            return 1e18

        execution_order.sort(key=time_key)

        out = {
            "index": vid,
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "source": "thread_logic",
            "tasks": tasks_out,
            "execution_order": execution_order,
        }

        out_path = out_dir / f"{vid}.json"
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[OK] {p.name} -> {out_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
