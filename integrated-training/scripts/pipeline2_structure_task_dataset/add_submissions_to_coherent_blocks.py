#!/usr/bin/env python3
"""
Add explicit Sub-mission (theme) names to coherent blocks.

Input:
  results/coherent_blocks/*.json

Output:
  results/coherent_blocks_with_submissions/<video_id>.json
  If an input file contains a LIST of missions, writes one output per mission.

Adds per block:
  - sub_mission_id: "<video_id>_subm<block_id>"
  - sub_mission_title: deterministic theme label derived from existing JSON words only
  - sub_mission_time: {start, end}

Theme naming policy (no LLM):
  - Uses block_preview_text + referenced task_texts (if present in task_categories)
  - Extracts top keywords (frequency-based, stopword filtered)
  - Produces: "Theme: kw1 / kw2 / kw3"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","as","at","from",
    "is","are","was","were","be","been","being","it","this","that","these","those",
    "we","you","they","he","she","i","our","your","their","its",
    "note","mention","highlight","observe","point","out","show","explain","discuss",
    "importance","significance","role","expected","increase",
    # domain-common words that dominate
    "market","testing","soil"
}


def tokenize(text: str) -> List[str]:
    t = (text or "").lower()
    return re.findall(r"[a-z][a-z\-]+", t)


def top_keywords(texts: List[str], k: int = 3) -> List[str]:
    freq: Dict[str, int] = {}
    for tx in texts:
        for w in tokenize(tx):
            if w in STOPWORDS:
                continue
            if len(w) <= 2:
                continue
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return []
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in items[:k]]


def build_task_text_map(data: Dict[str, Any]) -> Dict[int, str]:
    """
    task_index -> task_text from task_categories (if present).
    """
    task_texts: Dict[int, str] = {}
    for t in (data.get("task_categories") or []):
        try:
            ti = int(t.get("task_index"))
        except Exception:
            continue
        task_texts[ti] = (t.get("task_text") or "").strip()
    return task_texts


def process_one_mission(data: Dict[str, Any], out_dir: Path, topk: int) -> str:
    vid = data.get("index") or data.get("mission_id") or "unknown"
    title = data.get("title", "")

    task_texts = build_task_text_map(data)

    out_blocks = []
    for b in (data.get("blocks") or []):
        bid = b.get("block_id")
        preview = (b.get("block_preview_text") or "").strip()

        # Collect referenced task texts (if available)
        ref_task_indices = sorted({
            int(r.get("task_index"))
            for r in (b.get("subtask_refs") or [])
            if r.get("task_index") is not None
        })
        ref_task_texts = [task_texts.get(ti, "") for ti in ref_task_indices if task_texts.get(ti, "")]

        kws = top_keywords([preview] + ref_task_texts, k=topk)

        if kws:
            sub_title = "Theme: " + " / ".join(kws)
        else:
            # fallback: still only using existing JSON words (preview snippet)
            if preview:
                sub_title = "Theme: " + (preview[:60] + ("..." if len(preview) > 60 else ""))
            else:
                sub_title = f"Theme: block_{bid}"

        out_b = dict(b)
        out_b["sub_mission_id"] = f"{vid}_subm{bid}"
        out_b["sub_mission_title"] = sub_title
        out_b["sub_mission_time"] = {"start": b.get("time_start"), "end": b.get("time_end")}
        out_blocks.append(out_b)

    out = dict(data)
    out["mission_id"] = vid
    out["mission_title"] = title
    out["sub_missions"] = [{
        "sub_mission_id": b["sub_mission_id"],
        "sub_mission_title": b["sub_mission_title"],
        "time_start": b.get("time_start"),
        "time_end": b.get("time_end"),
        "block_id": b.get("block_id"),
    } for b in out_blocks]
    out["blocks"] = out_blocks

    out_path = out_dir / f"{vid}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return vid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/coherent_blocks")
    ap.add_argument("--output_dir", required=True, help="results/coherent_blocks_with_submissions")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No json files found in {in_dir}")
        return

    for p in files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skip unreadable JSON: {p.name} ({e})")
            continue

        if isinstance(obj, dict):
            vid = process_one_mission(obj, out_dir, args.topk)
            print(f"[OK] {p.name} -> sub_missions={len(obj.get('blocks', []) or [])} (mission={vid})")
        elif isinstance(obj, list):
            # If list of dict missions, process each
            if obj and all(isinstance(x, dict) for x in obj):
                for i, mission in enumerate(obj):
                    vid = process_one_mission(mission, out_dir, args.topk)
                    print(f"[OK] {p.name}[{i}] -> mission={vid}")
            else:
                print(f"[WARN] Skip {p.name}: top-level JSON is a list but not list[dict].")
        else:
            print(f"[WARN] Skip {p.name}: top-level JSON is {type(obj).__name__}, expected dict or list[dict].")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
