#!/usr/bin/env python3
"""
Generate check reports for:
- logical map (thread_logic): conflicts/needs_repair/monotonic timestamps
- coherent map (coherent_blocks): block span / mixed categories

Inputs:
  --thread_logic_dir results/thread_logic
  --coherent_blocks_dir results/coherent_blocks

Outputs:
  results/check_reports/<video_id>_logical_check.json
  results/check_reports/<video_id>_coherence_check.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def logical_check(thread_logic: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    threads = thread_logic.get("threads_with_logic", []) or []

    for th in threads:
        tix = th.get("thread_index")
        logic = th.get("logic", {}) or {}
        conflicts = logic.get("conflicts", []) or []
        repair = logic.get("repair", {}) or {}
        needs_repair = bool((repair.get("bridging_sentences") or []) or (repair.get("revised_subtasks") or []))

        # monotonic starts
        starts = []
        for s in (th.get("subtasks") or []):
            st = s.get("start", None)
            if is_num(st):
                starts.append(float(st))
        monotonic = all(starts[i] >= starts[i-1] for i in range(1, len(starts))) if starts else True

        if conflicts:
            issues.append({"thread_index": tix, "type": "conflicts", "count": len(conflicts)})
        if needs_repair:
            issues.append({"thread_index": tix, "type": "needs_repair", "details": "repair.bridging_sentences or repair.revised_subtasks non-empty"})
        if not monotonic:
            issues.append({"thread_index": tix, "type": "timestamps_non_monotonic"})

    return {
        "index": thread_logic.get("index", ""),
        "title": thread_logic.get("title", ""),
        "url": thread_logic.get("url", ""),
        "num_threads": len(threads),
        "num_issues": len(issues),
        "issues": issues,
    }


def coherence_check(coherent: Dict[str, Any], max_span_seconds: float = 240.0, max_entropy: float = 1.35) -> Dict[str, Any]:
    """
    Flags:
    - block_time_span_too_large
    - block_category_mixed (entropy high)
    """
    issues: List[Dict[str, Any]] = []

    for b in (coherent.get("blocks") or []):
        bid = b.get("block_id")
        t0 = safe_float(b.get("time_start"))
        t1 = safe_float(b.get("time_end"))
        if t0 is not None and t1 is not None:
            span = t1 - t0
            if span > max_span_seconds:
                issues.append({"block_id": bid, "type": "block_time_span_too_large", "span_seconds": round(span, 3)})

        dist = b.get("category_distribution", {}) or {}
        total = sum(int(v) for v in dist.values() if isinstance(v, (int, float)))
        if total > 0:
            # entropy in nats
            import math
            ent = 0.0
            for v in dist.values():
                if not isinstance(v, (int, float)) or v <= 0:
                    continue
                p = float(v) / float(total)
                ent -= p * math.log(p)
            if ent > max_entropy:
                issues.append({"block_id": bid, "type": "block_category_mixed", "entropy": round(ent, 3), "distribution": dist})

    return {
        "index": coherent.get("index", ""),
        "title": coherent.get("title", ""),
        "url": coherent.get("url", ""),
        "num_blocks": coherent.get("num_blocks", 0),
        "params": coherent.get("params", {}),
        "thresholds": {"max_span_seconds": max_span_seconds, "max_entropy": max_entropy},
        "num_issues": len(issues),
        "issues": issues,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread_logic_dir", required=True, help="results/thread_logic")
    ap.add_argument("--coherent_blocks_dir", required=True, help="results/coherent_blocks")
    ap.add_argument("--out_dir", required=True, help="results/check_reports")
    ap.add_argument("--max_block_span", type=float, default=240.0)
    ap.add_argument("--max_entropy", type=float, default=1.35)
    args = ap.parse_args()

    tl_dir = Path(args.thread_logic_dir)
    cb_dir = Path(args.coherent_blocks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tl_files = sorted([p for p in tl_dir.glob("*.json") if p.is_file()])
    cb_files = sorted([p for p in cb_dir.glob("*.json") if p.is_file()])

    cb_map = {}
    for p in cb_files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            cb_map[d.get("index", p.stem)] = d
        except Exception:
            continue

    if not tl_files:
        print(f"[WARN] No thread_logic JSON files found in {tl_dir}")
        return

    for p in tl_files:
        tl = json.loads(p.read_text(encoding="utf-8"))
        vid = tl.get("index", p.stem)

        logical = logical_check(tl)
        (out_dir / f"{vid}_logical_check.json").write_text(json.dumps(logical, indent=2), encoding="utf-8")

        coh = cb_map.get(vid)
        if coh:
            coherence = coherence_check(coh, max_span_seconds=args.max_block_span, max_entropy=args.max_entropy)
            (out_dir / f"{vid}_coherence_check.json").write_text(json.dumps(coherence, indent=2), encoding="utf-8")
            print(f"[OK] {p.name} -> reports written (logical+coherence)")
        else:
            print(f"[OK] {p.name} -> logical report written (no coherent_blocks match)")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
