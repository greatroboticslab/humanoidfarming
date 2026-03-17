#!/usr/bin/env python3
"""
Training-quality log generator (Mission → Sub-missions)

Reads:
  - results/thread_logic/<vid>.json
  - results/coherent_blocks_with_submissions/<vid>.json
  - results/check_reports/<vid>_logical_check.json
  - results/check_reports/<vid>_coherence_check.json  (optional)
  - results/task_blueprints/<vid>.json                (optional existence)

Writes:
  - results/training_quality_log_submissions/training_quality_log.json
  - results/training_quality_log_submissions/missions/<vid>.json

Key behavior:
- Mission = video
- Sub-mission = coherent block (sub_mission_id) with explicit sub_mission_title
- Checks are logged at BOTH levels:
    * mission-level summary
    * sub-mission-level status + issues scoped to that block/thread
- Human decisions are logged per sub-mission:
    accept | redo | give_up
  If a sub-mission has no issues → auto_pass → successful_entry → used_for_training=true
  Else → pending → pending_entry → used_for_training=false

Assumptions about issue scoping:
- Logical issues are thread-scoped via thread_index (from logical check report).
  We map thread_index → block_id using coherent blocks "units" list (contains thread_index).
- Coherence issues are block-scoped via block_id (from coherence check report).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def summarize_issues(issues: List[Dict[str, Any]], max_items: int = 6) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for it in issues:
        t = it.get("type", "unknown")
        counts[t] = counts.get(t, 0) + 1
    return {
        "num_issues": len(issues),
        "issue_type_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "sample_issues": issues[:max_items],
    }


def build_task_texts_from_thread_logic(tl: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for th in (tl.get("threads_with_logic") or []):
        for s in (th.get("subtasks") or []):
            ti = safe_int(s.get("task_index"), -1)
            if ti < 0:
                continue
            ttext = (s.get("task_text") or "").strip()
            if ti not in out:
                out[ti] = ttext
            elif not out[ti] and ttext:
                out[ti] = ttext
    return out


def build_thread_to_tasks(tl: Dict[str, Any]) -> Dict[int, Set[int]]:
    m: Dict[int, Set[int]] = {}
    for th in (tl.get("threads_with_logic") or []):
        tix = safe_int(th.get("thread_index"), -1)
        if tix < 0:
            continue
        m.setdefault(tix, set())
        for s in (th.get("subtasks") or []):
            ti = safe_int(s.get("task_index"), -1)
            if ti >= 0:
                m[tix].add(ti)
    return m


def build_block_to_tasks(cb: Dict[str, Any]) -> Dict[int, Set[int]]:
    m: Dict[int, Set[int]] = {}
    for b in (cb.get("blocks") or []):
        bid = safe_int(b.get("block_id"), -1)
        if bid < 0:
            continue
        m.setdefault(bid, set())
        for ref in (b.get("subtask_refs") or []):
            ti = safe_int(ref.get("task_index"), -1)
            if ti >= 0:
                m[bid].add(ti)
    return m


def build_thread_to_blocks(cb: Dict[str, Any]) -> Dict[int, Set[int]]:
    """
    Uses coherent block "units" which contain thread_index.
    Returns thread_index -> {block_id,...}
    """
    m: Dict[int, Set[int]] = {}
    for b in (cb.get("blocks") or []):
        bid = safe_int(b.get("block_id"), -1)
        if bid < 0:
            continue
        for u in (b.get("units") or []):
            tix = safe_int(u.get("thread_index"), -1)
            if tix < 0:
                continue
            m.setdefault(tix, set()).add(bid)
    return m


def block_id_to_submission(cb: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    block_id -> {sub_mission_id, sub_mission_title, time_start, time_end}
    """
    out: Dict[int, Dict[str, Any]] = {}
    for b in (cb.get("blocks") or []):
        bid = safe_int(b.get("block_id"), -1)
        if bid < 0:
            continue
        out[bid] = {
            "block_id": bid,
            "sub_mission_id": b.get("sub_mission_id"),
            "sub_mission_title": b.get("sub_mission_title"),
            "time_start": b.get("time_start"),
            "time_end": b.get("time_end"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_reports_dir", required=True, help="results/check_reports")
    ap.add_argument("--thread_logic_dir", required=True, help="results/thread_logic")
    ap.add_argument("--coherent_blocks_dir", required=True, help="results/coherent_blocks_with_submissions")
    ap.add_argument("--task_blueprints_dir", required=True, help="results/task_blueprints")
    ap.add_argument("--out_dir", required=True, help="results/training_quality_log_submissions")
    args = ap.parse_args()

    cr_dir = Path(args.check_reports_dir)
    tl_dir = Path(args.thread_logic_dir)
    cb_dir = Path(args.coherent_blocks_dir)
    bp_dir = Path(args.task_blueprints_dir)
    out_dir = Path(args.out_dir)
    missions_dir = out_dir / "missions"
    missions_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logical_reports = sorted(cr_dir.glob("*_logical_check.json"))
    if not logical_reports:
        print(f"[WARN] No *_logical_check.json files found in {cr_dir}")
        return

    global_log: Dict[str, Any] = {
        "schema": "training_quality_log_submissions_v1",
        "units": "mission(video) -> sub_mission(block) -> task -> subtask",
        "allowed_human_decisions": ["accept", "redo", "give_up"],
        "auto_policy": "If no issues detected for a sub_mission, auto log as successful_entry.",
        "num_missions": 0,
        "missions": []
    }

    for lr_path in logical_reports:
        vid = lr_path.name.replace("_logical_check.json", "")

        tl_path = tl_dir / f"{vid}.json"
        cb_path = cb_dir / f"{vid}.json"
        bp_path = bp_dir / f"{vid}.json"
        coherence_path = cr_dir / f"{vid}_coherence_check.json"

        if not tl_path.exists() or not cb_path.exists():
            # skip incomplete missions (can't map thread->block without coherent blocks)
            print(f"[WARN] Skip {vid}: missing thread_logic or coherent_blocks_with_submissions")
            continue

        logical = load_json(lr_path)
        coherence = load_json(coherence_path) if coherence_path.exists() else None
        tl = load_json(tl_path)
        cb = load_json(cb_path)

        title = (cb.get("mission_title") or cb.get("title") or tl.get("title") or "").strip()
        url = (cb.get("url") or tl.get("url") or "").strip()

        logical_issues = (logical.get("issues") or [])
        coherence_issues = (coherence.get("issues") or []) if coherence else []

        # Mission-level check summary
        mission_entry: Dict[str, Any] = {
            "mission_id": vid,
            "mission_title": title,
            "url": url,
            "artifacts_present": {
                "thread_logic": True,
                "coherent_blocks_with_submissions": True,
                "task_blueprint": bp_path.exists(),
                "coherence_report": coherence_path.exists(),
            },
            "checks": {
                "logical_check": {
                    "status": "pass" if (logical.get("num_issues") == 0) else "fail",
                    **summarize_issues(logical_issues),
                },
                "coherence_check": {
                    "status": "unknown" if coherence is None else ("pass" if (coherence.get("num_issues") == 0) else "fail"),
                    **summarize_issues(coherence_issues),
                },
            },
        }

        task_texts = build_task_texts_from_thread_logic(tl)
        thread_to_tasks = build_thread_to_tasks(tl)
        block_to_tasks = build_block_to_tasks(cb)
        thread_to_blocks = build_thread_to_blocks(cb)
        bid_to_subm = block_id_to_submission(cb)

        # Build per-submission (block) issue lists by scoping
        subm_entries: List[Dict[str, Any]] = []

        # Pre-aggregate issues by block_id
        block_logical_issues: Dict[int, List[Dict[str, Any]]] = {}
        for it in logical_issues:
            tix = safe_int(it.get("thread_index"), -1)
            for bid in thread_to_blocks.get(tix, set()):
                block_logical_issues.setdefault(bid, []).append(it)

        block_coherence_issues: Dict[int, List[Dict[str, Any]]] = {}
        for it in coherence_issues:
            bid = safe_int(it.get("block_id"), -1)
            if bid >= 0:
                block_coherence_issues.setdefault(bid, []).append(it)

        # Iterate blocks as sub-missions
        for bid, meta in sorted(bid_to_subm.items(), key=lambda kv: kv[0]):
            l_issues = block_logical_issues.get(bid, [])
            c_issues = block_coherence_issues.get(bid, [])

            issues_detected = (len(l_issues) + len(c_issues)) > 0

            if not issues_detected:
                human_decision = "auto_pass"
                final_status = "successful_entry"
                used_for_training = True
            else:
                human_decision = "pending"
                final_status = "pending_entry"
                used_for_training = False

            # Task-level status within this sub-mission
            affected_tasks = set()
            # logical issues -> tasks via thread mapping (through thread_index)
            for it in l_issues:
                tix = safe_int(it.get("thread_index"), -1)
                for ti in thread_to_tasks.get(tix, set()):
                    affected_tasks.add(ti)
            # coherence issues -> tasks via block refs
            for ti in block_to_tasks.get(bid, set()):
                # only mark fail if coherence issues exist; otherwise leave pass
                if c_issues:
                    affected_tasks.add(ti)

            tasks_out = []
            # list only tasks that appear in this block (cleaner for sub-mission granularity)
            block_tasks = sorted(block_to_tasks.get(bid, set()))
            for ti in block_tasks:
                tasks_out.append({
                    "task_index": ti,
                    "task_text": task_texts.get(ti, ""),
                    "status": "fail" if ti in affected_tasks else "pass"
                })

            subm_entries.append({
                "sub_mission_id": meta.get("sub_mission_id"),
                "sub_mission_title": meta.get("sub_mission_title"),
                "block_id": bid,
                "time_start": meta.get("time_start"),
                "time_end": meta.get("time_end"),
                "issues_detected": issues_detected,
                "checks": {
                    "logical_check": {
                        "status": "pass" if len(l_issues) == 0 else "fail",
                        **summarize_issues(l_issues),
                    },
                    "coherence_check": {
                        # if coherence report missing, we still scope whatever we have (likely none)
                        "status": "pass" if len(c_issues) == 0 else "fail",
                        **summarize_issues(c_issues),
                    }
                },
                "human_decision": human_decision,   # auto_pass | pending | accept | redo | give_up
                "final_status": final_status,       # successful_entry | pending_entry | redo_entry | give_up_entry
                "used_for_training": used_for_training,
                "tasks": tasks_out
            })

        # Mission-level status derived from sub-missions (used_for_training only if all are successful_entry)
        num_success = sum(1 for s in subm_entries if s["final_status"] == "successful_entry")
        mission_entry["sub_missions"] = subm_entries
        mission_entry["mission_rollup"] = {
            "num_sub_missions": len(subm_entries),
            "num_successful_entries": num_success,
            "all_successful": (num_success == len(subm_entries) and len(subm_entries) > 0),
        }

        # per-mission file
        (missions_dir / f"{vid}.json").write_text(json.dumps(mission_entry, indent=2), encoding="utf-8")
        global_log["missions"].append({
            "mission_id": vid,
            "mission_title": title,
            "url": url,
            "num_sub_missions": len(subm_entries),
            "num_successful_entries": num_success,
            "all_successful": (num_success == len(subm_entries) and len(subm_entries) > 0),
        })

    global_log["num_missions"] = len(global_log["missions"])
    (out_dir / "training_quality_log.json").write_text(json.dumps(global_log, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {out_dir / 'training_quality_log.json'}")
    print(f"[OK] Wrote per-mission logs to: {missions_dir}")


if __name__ == "__main__":
    main()
