#!/usr/bin/env python3
"""
Training-quality log generator (global JSON)

Goal:
- Document which missions/tasks pass logical + coherence checks.
- Auto-log "successful_entry" when no issues are detected.
- Otherwise create a human decision slot with allowed options:
    accept | redo | give_up
  and log status as:
    successful_entry | redo_entry | give_up_entry

Inputs:
  results/check_reports/<vid>_logical_check.json
  results/check_reports/<vid>_coherence_check.json   (may be missing)
  results/thread_logic/<vid>.json                    (for mission title/url + task texts)
  results/task_blueprints/<vid>.json                 (optional existence info)

Outputs:
  results/training_quality_log/training_quality_log.json
  results/training_quality_log/missions/<vid>.json

Notes:
- This script does NOT apply repairs. It only records pass/fail + decision slots.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_HUMAN_DECISIONS = {"accept", "redo", "give_up"}


def load_json(p: Path) -> Dict[str, Any]:
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


def build_task_texts(thread_logic: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """
    Extract task_index -> task_text from thread_logic subtasks.
    """
    out: Dict[int, str] = {}
    if not thread_logic:
        return out
    for th in (thread_logic.get("threads_with_logic") or []):
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


def affected_tasks_from_logical_issues(thread_logic: Optional[Dict[str, Any]], logical_issues: List[Dict[str, Any]]) -> List[int]:
    """
    If an issue has thread_index, mark tasks that appear in that thread.
    """
    if not thread_logic:
        return []
    thread_to_tasks: Dict[int, set] = {}
    for th in (thread_logic.get("threads_with_logic") or []):
        tix = safe_int(th.get("thread_index"), -1)
        if tix < 0:
            continue
        thread_to_tasks.setdefault(tix, set())
        for s in (th.get("subtasks") or []):
            ti = safe_int(s.get("task_index"), -1)
            if ti >= 0:
                thread_to_tasks[tix].add(ti)

    affected = set()
    for it in logical_issues:
        tix = safe_int(it.get("thread_index"), -1)
        for ti in thread_to_tasks.get(tix, set()):
            affected.add(ti)
    return sorted(affected)


def affected_tasks_from_coherence_issues(coherent_blocks: Optional[Dict[str, Any]], coherence_issues: List[Dict[str, Any]]) -> List[int]:
    """
    If an issue has block_id, mark tasks that appear in that block via subtask_refs.
    """
    if not coherent_blocks:
        return []
    block_to_tasks: Dict[int, set] = {}
    for b in (coherent_blocks.get("blocks") or []):
        bid = safe_int(b.get("block_id"), -1)
        if bid < 0:
            continue
        block_to_tasks.setdefault(bid, set())
        for ref in (b.get("subtask_refs") or []):
            ti = safe_int(ref.get("task_index"), -1)
            if ti >= 0:
                block_to_tasks[bid].add(ti)

    affected = set()
    for it in coherence_issues:
        bid = safe_int(it.get("block_id"), -1)
        for ti in block_to_tasks.get(bid, set()):
            affected.add(ti)
    return sorted(affected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_reports_dir", required=True, help="results/check_reports")
    ap.add_argument("--thread_logic_dir", required=True, help="results/thread_logic")
    ap.add_argument("--coherent_blocks_dir", required=True, help="results/coherent_blocks")
    ap.add_argument("--task_blueprints_dir", required=True, help="results/task_blueprints")
    ap.add_argument("--out_dir", required=True, help="results/training_quality_log")
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
        "schema": "training_quality_log_v2",
        "allowed_human_decisions": ["accept", "redo", "give_up"],
        "auto_policy": "If no issues detected, auto log as successful_entry.",
        "num_missions": 0,
        "missions": []
    }

    for lr_path in logical_reports:
        vid = lr_path.name.replace("_logical_check.json", "")
        logical = load_json(lr_path)
        coherence_path = cr_dir / f"{vid}_coherence_check.json"
        coherence = load_json(coherence_path) if coherence_path.exists() else None

        tl_path = tl_dir / f"{vid}.json"
        cb_path = cb_dir / f"{vid}.json"
        bp_path = bp_dir / f"{vid}.json"

        thread_logic = load_json(tl_path) if tl_path.exists() else None
        coherent_blocks = load_json(cb_path) if cb_path.exists() else None

        title = ""
        url = ""
        if thread_logic:
            title = (thread_logic.get("title") or "").strip()
            url = (thread_logic.get("url") or "").strip()
        elif coherent_blocks:
            title = (coherent_blocks.get("title") or "").strip()
            url = (coherent_blocks.get("url") or "").strip()

        logical_issues = logical.get("issues", []) or []
        coherence_issues = (coherence.get("issues", []) or []) if coherence else []

        # checks
        logical_pass = (logical.get("num_issues") == 0)
        # For training log completeness: if coherence report is missing, treat as "unknown"
        coherence_status = "unknown"
        coherence_pass = False
        if coherence is not None and coherence.get("num_issues") is not None:
            coherence_pass = (coherence.get("num_issues") == 0)
            coherence_status = "pass" if coherence_pass else "fail"

        blueprint_exists = bp_path.exists()

        issues_detected = (len(logical_issues) + len(coherence_issues)) > 0

        # decision + final status
        if not issues_detected:
            human_decision = "auto_pass"
            final_status = "successful_entry"
            used_for_training = True
        else:
            # slot for human (filled later)
            human_decision = "pending"   # accept | redo | give_up (filled by human)
            final_status = "pending_entry"
            used_for_training = False

        # Task-level pass/fail (only mark tasks affected by issues)
        task_texts = build_task_texts(thread_logic)
        affected_tasks = set()
        affected_tasks.update(affected_tasks_from_logical_issues(thread_logic, logical_issues))
        affected_tasks.update(affected_tasks_from_coherence_issues(coherent_blocks, coherence_issues))

        tasks_out = []
        for ti in sorted(task_texts.keys()):
            tasks_out.append({
                "task_index": ti,
                "task_text": task_texts.get(ti, ""),
                "status": "fail" if ti in affected_tasks else "pass"
            })

        mission_entry: Dict[str, Any] = {
            "mission_id": vid,
            "mission_title": title,
            "url": url,
            "artifacts_present": {
                "thread_logic": tl_path.exists(),
                "coherent_blocks": cb_path.exists(),
                "task_blueprint": blueprint_exists
            },
            "checks": {
                "logical_check": {
                    "status": "pass" if logical_pass else "fail",
                    **summarize_issues(logical_issues),
                },
                "coherence_check": {
                    "status": coherence_status,
                    **summarize_issues(coherence_issues),
                }
            },
            "issues_detected": issues_detected,
            "human_decision": human_decision,     # auto_pass | pending | accept | redo | give_up
            "final_status": final_status,         # successful_entry | redo_entry | give_up_entry | pending_entry
            "used_for_training": used_for_training,
            "tasks": tasks_out,
        }

        # Write per-mission entry
        (missions_dir / f"{vid}.json").write_text(json.dumps(mission_entry, indent=2), encoding="utf-8")

        global_log["missions"].append(mission_entry)

    global_log["num_missions"] = len(global_log["missions"])
    (out_dir / "training_quality_log.json").write_text(json.dumps(global_log, indent=2), encoding="utf-8")
    print(f"[OK] Wrote: {out_dir / 'training_quality_log.json'}")
    print(f"[OK] Wrote per-mission logs to: {missions_dir}")


if __name__ == "__main__":
    main()
