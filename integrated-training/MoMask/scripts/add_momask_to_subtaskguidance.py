#!/usr/bin/env python3
"""
add_momask_to_subtaskguidance.py

Reads subtaskguidance JSON files and injects a "motion" field into EACH subtask.
This script DOES NOT run MoMask. It only creates placeholders + prompts.

Input JSONs (default):  results/subtaskguidance/
Output JSONs (default): results/subtaskguidance_with_motion/

Env overrides:
  INPUT_DIR, OUTPUT_DIR, MOMASK_OUT_ROOT, FPS, SEED
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "results/subtaskguidance"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "results/subtaskguidance_with_motion"))
MOMASK_OUT_ROOT = Path(os.environ.get("MOMASK_OUT_ROOT", "results/momask"))

FPS = int(os.environ.get("FPS", "30"))
SEED = int(os.environ.get("SEED", "0"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# REGEX HELPERS
# -----------------------------
RE_STEP_LINE = re.compile(
    r"^\s*(\d+)\.\s*\[type=([a-zA-Z_]+)\s*,\s*frames=\[([0-9,\s]+|all)\]\]\s*(.+?)\s*$"
)

RE_CLEANUP = re.compile(r"\b(clean|cleanup|reset|stow|return|park|power down|close out|sanitize|put back|standby)\b", re.IGNORECASE)

# “not really a motion” actions to filter out for MoMask prompt
RE_NON_MOTION = re.compile(
    r"\b(use a camera|capture a close-up|take a photo|record|display|announce|report|summarize|read|inspect|observe|identify|analyze|verify|confirm|check|measure|cross-check)\b",
    re.IGNORECASE
)

# Detect obviously broken/truncated actions
def looks_bad_action(a: str) -> bool:
    s = (a or "").strip()
    if not s:
        return True
    # too short -> useless
    if len(s) < 12:
        return True
    low = s.lower()
    # classic truncation patterns
    if low.endswith(("adjust the", "adjust the.", "use the", "use the.", "move the", "move the.",
                     "place the", "place the.", "remove the", "remove the.", "in", "to", "for", "of", "on", "at", "with", "and", "or")):
        return True
    # "Adjust the." etc
    if re.fullmatch(r"(adjust|use|move|place|remove)\s+the\.?", low):
        return True
    # ends with just "the." or has dangling punctuation
    if low.endswith(("the.", "the;")):
        return True
    return False

def extract_actions_from_guidance(guidance_text: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (type, action_text) extracted from ORDERED_ROBOT_ACTION_STEPS.
    """
    actions: List[Tuple[str, str]] = []
    in_steps = False

    for ln in guidance_text or []:
        line = (ln or "").rstrip("\n")
        if line.strip() == "ORDERED_ROBOT_ACTION_STEPS:":
            in_steps = True
            continue
        if in_steps and line.strip() in ("SUBTASK_STORY:", "GLOBAL_SUMMARY:", "FRAME_BASED_OBSERVATIONS:",
                                         "INTEGRATED_SCENE_UNDERSTANDING:", "PRECONDITIONS_FOR_ROBOT:", "SUCCESS_CRITERIA:"):
            # next section
            in_steps = False
            continue
        if not in_steps:
            continue

        m = RE_STEP_LINE.match(line)
        if not m:
            continue

        step_type = (m.group(2) or "").strip().lower()
        action = (m.group(4) or "").strip()

        # remove trailing period for prompt readability
        if action.endswith("."):
            action = action[:-1].strip()

        actions.append((step_type, action))

    return actions

def build_momask_prompt_from_actions(actions: List[Tuple[str, str]]) -> str:
    """
    Heuristic:
      - prefer navigation/manipulation actions
      - drop obviously non-motion + truncated actions
      - ensure not "cleanup only" -> fallback to idle gesture
    """
    # collect candidate motion-like actions
    selected: List[str] = []
    for t, a in actions:
        if t not in ("navigation", "manipulation"):
            continue
        if looks_bad_action(a):
            continue
        if RE_NON_MOTION.search(a):
            continue
        selected.append(a)

    # Always keep cleanup if present (but don't let it be the only thing)
    cleanup_actions = [a for (t, a) in actions if RE_CLEANUP.search(a)]
    cleanup_action = cleanup_actions[-1] if cleanup_actions else "Return/reset to a safe standby position and stow any tools (cleanup)"

    # If nothing usable, return idle/gesture fallback
    if not selected:
        return "Stand in a neutral humanoid pose, look around as if observing a scene, gesture briefly with one arm, then return to a stable idle stance."

    # If it collapsed to only cleanup, also fallback
    if len(selected) == 1 and RE_CLEANUP.search(selected[0]):
        return "Stand in a neutral humanoid pose, look around as if observing a scene, gesture briefly with one arm, then return to a stable idle stance."

    # Keep at most 4 actions to avoid overly long prompts
    selected = selected[:4]

    # Ensure cleanup is appended (if not already)
    if not any(RE_CLEANUP.search(x) for x in selected):
        selected.append(cleanup_action)

    # Build prompt string
    seq = "; ".join(selected)
    return f"Perform these actions smoothly in sequence: {seq}; then return to a stable neutral stance."

def compute_duration_s(subtask: Dict[str, Any]) -> float:
    """
    Prefer subtask end-start if valid; fallback 4s.
    """
    try:
        st = float(subtask.get("start", 0.0))
        en = float(subtask.get("end", 0.0))
        dur = max(3.0, en - st)
        return float(dur)
    except Exception:
        return 4.0

def add_motion_to_subtask(video_index: str, task_i: int, sub_i: int, subtask: Dict[str, Any]) -> None:
    # If already has motion, leave it alone
    if isinstance(subtask.get("motion"), dict):
        return

    guidance_text = subtask.get("guidance_text") or []
    actions = extract_actions_from_guidance(guidance_text)
    prompt = build_momask_prompt_from_actions(actions)

    dur = compute_duration_s(subtask)
    out_dir = MOMASK_OUT_ROOT / video_index / f"task{task_i:02d}" / f"sub{sub_i:02d}"
    bvh_path = str(out_dir / "motion.bvh")
    params_path = str(out_dir / "params.json")

    subtask["motion"] = {
        "engine": "momask",
        "prompt": prompt,
        "duration_s": dur,
        "fps": FPS,
        "seed": SEED,
        "bvh_path": bvh_path,
        "params_path": params_path,
        "status": "pending_momask_execution",
        "note": "MoMask execution not yet wired; placeholder created.",
    }

def process_file(in_path: Path, out_path: Path) -> Tuple[int, int, int]:
    """
    Returns (num_subtasks, added_motion, already_had_motion)
    """
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_index = data.get("video_index") or in_path.stem

    total_subtasks = 0
    added = 0
    already = 0

    tasks = data.get("tasks") or []
    for ti, task in enumerate(tasks):
        subtasks = task.get("subtasks") or []
        for si, sub in enumerate(subtasks):
            total_subtasks += 1
            if isinstance(sub.get("motion"), dict):
                already += 1
                continue
            add_motion_to_subtask(video_index, ti, si, sub)
            added += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return total_subtasks, added, already

def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"[ERROR] INPUT_DIR not found: {INPUT_DIR}")

    files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix == ".json"])
    if not files:
        raise SystemExit(f"[WARN] No JSON files found under {INPUT_DIR}")

    grand_total = 0
    grand_added = 0
    grand_already = 0

    print(f"[INFO] INPUT_DIR={INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")
    print(f"[INFO] MOMASK_OUT_ROOT={MOMASK_OUT_ROOT}")

    for p in files:
        out_p = OUTPUT_DIR / p.name
        try:
            total, added, already = process_file(p, out_p)
            grand_total += total
            grand_added += added
            grand_already += already
            print(f"[OK] {p.name}: subtasks={total} added_motion={added} already_motion={already} -> {out_p}")
        except Exception as e:
            print(f"[ERROR] Failed {p.name}: {e}")

    print(f"[DONE] files={len(files)} subtasks={grand_total} added_motion={grand_added} already_motion={grand_already}")

if __name__ == "__main__":
    main()
