import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Category set (fixed)
# -----------------------------
CATEGORIES = ["motion", "perception", "planning", "narration"]


# -----------------------------
# Heuristic lexicons (tune as needed)
# -----------------------------
MOTION_VERBS = {
    "walk", "step", "move", "approach", "turn", "rotate",
    "bend", "kneel", "crouch", "lean", "reach", "grab", "grasp",
    "pick", "lift", "carry", "hold", "place", "put", "set",
    "open", "close", "push", "pull", "press", "twist",
    "pour", "stir", "mix", "insert", "remove", "drop",
    "wipe", "clean", "rinse", "cut", "slice", "chop",
    "screw", "unscrew", "tighten", "loosen", "plug", "unplug",
    "dig", "sow", "plant", "cover", "fill", "pack", "pat",
    "separate"
}

PERCEPTION_VERBS = {
    "look", "see", "watch", "observe", "inspect", "check", "measure",
    "read", "scan", "identify", "verify", "confirm", "detect", "notice"
}

# Planning: cognitive verbs (more precise than "blending")
PLANNING_VERBS = {
    "plan", "decide", "choose", "determine", "specify", "select", "define",
    "evaluate", "compare", "analyze", "estimate", "calculate", "reason",
    "consider", "prioritize", "schedule"
}

# Planning connector phrases (must match as phrases, not substrings)
PLANNING_PHRASES = [
    "so that",
    "in order to",
    "to ensure",
    "to avoid",
    "as a result",
    "therefore",
    "thus",
    "hence",
    "due to",
    "which requires",
    "which helps",
    "which is why",
    "this leads to",
]

# Single-word connectors (must match whole words)
PLANNING_WORDS = {"therefore", "thus", "hence"}

# Strong narration/discourse leaders: "talking about", not doing
NARRATION_LEADERS = {
    "describe", "explain", "discuss", "highlight", "note", "mention",
    "outline", "introduce", "summarize", "report", "argue", "compare",
    "show", "point"
}

# Capability / descriptive patterns that should not become motion
MODAL_BLOCKERS = {"can", "cannot", "can't", "able", "unable", "could", "would", "should", "may", "might"}

# "process involves..." descriptive narration patterns
PROCESS_NARRATION_PHRASES = {
    "process involves", "involves", "consists of", "followed by",
    "based on", "is done by", "is performed by", "includes"
}

# "open canopy" adjective usage (avoid mislabeling as motion)
OPEN_ADJ_NOUNS = {"canopy", "conditions", "area", "space", "field", "environment", "terrain", "ground"}


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    t = normalize(text)
    return re.findall(r"[a-z]+", t)


def contains_any_phrase(tnorm: str, phrases: set) -> bool:
    return any(p in tnorm for p in phrases)


def starts_with_any(tokens: List[str], starters: set) -> bool:
    return bool(tokens) and tokens[0] in starters


def open_used_as_adjective(tokens: List[str]) -> bool:
    for i in range(len(tokens) - 1):
        if tokens[i] == "open" and tokens[i + 1] in OPEN_ADJ_NOUNS:
            return True
    return False


def has_modal_near_verb(tokens: List[str], verb: str, window: int = 2) -> bool:
    for i, t in enumerate(tokens):
        if t != verb:
            continue
        lo = max(0, i - window)
        hi = min(len(tokens), i + window + 1)
        if any(w in MODAL_BLOCKERS for w in tokens[lo:hi]):
            return True
    return False


def match_planning_markers(tnorm: str, tokens: List[str]) -> List[str]:
    """
    Return matched planning markers with correct boundaries.
    - phrases: substring match is fine
    - single words: must match as whole word, never substring (fixes 'so' in 'soil')
    """
    matched = []
    # phrase markers
    for p in PLANNING_PHRASES:
        if p in tnorm:
            matched.append(p)

    # single-word markers (whole word)
    token_set = set(tokens)
    for w in PLANNING_WORDS:
        if w in token_set:
            matched.append(w)

    return matched


def classify_subtask_4cat(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Return (category, meta) where category in:
      motion | perception | planning | narration
    """
    tnorm = normalize(text)
    tokens = tokenize(text)
    words = set(tokens)

    motion_found = sorted([v for v in MOTION_VERBS if v in words])
    perception_found = sorted([v for v in PERCEPTION_VERBS if v in words])
    planning_verbs_found = sorted([v for v in PLANNING_VERBS if v in words])
    planning_markers = match_planning_markers(tnorm, tokens)

    meta: Dict[str, Any] = {
        "matched_motion_verbs": motion_found,
        "matched_perception_verbs": perception_found,
        "matched_planning_verbs": planning_verbs_found,
        "matched_planning_markers": planning_markers,
        "starts_with_narration_leader": starts_with_any(tokens, NARRATION_LEADERS),
        "process_narration": contains_any_phrase(tnorm, PROCESS_NARRATION_PHRASES),
        "blocked_open_adjective": bool("open" in words and open_used_as_adjective(tokens)),
        "blocked_modal_capability": False,
    }

    # --- Hard blocks that should go to narration ---
    if meta["process_narration"]:
        return "narration", meta

    if meta["blocked_open_adjective"]:
        return "narration", meta

    # modal capability near motion => descriptive, not action
    if motion_found and any(has_modal_near_verb(tokens, v) for v in motion_found):
        meta["blocked_modal_capability"] = True
        if perception_found:
            return "perception", meta
        return "narration", meta

    # --- Planning: cognitive verbs or explicit planning markers ---
    # (planning is not "highlight/note/mention"; those stay narration unless they truly contain planning verbs/markers)
    if planning_verbs_found or planning_markers:
        # If it is just "note/mention/highlight" without planning content, keep narration
        # But we already checked planning content, so this is okay:
        return "planning", meta

    # --- Narration leader (explain/discuss/etc.) ---
    if meta["starts_with_narration_leader"]:
        if perception_found and not motion_found:
            return "perception", meta
        return "narration", meta

    # --- Mixed perception + motion: treat as planning (per prof: planning > blending) ---
    if motion_found and perception_found:
        return "planning", meta

    # --- Pure motion/perception ---
    if motion_found:
        return "motion", meta
    if perception_found:
        return "perception", meta

    # default
    return "narration", meta


def majority_category(counts: Dict[str, int]) -> str:
    # deterministic: sort by count desc then by category name
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_cat, top_cnt = items[0]
    if top_cnt == 0:
        return "narration"

    # if tie among top categories -> planning (mixed intent)
    tied = [c for c, v in items if v == top_cnt]
    if len(tied) > 1:
        return "planning"
    return top_cat


def group_consecutive_by_category(subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    if not subtasks:
        return groups

    def st_of(s): return s.get("start")
    def en_of(s): return s.get("end")

    cur_cat = subtasks[0].get("subtask_category", "narration")
    cur_refs = [{"task_index": subtasks[0].get("task_index"), "sub_index": subtasks[0].get("sub_index")}]
    cur_start = st_of(subtasks[0])
    cur_end = en_of(subtasks[0])

    gid = 0
    for s in subtasks[1:]:
        cat = s.get("subtask_category", "narration")
        if cat != cur_cat:
            groups.append({
                "group_id": gid,
                "category": cur_cat,
                "time_start": cur_start,
                "time_end": cur_end,
                "subtask_refs": cur_refs
            })
            gid += 1
            cur_cat = cat
            cur_refs = []
            cur_start = st_of(s)
            cur_end = en_of(s)

        cur_refs.append({"task_index": s.get("task_index"), "sub_index": s.get("sub_index")})

        if isinstance(en_of(s), (int, float)):
            if cur_end is None or float(en_of(s)) > float(cur_end):
                cur_end = en_of(s)
        if cur_start is None and isinstance(st_of(s), (int, float)):
            cur_start = st_of(s)

    groups.append({
        "group_id": gid,
        "category": cur_cat,
        "time_start": cur_start,
        "time_end": cur_end,
        "subtask_refs": cur_refs
    })
    return groups


def compute_thread_metrics(thread: Dict[str, Any]) -> Dict[str, Any]:
    subs = thread.get("subtasks", []) or []

    # timestamps monotonic check
    starts = [s.get("start") for s in subs if isinstance(s.get("start"), (int, float))]
    monotonic = True
    for i in range(1, len(starts)):
        if float(starts[i]) < float(starts[i - 1]):
            monotonic = False
            break

    # avg gap seconds between consecutive subtasks (start[i] - end[i-1])
    gaps = []
    prev_end = None
    for s in subs:
        st = s.get("start")
        en = s.get("end")
        if isinstance(st, (int, float)) and isinstance(prev_end, (int, float)):
            gaps.append(float(st) - float(prev_end))
        if isinstance(en, (int, float)):
            prev_end = en

    avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

    logic = thread.get("logic", {}) or {}
    conflicts = logic.get("conflicts", []) or []
    repair = logic.get("repair", {}) or {}
    needs_repair = bool((repair.get("bridging_sentences") or []) or (repair.get("revised_subtasks") or []))

    return {
        "timestamps_monotonic": monotonic,
        "avg_gap_seconds": round(avg_gap, 3),
        "has_conflicts": bool(conflicts),
        "needs_repair": needs_repair,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/thread_logic")
    ap.add_argument("--output_dir", required=True, help="results/categorized_threads")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No json files in {in_dir}")
        return

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))

        # Aggregate task-level category evidence from all subtasks
        task_texts: Dict[int, str] = {}
        task_counts: Dict[int, Dict[str, int]] = {}

        out_threads = []
        for th in data.get("threads_with_logic", []) or []:
            subs = th.get("subtasks", []) or []

            categorized_subs = []
            thread_counts = {c: 0 for c in CATEGORIES}
            for s in subs:
                cat, meta = classify_subtask_4cat(s.get("text", ""))
                out_s = dict(s)
                out_s["subtask_category"] = cat
                out_s["category_confidence"] = None  # heuristic
                out_s["category_meta"] = meta
                categorized_subs.append(out_s)
                thread_counts[cat] += 1

                ti = int(s.get("task_index", -1))
                if ti >= 0:
                    task_texts.setdefault(ti, s.get("task_text", ""))
                    task_counts.setdefault(ti, {c: 0 for c in CATEGORIES})
                    task_counts[ti][cat] += 1

            # ensure order by timestamp (defensive)
            categorized_subs.sort(key=lambda x: (x.get("start") if isinstance(x.get("start"), (int, float)) else 1e18))

            thread_dom = majority_category(thread_counts)
            category_groups = group_consecutive_by_category(categorized_subs)

            logic = th.get("logic", {}) or {}
            # normalize single-subtask thread links to []
            if len(categorized_subs) <= 1:
                logic["logical_links"] = []

            out_thread = {
                "thread_index": th.get("thread_index"),
                "time_start": th.get("time_start"),
                "time_end": th.get("time_end"),
                "thread_category_distribution": thread_counts,
                "thread_dominant_category": thread_dom,
                "subtasks": categorized_subs,
                "category_groups": category_groups,
                "logic": logic,
            }
            out_thread["metrics"] = compute_thread_metrics(out_thread)
            out_threads.append(out_thread)

        task_categories = []
        for ti in sorted(task_texts.keys()):
            counts = task_counts.get(ti, {c: 0 for c in CATEGORIES})
            dom = majority_category(counts)
            task_categories.append({
                "task_index": ti,
                "task_text": task_texts.get(ti, ""),
                "task_category": dom,
                "evidence_counts": counts
            })

        out = {
            "index": data.get("index", p.stem),
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "category_schema": CATEGORIES,
            "task_categories": task_categories,
            "num_threads": len(out_threads),
            "threads": out_threads,
        }

        (out_dir / f"{out['index']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[OK] {p.name} -> {out_dir / (out['index'] + '.json')}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
