#!/usr/bin/env python3
"""
momask_prompt_from_guidance.py (caption-grounded v5, dataset-safe)

Fixes:
- Always starts with a walking/orient intro (never "look closely..." first).
- Reduces repetition even when many subtasks share similar captions.
- Uses deterministic variant rotation keyed by line_id.
- Keeps prompts generic and physically plausible (no “calendar/market/year/price” semantics).
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple, Dict, Optional

STEP_RE = re.compile(
    r"^\s*\d+\.\s*\[type=([a-z_]+),\s*frames=\[[^\]]*\]\]\s*(.+?)\s*$",
    re.IGNORECASE
)

KEEP_TYPES = {"navigation", "manipulation", "perception"}  # ignore communication

RE_FRAME_TAG = re.compile(r"\[frames=\[[^\]]*\]\]", re.IGNORECASE)
RE_VIDEO_ARTIFACTS = re.compile(
    r"\b(frames?|caption(s)?|text overlay|on-screen|on screen|screen|video|timeline)\b",
    re.IGNORECASE,
)
RE_CLEANUP = re.compile(
    r"\b(clean|cleanup|reset|stow|standby|safe standby|power down|close out|return/reset)\b",
    re.IGNORECASE,
)
RE_COGNITIVE = re.compile(
    r"\b(extract|identify|analyze|interpret|understand|infer|predict|project(ed)?|estimate|"
    r"market|billion|year|growth|trend|data|result|value|size|dominance|importance|price|pricing|"
    r"certified|sustainable|security|demand)\b",
    re.IGNORECASE
)
RE_ABSTRACT = re.compile(
    r"\b(database|cross-check|stored data|internal data|log|record|report|announce|summarize|communicate)\b",
    re.IGNORECASE
)

RE_LOOK_THE = re.compile(r"\blook\s+the\s+", re.IGNORECASE)
RE_WS = re.compile(r"\s+")
RE_BAD_END = re.compile(r"\b(the|a|an|to|for|of|in|on|at|with|and|or|by|as)\s*$", re.IGNORECASE)

RE_NAV_VERB = re.compile(r"\b(navigate|move|go to|approach|return|walk|reach|position|transition)\b", re.IGNORECASE)
RE_MANIP_VERB = re.compile(r"\b(pick|place|grasp|collect|pour|turn|press|open|close|insert|remove|adjust|hold|extend|point|cut)\b", re.IGNORECASE)

# Caption cues
RE_KNEEL = re.compile(r"\b(kneel|kneeling)\b", re.IGNORECASE)
RE_HOLD = re.compile(r"\b(hold|holding|carry|carrying)\b", re.IGNORECASE)
RE_CUT  = re.compile(r"\b(cut|cutting|slice|slicing|knife)\b", re.IGNORECASE)
RE_POINT = re.compile(r"\b(point|pointing|gesture)\b", re.IGNORECASE)
RE_TAG_PRICE = re.compile(r"\b(price tag|tag)\b", re.IGNORECASE)

# Global structure
OUTRO_WALK_BACK = "walk back."
OUTRO_NEUTRAL = "Return to a neutral standing pose."

# Deterministic variants (rotate via line_id)
INTRO_VARIANTS = [
    "walk a few steps forward and orient toward the scene.",
    "walk forward a few steps and face the scene.",
    "take a few steps forward and orient to the scene.",
    "walk forward slightly and align your body with the scene.",
]

SCAN_VARIANTS = [
    "turn head left and right to scan the scene.",
    "scan the scene by slowly turning the head side to side.",
    "tilt head slightly and scan left to right.",
    "turn upper body slightly and scan the scene.",
]

LOOK_VARIANTS = [
    "look closely at the scene.",
    "lean forward slightly and look closely.",
    "tilt head forward and look closely.",
    "pause and look closely again.",
]

PRESENT_VARIANTS = [
    "raise both hands slightly as if holding or presenting an object.",
    "raise both forearms slightly as if presenting an object.",
    "bring both hands up briefly as if showing an object.",
]

POINT_VARIANTS = [
    "raise one arm and point forward briefly.",
    "lift one hand and point forward briefly.",
    "gesture forward briefly with one arm.",
]

CUTLIKE_VARIANTS = [
    "reach forward and perform a brief cutting-like hand motion over an object area.",
    "perform a brief slicing-like hand motion over an object area.",
]

KNEEL_SEQ_VARIANTS = [
    ["bend knees and lower into a brief kneeling posture.", "rise back to standing."],
    ["lower into a short kneeling posture.", "stand back up."],
]

MIN_CLAUSES = 5
SIMILARITY_THRESH = 0.83  # more aggressive than v4


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = RE_FRAME_TAG.sub("", s)
    s = RE_VIDEO_ARTIFACTS.sub("scene", s)
    s = RE_LOOK_THE.sub("look at the ", s)
    s = RE_WS.sub(" ", s).strip()
    return s


def _bad_line(s: str) -> bool:
    s = (s or "").strip()
    if not s or len(s) < 8:
        return True
    if s.endswith(("..", "…")):
        return True
    if s.endswith((",", ";", ":")):
        return True
    if RE_BAD_END.search(s):
        return True
    return False


def extract_steps(guidance_lines: List[str]) -> List[Tuple[str, str]]:
    steps: List[Tuple[str, str]] = []
    in_steps = False
    for ln in guidance_lines:
        t = ln.strip()
        if t == "ORDERED_ROBOT_ACTION_STEPS:":
            in_steps = True
            continue
        if in_steps and t.endswith(":") and t != "ORDERED_ROBOT_ACTION_STEPS:":
            break
        if in_steps:
            m = STEP_RE.match(t)
            if m:
                step_type = m.group(1).lower().strip()
                action = _clean(m.group(2))
                if action:
                    steps.append((step_type, action))
    return steps


def _dedup_consecutive(sentences: List[str]) -> List[str]:
    out: List[str] = []
    prev = None
    for s in sentences:
        s2 = RE_WS.sub(" ", (s or "").strip())
        if not s2:
            continue
        if prev is not None and s2.lower() == prev.lower():
            continue
        out.append(s2)
        prev = s2
    return out


def caption_signature(frames: List[Dict[str, object]]) -> str:
    caps = []
    for f in frames or []:
        c = (f.get("caption") or "")
        if isinstance(c, str) and c.strip():
            caps.append(c.strip())
    return " ".join(caps)


def _pick(lst: List[str], line_id: int, salt: int = 0) -> str:
    return lst[(line_id + salt) % len(lst)]


def caption_based_primitives(caps: str, line_id: int, force_rotate: int = 0) -> List[str]:
    """
    Deterministic, diversified primitives based on caption cues + line_id.
    force_rotate nudges which variants we pick when repetition is detected.
    """
    s = caps or ""
    intro = _pick(INTRO_VARIANTS, line_id, salt=force_rotate)
    scan = _pick(SCAN_VARIANTS, line_id, salt=1 + force_rotate)
    look = _pick(LOOK_VARIANTS, line_id, salt=2 + force_rotate)
    present = _pick(PRESENT_VARIANTS, line_id, salt=3 + force_rotate)
    point = _pick(POINT_VARIANTS, line_id, salt=4 + force_rotate)
    cutlike = _pick(CUTLIKE_VARIANTS, line_id, salt=5 + force_rotate)
    kneel_pair = KNEEL_SEQ_VARIANTS[(line_id + force_rotate) % len(KNEEL_SEQ_VARIANTS)]

    prim: List[str] = [intro, scan]

    if RE_KNEEL.search(s):
        prim += [kneel_pair[0], look, kneel_pair[1]]
    else:
        prim += [look]

    if RE_HOLD.search(s):
        prim += [present, look]

    if RE_CUT.search(s):
        prim += [cutlike, "pause briefly and look again."]

    if RE_TAG_PRICE.search(s) or RE_POINT.search(s):
        prim += [point, look]

    prim += [OUTRO_WALK_BACK]
    return prim


def rewrite_step_to_motion(action: str, step_type: str, line_id: int) -> Optional[str]:
    a = (action or "").strip()
    if not a:
        return None

    a = re.sub(r"^\s*(the\s+robot|robot)\s+", "", a, flags=re.IGNORECASE)
    a = _clean(a)

    if RE_CLEANUP.search(a):
        return None
    if RE_COGNITIVE.search(a) or RE_ABSTRACT.search(a):
        return None

    if step_type == "navigation":
        if not RE_NAV_VERB.search(a):
            a = "walk a few steps and orient toward the scene"
        else:
            a = "walk a few steps and orient toward the scene"

    elif step_type == "manipulation":
        if not RE_MANIP_VERB.search(a):
            return None
        low = a.lower()
        if any(k in low for k in ["pick", "grasp", "retrieve", "collect"]):
            a = "reach down and pick up an object using a hand"
        elif any(k in low for k in ["place", "put"]):
            a = "place the object down in front"
        elif "point" in low:
            a = _pick(POINT_VARIANTS, line_id, salt=0).rstrip(".")
        elif "adjust" in low:
            a = "adjust posture slightly"
        elif any(k in low for k in ["cut", "slice"]):
            a = _pick(CUTLIKE_VARIANTS, line_id, salt=0).rstrip(".")
        else:
            return None

    elif step_type == "perception":
        low = a.lower()
        if "scan" in low:
            a = _pick(SCAN_VARIANTS, line_id, salt=0).rstrip(".")
        else:
            a = _pick(LOOK_VARIANTS, line_id, salt=0).rstrip(".")
    else:
        return None

    a = a.strip()
    if not a.endswith("."):
        a += "."
    a = RE_WS.sub(" ", a).strip()
    if _bad_line(a):
        return None
    return a


def ensure_structure(motions: List[str], frames: List[Dict[str, object]], line_id: int, rotate: int = 0) -> List[str]:
    """
    Hard enforce:
    - Intro must be first (always).
    - Minimum clauses.
    - Must include walk back.
    """
    motions = _dedup_consecutive(motions)

    # FORCE intro first (remove any intro-like later to avoid duplicates)
    intro = _pick(INTRO_VARIANTS, line_id, salt=rotate)
    motions = [m for m in motions if m.lower() not in {x.lower() for x in INTRO_VARIANTS}]
    motions.insert(0, intro)
    motions = _dedup_consecutive(motions)

    if len(motions) < MIN_CLAUSES:
        caps = caption_signature(frames)
        if caps.strip():
            motions = caption_based_primitives(caps, line_id=line_id, force_rotate=rotate)
        else:
            # deterministic fallback without captions
            motions = [
                intro,
                _pick(SCAN_VARIANTS, line_id, salt=1 + rotate),
                "pause briefly.",
                _pick(LOOK_VARIANTS, line_id, salt=2 + rotate),
                OUTRO_WALK_BACK,
            ]

    motions = _dedup_consecutive(motions)

    if not any("walk back" in s.lower() for s in motions):
        motions.append(OUTRO_WALK_BACK)

    return _dedup_consecutive(motions)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def steps_to_momask_prompt(
    steps: List[Tuple[str, str]],
    frames: List[Dict[str, object]],
    line_id: int,
    prev_prompts: List[str],
) -> str:
    motions: List[str] = []
    saw_nav_or_manip = False

    for t, a in steps:
        t = (t or "").lower().strip()
        if t not in KEEP_TYPES:
            continue
        m = rewrite_step_to_motion(a, t, line_id=line_id)
        if m:
            motions.append(m)
            if t in ("navigation", "manipulation"):
                saw_nav_or_manip = True

    motions = _dedup_consecutive(motions)

    # If weak, fallback to caption primitives
    if len(motions) < 3 or not saw_nav_or_manip:
        caps = caption_signature(frames)
        motions = caption_based_primitives(caps, line_id=line_id) if caps.strip() else []

    # Enforce structure
    motions = ensure_structure(motions, frames=frames, line_id=line_id, rotate=0)

    # Anti-repetition: if too similar, rotate variants (try up to 2 rotations)
    candidate = " ".join(motions).strip()
    if prev_prompts:
        best_sim = max(similarity(candidate, p) for p in prev_prompts)
        if best_sim >= SIMILARITY_THRESH:
            motions = ensure_structure(motions, frames=frames, line_id=line_id, rotate=1)
            candidate = " ".join(motions).strip()
            best_sim = max(similarity(candidate, p) for p in prev_prompts)
            if best_sim >= SIMILARITY_THRESH:
                motions = ensure_structure(motions, frames=frames, line_id=line_id, rotate=2)
                candidate = " ".join(motions).strip()

    if not candidate.endswith("."):
        candidate += "."
    prompt = f"{candidate} {OUTRO_NEUTRAL}".strip()
    prompt = RE_WS.sub(" ", prompt).strip()
    prompt = RE_LOOK_THE.sub("look at the ", prompt)
    return prompt


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--motion_len", default="NA")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    vid = data.get("video_index") or Path(args.input_json).stem

    prompts_path = out_dir / f"{vid}_momask_prompts.txt"
    index_path = out_dir / f"{vid}_momask_index.json"

    prompts: List[str] = []
    index: List[Dict[str, object]] = []
    prev_prompts_for_similarity: List[str] = []

    for ti, task in enumerate(data.get("tasks", [])):
        for si, sub in enumerate(task.get("subtasks", [])):
            gl = sub.get("guidance_text")
            if not gl or not isinstance(gl, list):
                continue

            steps = extract_steps(gl)
            frames = sub.get("frames") or []
            text_prompt = steps_to_momask_prompt(
                steps, frames, line_id=len(prompts), prev_prompts=prev_prompts_for_similarity
            )

            prompts.append(f"{text_prompt}#{args.motion_len}")
            prev_prompts_for_similarity.append(text_prompt)

            index.append({
                "line_id": len(prompts) - 1,
                "task_i": ti,
                "sub_i": si,
                "subtask_text": sub.get("text", ""),
            })

    prompts_path.write_text("\n".join(prompts) + ("\n" if prompts else ""), encoding="utf-8")
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    print(f"[OK] wrote {prompts_path}")
    print(f"[OK] wrote {index_path}")
    print(f"[OK] total prompts: {len(prompts)}")


if __name__ == "__main__":
    main()
