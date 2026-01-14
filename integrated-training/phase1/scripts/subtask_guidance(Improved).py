#!/usr/bin/env python3
"""
subtask_guidance_generate.py

LLM (writer) -> Python (judge/repair) -> (optional) LLM retry

This version enforces:
- no truncation anywhere (all sections scanned)
- imperative steps (no "The robot ...")
- stronger semantic correctness of step type vs action verb
- no generic filler like "Highlight the key visible element..."
- SUBTASK_STORY rewritten as 1–2 clean paragraphs (non-repetitive)
- always ends with cleanup step
- perception-before-manipulation
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DEFAULT_FRAME_CAPTION_DIR = "/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/frame_captions"
FRAME_CAPTION_DIR = Path(os.environ.get("FRAME_CAPTION_DIR", DEFAULT_FRAME_CAPTION_DIR))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "results/subtaskguidance"))

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

MAX_FRAMES_PER_SUBTASK = int(os.environ.get("MAX_FRAMES_PER_SUBTASK", "8"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "384"))

MIN_STEPS = int(os.environ.get("MIN_STEPS", "5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
REQUIRE_VERIFICATION = os.environ.get("REQUIRE_VERIFICATION", "1") == "1"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "1"))

# Optional strict equipment gating:
EQUIPMENT_ALLOWLIST = [s.strip().lower() for s in os.environ.get("EQUIPMENT_ALLOWLIST", "").split(",") if s.strip()]

ALLOWED_TYPES = {"navigation", "manipulation", "perception", "communication"}
TYPE_FIX_MAP = {
    "sensing": "perception",
    "sensor": "perception",
    "processing": "perception",
    "analysis": "perception",
    "planning": "perception",
    "reasoning": "perception",
    "actuation": "manipulation",
    "movement": "navigation",
    "recording": "perception",
    "verification": "perception",
}

RE_VERIFICATION = re.compile(r"\b(verify|confirm|check|read|measure|validate|cross-check)\b", re.IGNORECASE)
RE_CLEANUP = re.compile(r"\b(clean|cleanup|reset|stow|return|park|power down|close out|sanitize|put back|standby)\b", re.IGNORECASE)

RE_HUMAN_DELEGATION = re.compile(
    r"\b(ask\s+the\s+user|ask\s+someone|tell\s+the\s+user|tell\s+someone|instruct\s+the\s+user|have\s+someone|let\s+the\s+user)\b",
    re.IGNORECASE,
)

BANNED_PATTERNS = [
    r"\bweapon(s)?\b",
    r"\bguns?\b",
    r"\bammunition\b",
    r"\bexplosive(s)?\b",
    r"\bbomb\b",
    r"\bgrenade\b",
    r"\bpoison\b",
    r"\btoxin(s)?\b",
    r"\bhack\b",
    r"\bmalware\b",
    r"\bspyware\b",
    r"\bsteal\b",
]
RE_BANNED = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)

# Action-word classifiers
RE_NAV_WORD = re.compile(r"\b(navigate|move|go to|approach|return|walk|drive|reach|position|transition)\b", re.IGNORECASE)
RE_MANIP_WORD = re.compile(r"\b(pick|place|grasp|collect|pour|turn|press|open|close|insert|remove|adjust|hold|stow|extend)\b", re.IGNORECASE)
RE_PERC_WORD = re.compile(r"\b(observe|look|read|detect|identify|analyze|inspect|scan|measure|verify|check|confirm)\b", re.IGNORECASE)
RE_COMM_WORD = re.compile(r"\b(report|say|announce|explain|summarize|communicate|display|log|record|present)\b", re.IGNORECASE)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

def load_model_and_tokenizer():
    device_msg = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading model {MODEL_NAME} on {device_msg}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    print("[INFO] Model loaded.")
    return tokenizer, model

# ---------------------------------------------------------------------
# PROMPT
# ---------------------------------------------------------------------

def summarize_frames_for_prompt(frames: List[Dict[str, Any]]) -> str:
    if not frames:
        return "No frames available for this subtask."

    frames_sorted = sorted(frames, key=lambda f: int(f.get("frame_index", 0)))
    frames_trimmed = frames_sorted[:MAX_FRAMES_PER_SUBTASK]

    lines: List[str] = []
    for f in frames_trimmed:
        idx = int(f.get("frame_index", 0))
        cap = (f.get("caption", "") or "").strip()
        rel = f.get("relative_path", "")
        lines.append(f"- frame_index={idx}, file={rel}: {cap}")
    return "\n".join(lines)

def build_subtask_prompt(
    video_index: str,
    title: str,
    task_obj: Dict[str, Any],
    subtask_obj: Dict[str, Any],
    strict_fix: bool = False,
) -> str:
    task_text = (task_obj.get("task", "") or "").strip()
    sub_text = (subtask_obj.get("text", "") or "").strip()
    start = subtask_obj.get("start", None)
    end = subtask_obj.get("end", None)
    frames = subtask_obj.get("frames") or []
    frame_summary = summarize_frames_for_prompt(frames)

    if start is not None and end is not None:
        try:
            time_range = f"{float(start):.1f}s–{float(end):.1f}s"
        except Exception:
            time_range = "unknown time range"
    else:
        time_range = "unknown time range"

    equipment_rule = ""
    if EQUIPMENT_ALLOWLIST:
        equipment_rule = (
            "\n- EQUIPMENT RULE: Only mention/assume equipment from this allowlist: "
            + ", ".join(EQUIPMENT_ALLOWLIST)
            + ". If unsure, keep actions informational (read/verify/report) rather than physical manipulation."
        )

    extra_fix = ""
    if strict_fix:
        extra_fix = f"""
ADDITIONAL STRICT REQUIREMENTS (YOU MUST FOLLOW)
-----------------------------------------------
- ORDERED_ROBOT_ACTION_STEPS must have between {MIN_STEPS} and {MAX_STEPS} steps.
- Each step must use: [type=<navigation|manipulation|perception|communication>, frames=[...]]
- Steps MUST be imperative (start with a verb). Do NOT start steps with "The robot ...".
- Frames MUST be chosen only from the listed frame_index values. If unsure, use frames=[all].
- Include at least one explicit verification word (verify/check/confirm/read/measure/cross-check).
- Ensure at least one perception step happens BEFORE the first manipulation step.
- End with a cleanup-style step (return/reset/stow/cleanup/standby).
- Avoid delegating actions to a human.
- Do not truncate: finish sentences; do not cut words.
{equipment_rule}
"""

    prompt = f"""You are helping design a *high-level, human-readable* description of what a humanoid robot
should understand and do for a specific **subtask** in a tutorial video.

VIDEO METADATA
--------------
- video_index: {video_index}
- title: {title}
- high_level_task: {task_text}

SUBTASK INFO
------------
- subtask_text: {sub_text}
- subtask_time_range: {time_range}

FRAMES FOR THIS SUBTASK
-----------------------
Each frame has a frame_index and a caption. Use these as your only visual clues.
Do NOT invent unrelated objects/scenes; stay consistent with captions.

{frame_summary}

YOUR JOB
--------
Produce a **SECTIONED TEXT DESCRIPTION** for this subtask.

IMPORTANT FORMAT RULES
----------------------
- DO NOT output JSON.
- DO NOT use curly braces `{{` or `}}`.
- Use exactly these headings in this order:
  1. GLOBAL_SUMMARY:
  2. FRAME_BASED_OBSERVATIONS:
  3. INTEGRATED_SCENE_UNDERSTANDING:
  4. PRECONDITIONS_FOR_ROBOT:
  5. SUCCESS_CRITERIA:
  6. ORDERED_ROBOT_ACTION_STEPS:
  7. SUBTASK_STORY:
- Blank line after each heading.
- FRAME_BASED_OBSERVATIONS bullets:
  - [frames=[0,1]] ...
- ORDERED_ROBOT_ACTION_STEPS numbered:
  1. [type=navigation, frames=[0,1]] ...
- You may use frames=[all] ONLY inside tags.

CONTENT GUIDELINES
------------------
- GLOBAL_SUMMARY: 1–3 sentences.
- FRAME_BASED_OBSERVATIONS: 3–8 bullets.
- INTEGRATED_SCENE_UNDERSTANDING: 1–2 short paragraphs.
- PRECONDITIONS_FOR_ROBOT: 3–6 bullets.
- SUCCESS_CRITERIA: 3–6 bullets.
- ORDERED_ROBOT_ACTION_STEPS: {MIN_STEPS}–{MAX_STEPS} steps.
- SUBTASK_STORY: 1–2 paragraphs, time-ordered narrative.
- Do not include unsafe/banned content.
- Do not delegate steps to a human; the robot performs the steps.
{equipment_rule}
{extra_fix}

Now write ONLY the sections above, in order, with no extra text.
"""
    return prompt

# ---------------------------------------------------------------------
# QC / REPAIR
# ---------------------------------------------------------------------

HEADINGS = [
    "GLOBAL_SUMMARY:",
    "FRAME_BASED_OBSERVATIONS:",
    "INTEGRATED_SCENE_UNDERSTANDING:",
    "PRECONDITIONS_FOR_ROBOT:",
    "SUCCESS_CRITERIA:",
    "ORDERED_ROBOT_ACTION_STEPS:",
    "SUBTASK_STORY:",
]

RE_STEP_LINE = re.compile(
    r"^\s*(\d+)\.\s*\[type=([a-zA-Z_]+)\s*,\s*frames=\[([0-9,\s]+|all)\]\]\s*(.+?)\s*$"
)

RE_FRAMES_TAG = re.compile(r"\[frames=\[([0-9,\s]+|all)\]\]")
RE_END_PUNCT = re.compile(r"[.!?]\s*$")

def _normalize_blank_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    empty = 0
    for ln in lines:
        ln = ln.replace("\r\n", "\n").rstrip("\n").rstrip()
        if ln.strip() == "":
            empty += 1
            if empty <= 1:
                out.append("")
        else:
            empty = 0
            out.append(ln)
    return out

def _closest_allowed_frame(x: int, allowed: List[int]) -> int:
    if not allowed:
        return x
    return min(allowed, key=lambda a: abs(a - x))

def _expand_frames_token(token: str, allowed: List[int]) -> List[int]:
    token = (token or "").strip()
    if token.lower() == "all":
        return list(allowed)
    ids: List[int] = []
    for part in token.split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    return ids

def looks_truncated(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if RE_END_PUNCT.search(s):
        return False
    if len(s) < 45:
        return True
    if s.lower().endswith((" in", " to", " for", " of", " on", " at", " with", " and", " or", " by", " as")):
        return True
    # unmatched quotes/parens
    if s.count('"') % 2 == 1:
        return True
    if s.count("(") != s.count(")"):
        return True
    return False

def normalize_imperative(action: str) -> str:
    a = (action or "").strip()

    # Remove leading "The robot ..." narration
    a = re.sub(r"^\s*the\s+robot\s+", "", a, flags=re.IGNORECASE)
    a = re.sub(r"^\s*robot\s+", "", a, flags=re.IGNORECASE)

    # Convert "perceives/observes/reads" style into imperative verb
    a = re.sub(r"^\s*perceives\s+", "Observe ", a, flags=re.IGNORECASE)
    a = re.sub(r"^\s*reads\s+", "Read ", a, flags=re.IGNORECASE)
    a = re.sub(r"^\s*confirms\s+", "Confirm ", a, flags=re.IGNORECASE)
    a = re.sub(r"^\s*captures\s+", "Capture ", a, flags=re.IGNORECASE)
    a = re.sub(r"^\s*analyzes\s+", "Analyze ", a, flags=re.IGNORECASE)

    # Ensure it starts with a capital letter
    if a and a[0].islower():
        a = a[0].upper() + a[1:]

    # Ensure sentence ends nicely (for step lines)
    if a and not RE_END_PUNCT.search(a):
        a = a.rstrip() + "."
    return a

def _infer_type_from_action(action: str) -> Optional[str]:
    a = action.lower()
    if RE_COMM_WORD.search(a):
        return "communication"
    if RE_PERC_WORD.search(a):
        return "perception"
    if RE_MANIP_WORD.search(a):
        return "manipulation"
    if RE_NAV_WORD.search(a):
        return "navigation"
    return None

def _force_type_by_leading_verb(action: str, current_type: str) -> str:
    a = action.strip().lower()

    # HARD override by leading verb/phrase
    if a.startswith(("navigate", "move", "return", "go ", "approach", "position", "transition")):
        return "navigation"
    if a.startswith(("read", "observe", "inspect", "verify", "check", "analyze", "identify", "scan", "measure", "confirm", "capture")):
        return "perception"
    if a.startswith(("report", "announce", "explain", "summarize", "communicate", "display", "log", "record", "present", "state")):
        return "communication"
    if a.startswith(("pick", "place", "grasp", "collect", "open", "close", "insert", "remove", "adjust", "hold", "stow", "extend")):
        return "manipulation"

    inferred = _infer_type_from_action(action)
    if inferred and inferred in ALLOWED_TYPES:
        return inferred
    return current_type

def _equipment_violation_to_safe_action(action: str) -> Optional[str]:
    if not EQUIPMENT_ALLOWLIST:
        return None
    a = action.lower()
    physical_cues = ["sensor", "deploy", "collect a sample", "collect soil", "test tube", "petri dish", "instrument", "device", "sampling arm"]
    if any(cue in a for cue in physical_cues):
        ok = any(eq in a for eq in EQUIPMENT_ALLOWLIST)
        if not ok:
            return "Read/verify the on-screen information and report the relevant result without using unlisted equipment."
    return None

def _split_into_sections(text: str) -> Dict[str, List[str]]:
    lines = _normalize_blank_lines(text.splitlines())
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for ln in lines:
        if ln.strip() in HEADINGS:
            current = ln.strip()
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        sections[current].append(ln)
    return sections

def _autofill_missing_sections(sections: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    issues: List[str] = []
    for h in HEADINGS:
        if h not in sections:
            issues.append(f"missing_heading_autofill:{h}")
            if h == "GLOBAL_SUMMARY:":
                sections[h] = ["(auto-filled) Summary unavailable; proceed using frames and subtask text."]
            elif h == "FRAME_BASED_OBSERVATIONS:":
                sections[h] = ["- [frames=[all]] (auto-filled) Use the captions above as observations."]
            elif h == "INTEGRATED_SCENE_UNDERSTANDING:":
                sections[h] = ["(auto-filled) Integrate observations conservatively and avoid inventing details."]
            elif h == "PRECONDITIONS_FOR_ROBOT:":
                sections[h] = ["- (auto-filled) Robot is powered on; cameras/sensors are active."]
            elif h == "SUCCESS_CRITERIA:":
                sections[h] = ["- (auto-filled) Robot completes the subtask and reports outcomes."]
            elif h == "ORDERED_ROBOT_ACTION_STEPS:":
                sections[h] = []
            elif h == "SUBTASK_STORY:":
                sections[h] = []
    return sections, issues

def _repair_frames_tags_anywhere(
    lines: List[str],
    allowed_frame_indices: List[int],
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    allowed = sorted(set(int(x) for x in allowed_frame_indices))
    allowed_set = set(allowed)

    def repl(m):
        token = m.group(1).strip()
        ids = _expand_frames_token(token, allowed)
        fixed: List[int] = []
        for fid in ids:
            if fid in allowed_set:
                fixed.append(fid)
            else:
                new_fid = _closest_allowed_frame(fid, allowed)
                issues.append(f"fixed_frames_tag:{fid}->{new_fid}")
                fixed.append(new_fid)
        seen = set()
        fixed = [x for x in fixed if not (x in seen or seen.add(x))]
        if not allowed:
            return "[frames=[all]]"
        if not fixed:
            fixed = allowed[:]
        return f"[frames=[{','.join(str(x) for x in fixed)}]]"

    out = [RE_FRAMES_TAG.sub(repl, ln) for ln in lines]
    out = _normalize_blank_lines(out)
    return out, issues

def _repair_truncation_anywhere(
    sections: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    issues: List[str] = []

    def fix_line(ln: str) -> str:
        if not looks_truncated(ln):
            return ln
        # If it's clearly a bullet, keep bullet but finish safely
        stripped = ln.strip()
        if stripped.startswith("-"):
            return ln.rstrip() + " (completed)."
        # Otherwise finish sentence
        return ln.rstrip() + " (completed)."

    for h, lines in sections.items():
        new_lines = []
        for ln in lines:
            # don't touch empty lines
            if not ln.strip():
                new_lines.append(ln)
                continue
            if looks_truncated(ln):
                issues.append(f"fixed_truncation:{h}")
                new_lines.append(fix_line(ln))
            else:
                new_lines.append(ln)
        sections[h] = new_lines

    return sections, issues

def _repair_steps_section(
    step_lines_raw: List[str],
    allowed_frame_indices: List[int],
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    allowed = sorted(set(int(x) for x in allowed_frame_indices))
    allowed_set = set(allowed)

    parsed_steps: List[Tuple[str, List[int], str]] = []
    saw_verification = False

    for ln in step_lines_raw:
        m = RE_STEP_LINE.match(ln)
        if not m:
            continue

        step_type_raw = m.group(2).strip().lower()
        frames_token = m.group(3).strip()
        action = m.group(4).strip()

        # remove banned content entirely
        if RE_BANNED.search(action):
            issues.append("banned_content_removed")
            continue

        # remove human delegation by rewriting
        if RE_HUMAN_DELEGATION.search(action):
            issues.append("human_delegation_rewritten")
            action = re.sub(r"\bask\s+the\s+user\s+to\b", "request confirmation via UI and", action, flags=re.IGNORECASE)
            action = re.sub(r"\btell\s+the\s+user\s+to\b", "inform via UI and", action, flags=re.IGNORECASE)

        action = normalize_imperative(action)

        # replace low-signal generic filler (your current placeholder)
        if "Highlight the key visible element" in action:
            issues.append("generic_filler_replaced")
            action = "Read/verify the key on-screen text or graph value and state it clearly."

        # fix truncation in action
        if looks_truncated(action):
            issues.append("truncated_step_action_repaired")
            action = "Read/verify the key on-screen text or graph value and state it clearly."

        # equipment gating (optional)
        safe = _equipment_violation_to_safe_action(action)
        if safe is not None:
            issues.append("equipment_action_softened")
            action = normalize_imperative(safe)

        # type mapping for invalid raw types
        step_type = step_type_raw
        if step_type not in ALLOWED_TYPES:
            fixed = TYPE_FIX_MAP.get(step_type, "perception")
            issues.append(f"fixed_step_type:{step_type_raw}->{fixed}")
            step_type = fixed

        # semantic type override
        forced = _force_type_by_leading_verb(action, step_type)
        if forced != step_type:
            issues.append(f"fixed_step_type_semantic:{step_type}->{forced}")
            step_type = forced

        # frames
        frame_ids = _expand_frames_token(frames_token, allowed)
        fixed_frame_ids: List[int] = []
        for fid in frame_ids:
            if fid in allowed_set:
                fixed_frame_ids.append(fid)
            else:
                new_fid = _closest_allowed_frame(fid, allowed)
                issues.append(f"fixed_step_frame:{fid}->{new_fid}")
                fixed_frame_ids.append(new_fid)

        seen = set()
        fixed_frame_ids = [x for x in fixed_frame_ids if not (x in seen or seen.add(x))]
        if not fixed_frame_ids and allowed:
            fixed_frame_ids = allowed[:]

        if RE_VERIFICATION.search(action):
            saw_verification = True

        parsed_steps.append((step_type, fixed_frame_ids, action))

    if not parsed_steps:
        issues.append("steps_missing_autofill")
        parsed_steps = [
            ("navigation", allowed, "Navigate to the relevant scene for this subtask."),
            ("perception", allowed, "Observe key items and read any on-screen text."),
            ("perception", allowed, "Verify/confirm the key detail using the frames."),
            ("communication", allowed, "Report the verified result clearly."),
            ("navigation", allowed, "Return to a safe standby position (cleanup)."),
        ]
        saw_verification = True

    # Smoothness: ensure at least one perception before first manipulation
    first_manip = next((i for i, (t, _, __) in enumerate(parsed_steps) if t == "manipulation"), None)
    if first_manip is not None:
        any_perc_before = any(t == "perception" for (t, _, __) in parsed_steps[:first_manip])
        if not any_perc_before:
            issues.append("injected_perception_before_manipulation")
            parsed_steps.insert(0, ("perception", allowed, "Inspect/verify the scene before any physical action."))

            saw_verification = True

    # Ensure verification exists
    if REQUIRE_VERIFICATION and not saw_verification:
        issues.append("injected_verification_step")
        parsed_steps.append(("perception", allowed, "Verify/confirm the key observation before reporting."))

    # Ensure cleanup at end
    if parsed_steps:
        last_action = parsed_steps[-1][2]
        if not RE_CLEANUP.search(last_action):
            issues.append("injected_cleanup_step")
            parsed_steps.append(("navigation", allowed, "Return/reset to a safe standby position and stow any tools (cleanup)."))

    # Pad/truncate
    if len(parsed_steps) < MIN_STEPS:
        issues.append(f"padded_steps:{len(parsed_steps)}->{MIN_STEPS}")
        while len(parsed_steps) < MIN_STEPS:
            parsed_steps.insert(-1, ("perception", allowed, "Check/confirm the key detail matches the captions and scene."))

    if len(parsed_steps) > MAX_STEPS:
        issues.append(f"truncated_steps:{len(parsed_steps)}->{MAX_STEPS}")
        parsed_steps = parsed_steps[:MAX_STEPS]

    # Renumber and format
    formatted: List[str] = []
    for i, (t, fr, act) in enumerate(parsed_steps, start=1):
        fr_txt = "all" if not allowed else ",".join(str(x) for x in fr) if fr else "all"
        formatted.append(f"{i}. [type={t}, frames=[{fr_txt}]] {act}")

    return formatted, issues

def _story_bad(story_lines: List[str]) -> bool:
    clean = [ln.strip() for ln in story_lines if ln.strip()]
    if not clean:
        return True
    if len(" ".join(clean)) < 120:
        return True
    if any(looks_truncated(ln) for ln in clean):
        return True
    # template detection
    if any(ln.lower().startswith(("the robot starts by:", "next, it follows", "finally,")) for ln in clean):
        return True
    return False

def _rewrite_story_from_steps(step_lines: List[str]) -> List[str]:
    # Convert steps into 1–2 paragraph story, avoid repeating step 1 verbatim.
    actions: List[str] = []
    for s in step_lines:
        parts = s.split("]]", 1)
        act = parts[1].strip() if len(parts) == 2 else s
        act = act.rstrip(".")
        actions.append(act)

    if not actions:
        return ["The robot follows the planned steps, verifies key details from the frames, then reports and resets safely."]

    # pick 3 distinct actions: early, verify-like, and cleanup/report
    early = actions[0]
    verify = next((a for a in actions if RE_VERIFICATION.search(a)), None)
    report = next((a for a in actions if a.lower().startswith(("report", "announce", "summarize", "display", "state"))), None)
    cleanup = next((a for a in reversed(actions) if RE_CLEANUP.search(a)), actions[-1])

    s1 = f"The robot begins by orienting to the relevant part of the video and grounding the subtask in what is visible: {early}."
    mid_bits = []
    if verify:
        mid_bits.append(f"It then verifies the key detail directly from the frames ({verify}).")
    if report:
        mid_bits.append(f"After verification, it communicates the result clearly ({report}).")
    if not mid_bits:
        mid_bits.append("It verifies the key detail from the frames and communicates the result clearly.")

    s2 = " ".join(mid_bits)
    s3 = f"To finish safely, it completes cleanup and returns to standby: {cleanup}."

    return [s1 + " " + s2, s3]

def qc_and_repair_guidance(
    raw_lines: List[str],
    allowed_frame_indices: List[int],
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    raw_text = "\n".join(_normalize_blank_lines(raw_lines)).strip()

    sections = _split_into_sections(raw_text)
    sections, miss_issues = _autofill_missing_sections(sections)
    issues.extend(miss_issues)

    # Repair steps
    repaired_steps, step_issues = _repair_steps_section(
        sections.get("ORDERED_ROBOT_ACTION_STEPS:", []),
        allowed_frame_indices
    )
    issues.extend(step_issues)
    sections["ORDERED_ROBOT_ACTION_STEPS:"] = repaired_steps

    # Repair story if low quality
    story_lines = sections.get("SUBTASK_STORY:", [])
    if _story_bad(story_lines):
        issues.append("story_rewritten_from_steps")
        sections["SUBTASK_STORY:"] = _rewrite_story_from_steps(repaired_steps)

    # Fix truncation anywhere else (including GLOBAL_SUMMARY, etc.)
    sections, trunc_issues = _repair_truncation_anywhere(sections)
    issues.extend(trunc_issues)

    # Rebuild ordered text
    out_lines: List[str] = []
    for h in HEADINGS:
        out_lines.append(h)
        out_lines.append("")
        content = sections.get(h, [])
        if not any(ln.strip() for ln in content):
            if h == "FRAME_BASED_OBSERVATIONS:":
                content = ["- [frames=[all]] (auto-filled) Use the captions above as observations."]
            elif h == "PRECONDITIONS_FOR_ROBOT:":
                content = ["- (auto-filled) Robot is powered on; cameras/sensors are active."]
            elif h == "SUCCESS_CRITERIA:":
                content = ["- (auto-filled) Robot completes the subtask and reports outcomes."]
            else:
                content = ["(auto-filled)"]
            issues.append(f"section_empty_autofill:{h}")
        out_lines.extend(content)
        out_lines.append("")

    out_lines = _normalize_blank_lines(out_lines)

    # Repair frame tags anywhere
    out_lines, tag_issues = _repair_frames_tags_anywhere(out_lines, allowed_frame_indices)
    issues.extend(tag_issues)

    return out_lines, issues

# ---------------------------------------------------------------------
# GENERATION
# ---------------------------------------------------------------------

def _run_generation(prompt: str, tokenizer, model) -> str:
    messages = [
        {"role": "system", "content": "You are a precise assistant that follows formatting instructions exactly."},
        {"role": "user", "content": prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def generate_guidance_lines_for_subtask(
    video_index: str,
    title: str,
    task_obj: Dict[str, Any],
    subtask_obj: Dict[str, Any],
    tokenizer,
    model,
) -> Optional[List[str]]:
    frames = subtask_obj.get("frames") or []
    if not frames:
        return None

    allowed_frame_indices = sorted({int(f.get("frame_index", 0)) for f in frames})

    best_lines: Optional[List[str]] = None
    best_issues: Optional[List[str]] = None

    for attempt in range(MAX_RETRIES + 1):
        strict_fix = (attempt > 0)
        prompt = build_subtask_prompt(video_index, title, task_obj, subtask_obj, strict_fix=strict_fix)

        try:
            raw = _run_generation(prompt, tokenizer, model)
        except torch.cuda.OutOfMemoryError:
            print("[ERROR] CUDA OOM during generation; skipping this subtask.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        lines = _normalize_blank_lines([ln.rstrip() for ln in raw.splitlines()])
        cleaned, issues = qc_and_repair_guidance(lines, allowed_frame_indices)

        if best_lines is None or (best_issues is not None and len(issues) < len(best_issues)):
            best_lines = cleaned
            best_issues = issues

        if not issues:
            break

        if attempt == 0:
            print(f"    [QC] attempt {attempt+1}/{MAX_RETRIES+1}: {', '.join(issues[:12])}" + (" ..." if len(issues) > 12 else ""))

    if best_lines is not None and best_issues:
        uniq = []
        for x in best_issues:
            if x not in uniq:
                uniq.append(x)
        print(f"    [QC] fixed/issues: {', '.join(uniq[:12])}" + (" ..." if len(uniq) > 12 else ""))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_lines

# ---------------------------------------------------------------------
# PROCESS ONE VIDEO
# ---------------------------------------------------------------------

def process_video_file(path: Path, tokenizer, model):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video_index = data.get("video_index") or path.stem
    title = data.get("title", "")

    print(f"[INFO] Processing {video_index}.json")

    tasks = data.get("tasks") or []
    for ti, task in enumerate(tasks):
        subtasks = task.get("subtasks") or []
        for si, sub in enumerate(subtasks):
            frames = sub.get("frames") or []
            if not frames:
                continue

            print(f"  [LLM] video={video_index}, task={ti}, subtask={si}, frames={len(frames)}")
            guidance_lines = generate_guidance_lines_for_subtask(
                video_index=video_index,
                title=title,
                task_obj=task,
                subtask_obj=sub,
                tokenizer=tokenizer,
                model=model,
            )
            if guidance_lines is None:
                continue
            sub["guidance_text"] = guidance_lines

    out_path = OUTPUT_DIR / path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {out_path}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    tokenizer, model = load_model_and_tokenizer()

    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        print(f"[INFO] Running ONLY on {len(selected)} provided files under {FRAME_CAPTION_DIR}")
        for name in selected:
            path = FRAME_CAPTION_DIR / name
            if not path.exists():
                print(f"[WARN] File not found: {path}")
                continue
            try:
                process_video_file(path, tokenizer, model)
            except Exception as e:
                print(f"[ERROR] Failed processing {name}: {e}")
        print("[INFO] Done selected videos.")
        return

    if not FRAME_CAPTION_DIR.exists():
        print(f"[ERROR] FRAME_CAPTION_DIR does not exist: {FRAME_CAPTION_DIR}")
        print("       Fix by setting: export FRAME_CAPTION_DIR=/path/to/results/frame_captions")
        return

    files = sorted(p for p in FRAME_CAPTION_DIR.iterdir() if p.is_file() and p.suffix == ".json")
    if not files:
        print(f"[WARN] No JSON files found under {FRAME_CAPTION_DIR}")
        return

    print(f"[INFO] Found {len(files)} frame-caption JSON files.")

    for path in files:
        try:
            process_video_file(path, tokenizer, model)
        except Exception as e:
            print(f"[ERROR] Failed processing {path.name}: {e}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
