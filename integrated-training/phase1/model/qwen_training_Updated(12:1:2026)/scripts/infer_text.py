#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------
# REQUIRED HEADINGS (text-only inference)
# -----------------------------
HEADINGS = [
    "GLOBAL_SUMMARY:",
    "FRAME_BASED_OBSERVATIONS:",
    "INTEGRATED_SCENE_UNDERSTANDING:",
    "PRECONDITIONS_FOR_ROBOT:",
    "SUCCESS_CRITERIA:",
    "ORDERED_ROBOT_ACTION_STEPS:",
    "SUBTASK_STORY:",
]

TEXT_ONLY_FRAME_BLOCK = (
    "FRAME_BASED_OBSERVATIONS:\n\n"
    "None (text-only mode; no frames provided).\n"
)

ALLOWED_TYPES = {"navigation", "manipulation", "perception", "communication"}

TYPE_FIX_MAP = {
    "cleanup": "manipulation",
    "clean_up": "manipulation",
    "clean": "manipulation",
    "verification": "perception",
    "sensing": "perception",
    "analysis": "perception",
    "planning": "perception",
    "actuation": "manipulation",
    "movement": "navigation",
}

# Ban any vision/scene grounding language
BANNED = re.compile(
    r"\b("
    r"frame|frames|image|images|caption|captions|video|camera|visual|visually|visible|see|seen|scene|shown|showing|"
    r"grounding|grounded|in what is visible|in the environment|in the picture|in the video"
    r")\b",
    re.IGNORECASE,
)

RE_STEP = re.compile(r"^\s*(\d+)\.\s*\[type=([a-zA-Z_]+)\]\s*(.+?)\s*$")
RE_VERIFY = re.compile(r"\b(verify|check|confirm|measure|cross-check|validate)\b", re.IGNORECASE)
RE_ACTIONFUL = re.compile(
    r"\b(collect|scoop|sample|mix|insert|dip|calibrate|read|record|log|report|rinse|clean|store|open|close|start|stop|stow)\b",
    re.IGNORECASE,
)
RE_PH = re.compile(r"\bph\b", re.IGNORECASE)

MIN_STEPS = 7
MAX_STEPS = 10


TEXT_ONLY_SYSTEM = f"""You are a precise assistant generating robot-centric guidance from TEXT ONLY.

Hard rules:
- Do NOT mention frames/images/captions/video/camera/visibility/scenes/grounding/shown/seen.
- Do NOT assume a specific environment (no "garden/kitchen/field"). Use "workspace" or "task area".
- Use only these step types: navigation, manipulation, perception, communication.
- Include at least one explicit verification word: verify/check/confirm/measure/cross-check.
- Include at least one explicit pH measurement reference (pH).

Output format:
Use exactly these headings, in this order:
GLOBAL_SUMMARY:
FRAME_BASED_OBSERVATIONS:
INTEGRATED_SCENE_UNDERSTANDING:
PRECONDITIONS_FOR_ROBOT:
SUCCESS_CRITERIA:
ORDERED_ROBOT_ACTION_STEPS:
SUBTASK_STORY:

Formatting:
- Blank line after each heading.
- FRAME_BASED_OBSERVATIONS must be exactly:
  None (text-only mode; no frames provided).
- PRECONDITIONS_FOR_ROBOT and SUCCESS_CRITERIA are bullet lists.
- ORDERED_ROBOT_ACTION_STEPS is a numbered list:
  1. [type=navigation] ...
- Produce {MIN_STEPS}–{MAX_STEPS} action steps.
- Do NOT output JSON.
"""

REWRITE_SYSTEM = "You are a careful rewriter. Follow the defect report exactly and preserve required headings/format."


def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        try:
            if torch.cuda.device_count() > 0:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def load_base(model_name: str, device: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if device == "cpu":
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=None,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)
    return tok, model


def attach_lora_if_possible(base_model, lora_dir: Optional[str], device: str) -> Tuple[Any, bool]:
    if not lora_dir:
        return base_model, False
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        print(f"[WARN] LoRA path not found: {lora_dir}. Using base model.")
        return base_model, False
    if not PEFT_AVAILABLE:
        print("[WARN] peft not importable. Using base model.")
        return base_model, False
    try:
        model = PeftModel.from_pretrained(base_model, lora_dir, is_trainable=False)
        model.eval()
        model.to(device)
        return model, True
    except Exception as e:
        print(f"[WARN] Failed to load LoRA: {e}. Using base model.")
        return base_model, False


def build_messages(system: str, user: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system.strip()},
            {"role": "user", "content": user}]


@torch.no_grad()
def run_chat(tokenizer, model, system: str, user: str, max_new_tokens: int) -> str:
    messages = build_messages(system, user)
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def extract_steps(text: str) -> List[Tuple[int, str, str]]:
    steps: List[Tuple[int, str, str]] = []
    in_steps = False
    for ln in text.splitlines():
        if ln.strip() == "ORDERED_ROBOT_ACTION_STEPS:":
            in_steps = True
            continue
        if in_steps and ln.strip() == "SUBTASK_STORY:":
            break
        if in_steps:
            m = RE_STEP.match(ln)
            if m:
                steps.append((int(m.group(1)), m.group(2).lower(), m.group(3).strip()))
    return steps


def ensure_frame_based_observations_block(text: str) -> str:
    if "FRAME_BASED_OBSERVATIONS:" not in text:
        if "GLOBAL_SUMMARY:" in text and "INTEGRATED_SCENE_UNDERSTANDING:" in text:
            parts = text.split("INTEGRATED_SCENE_UNDERSTANDING:", 1)
            return parts[0].rstrip() + "\n\n" + TEXT_ONLY_FRAME_BLOCK + "\nINTEGRATED_SCENE_UNDERSTANDING:" + parts[1]
        return TEXT_ONLY_FRAME_BLOCK + "\n" + text

    pre, rest = text.split("FRAME_BASED_OBSERVATIONS:", 1)
    next_idx = len(rest)
    for h in HEADINGS:
        if h == "FRAME_BASED_OBSERVATIONS:":
            continue
        pos = rest.find(h)
        if pos != -1:
            next_idx = min(next_idx, pos)
    tail = rest[next_idx:] if next_idx < len(rest) else ""
    return pre.rstrip() + "\n\n" + TEXT_ONLY_FRAME_BLOCK + "\n" + tail.lstrip()


def normalize_step_types_and_renumber(text: str) -> str:
    lines = text.splitlines()
    out_lines: List[str] = []
    in_steps = False
    step_buf: List[str] = []

    for ln in lines:
        if ln.strip() == "ORDERED_ROBOT_ACTION_STEPS:":
            in_steps = True
            out_lines.append("ORDERED_ROBOT_ACTION_STEPS:")
            continue
        if in_steps and ln.strip() == "SUBTASK_STORY:":
            in_steps = False

            cleaned: List[Tuple[str, str]] = []
            for s in step_buf:
                m = RE_STEP.match(s)
                if not m:
                    continue
                typ = m.group(2).lower()
                action = m.group(3).strip()
                if typ not in ALLOWED_TYPES:
                    typ = TYPE_FIX_MAP.get(typ, "perception")
                cleaned.append((typ, action))

            if len(cleaned) > MAX_STEPS:
                cleaned = cleaned[:MAX_STEPS]
            while len(cleaned) < MIN_STEPS:
                cleaned.append(("perception", "Verify/confirm the pH measurement is consistent before reporting."))

            # remove exact-duplicate actions (keep first)
            seen_actions = set()
            deduped: List[Tuple[str, str]] = []
            for typ, action in cleaned:
                key = action.strip().lower()
                if key in seen_actions:
                    continue
                seen_actions.add(key)
                deduped.append((typ, action))

            # re-pad if dedupe reduced too far
            while len(deduped) < MIN_STEPS:
                deduped.append(("communication", "Report the verified pH value and log it for future reference."))

            if len(deduped) > MAX_STEPS:
                deduped = deduped[:MAX_STEPS]

            for i, (typ, action) in enumerate(deduped, start=1):
                out_lines.append(f"{i}. [type={typ}] {action}")

            out_lines.append("SUBTASK_STORY:")
            continue

        if in_steps:
            step_buf.append(ln)
        else:
            out_lines.append(ln)

    return "\n".join(out_lines).strip()


def sanitize_subtask_story(text: str) -> str:
    if "SUBTASK_STORY:" not in text:
        return text

    pre, story = text.split("SUBTASK_STORY:", 1)
    if not BANNED.search(story):
        return text

    safe_story = (
        "\n"
        "The robot moves to the task area, gathers the required tools, and collects a representative soil sample.\n"
        "It measures soil pH using the available testing device, verifies the reading for consistency, then\n"
        "communicates and logs the result before returning materials and tools to their proper locations.\n"
    )
    return pre.rstrip() + "\n\nSUBTASK_STORY:" + safe_story


def validate(text: str) -> List[str]:
    issues: List[str] = []

    for h in HEADINGS:
        if h not in text:
            issues.append(f"missing_heading:{h}")

    # Frame block exact placeholder
    if "FRAME_BASED_OBSERVATIONS:" in text:
        after = text.split("FRAME_BASED_OBSERVATIONS:", 1)[1]
        next_idx = len(after)
        for h in HEADINGS:
            if h == "FRAME_BASED_OBSERVATIONS:":
                continue
            pos = after.find(h)
            if pos != -1:
                next_idx = min(next_idx, pos)
        block = after[:next_idx].strip()
        if block != "None (text-only mode; no frames provided).":
            issues.append("frame_based_observations_not_placeholder")

    if BANNED.search(text):
        issues.append("contains_banned_visual_language")

    steps = extract_steps(text)
    if not steps:
        issues.append("missing_steps")
        return issues

    if not (MIN_STEPS <= len(steps) <= MAX_STEPS):
        issues.append(f"bad_step_count:{len(steps)}")

    bad_types = [t for _, t, _ in steps if t not in ALLOWED_TYPES]
    if bad_types:
        issues.append(f"bad_step_types:{sorted(set(bad_types))}")

    if not any(RE_VERIFY.search(s) for _, _, s in steps):
        issues.append("missing_verification_word")

    if not any(RE_PH.search(s) for _, _, s in steps):
        issues.append("missing_ph_reference")

    actionful = sum(1 for _, _, s in steps if RE_ACTIONFUL.search(s))
    if actionful < 4:
        issues.append(f"too_few_actionful_steps:{actionful}")

    texts = [s.lower() for _, _, s in steps]
    dup = len(texts) - len(set(texts))
    if dup > 0:
        issues.append(f"duplicate_steps:{dup}")

    return issues


def defect_report(issues: List[str]) -> str:
    bullets = "\n".join(f"- {x}" for x in issues)
    return f"""DEFECT REPORT (fix all items)

{bullets}

REWRITE REQUIREMENTS
- Keep the exact required headings and order.
- FRAME_BASED_OBSERVATIONS must be exactly:
  None (text-only mode; no frames provided).
- Only allowed step types: navigation, manipulation, perception, communication.
- No banned visual language.
- Ensure {MIN_STEPS}-{MAX_STEPS} non-duplicate, actionful steps.
- Include explicit pH reference and at least one verification step.
"""


def generate_with_validate_repair(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    out = run_chat(tokenizer, model, TEXT_ONLY_SYSTEM, prompt, max_new_tokens=max_new_tokens)

    out = ensure_frame_based_observations_block(out)

    issues = validate(out)
    if issues:
        out = run_chat(
            tokenizer,
            model,
            REWRITE_SYSTEM,
            out + "\n\n" + defect_report(issues),
            max_new_tokens=max_new_tokens
        )

    # Hard post-processing (always safe)
    out = ensure_frame_based_observations_block(out)
    out = normalize_step_types_and_renumber(out)
    out = sanitize_subtask_story(out)
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_prompt(row: Dict[str, Any]) -> str:
    if "prompt" in row and isinstance(row["prompt"], str):
        return row["prompt"]
    if "messages" in row and isinstance(row["messages"], list):
        last_user = ""
        for m in row["messages"]:
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = m.get("content", "") or ""
        if last_user:
            return last_user
    return json.dumps(row, ensure_ascii=False)


def slugify(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = s.strip("_")
    if not s:
        s = "prompt"
    return s[:max_len]


def default_out_txt(prompt: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = slugify(prompt)
    return str(Path("qwen_training/inference_results") / f"{ts}_{name}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", default=None)

    # Single prompt mode
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out_txt", default=None, help="Optional. If omitted, auto-saves to qwen_training/inference_results/")
    ap.add_argument("--quiet", type=int, default=0, help="1 = do not print full output (still writes file).")

    # Batch mode
    ap.add_argument("--in_jsonl", default=None)
    ap.add_argument("--out_jsonl", default=None)

    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--prefer_cuda", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    device = pick_device(prefer_cuda=bool(args.prefer_cuda))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[INFO] device={device}, dtype={args.dtype}, base={args.base}")
    if args.lora:
        print(f"[INFO] lora={args.lora}")

    try:
        tokenizer, base_model = load_base(args.base, device=device, dtype=dtype)
    except Exception as e:
        print(f"[WARN] failed loading on {device}: {e}\n[INFO] falling back to CPU")
        device = "cpu"
        tokenizer, base_model = load_base(args.base, device=device, dtype=torch.float32)

    model, used_lora = attach_lora_if_possible(base_model, args.lora, device=device)
    print(f"[INFO] using {'LoRA+base' if used_lora else 'base-only'}")

    # ---------------- Single prompt: AUTO-SAVE ----------------
    if args.prompt:
        out = generate_with_validate_repair(tokenizer, model, args.prompt, args.max_new_tokens)

        out_path = args.out_txt or default_out_txt(args.prompt)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out.rstrip() + "\n")

        print(f"[OK] wrote {out_path}")
        if not args.quiet:
            print(out)
        return

    # ---------------- Batch mode ----------------
    if args.in_jsonl:
        rows = read_jsonl(args.in_jsonl)
        out_rows: List[Dict[str, Any]] = []
        for r in rows:
            p = extract_prompt(r)
            pred = generate_with_validate_repair(tokenizer, model, p, args.max_new_tokens)
            r2 = dict(r)
            r2["prediction"] = pred
            r2["used_lora"] = used_lora
            out_rows.append(r2)

        # Default out_jsonl if user didn't pass it
        out_jsonl = args.out_jsonl or "qwen_training/inference_results/predictions.jsonl"
        write_jsonl(out_jsonl, out_rows)
        print(f"[OK] wrote {out_jsonl} ({len(out_rows)} rows)")
        return

    raise SystemExit("Provide --prompt (auto-saves) OR --in_jsonl (writes to qwen_training/inference_results by default).")


if __name__ == "__main__":
    main()
