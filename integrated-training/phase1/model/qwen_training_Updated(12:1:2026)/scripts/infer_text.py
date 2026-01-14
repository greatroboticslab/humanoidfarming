#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


HEADINGS = [
    "GLOBAL_SUMMARY:",
    "INTEGRATED_SCENE_UNDERSTANDING:",
    "PRECONDITIONS_FOR_ROBOT:",
    "SUCCESS_CRITERIA:",
    "ORDERED_ROBOT_ACTION_STEPS:",
    "SUBTASK_STORY:",
]

ALLOWED_TYPES = {"navigation", "manipulation", "perception", "communication"}

# Expanded banned list (catch your exact leakage)
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
    r"\b(collect|scoop|sample|mix|insert|dip|calibrate|read|record|log|report|rinse|clean|store|open|close|start)\b",
    re.IGNORECASE,
)
RE_PH = re.compile(r"\bph\b", re.IGNORECASE)

MIN_STEPS = 7
MAX_STEPS = 10

TEXT_ONLY_SYSTEM = f"""You are a precise assistant generating robot-centric guidance from TEXT ONLY.

Hard rules:
- Do NOT mention frames, images, captions, video, camera, visibility, scenes, grounding, or anything being shown/seen.
- Do NOT assume a specific environment (no "garden/kitchen/field"). Use "workspace" or "task area".
- Use only these step types: navigation, manipulation, perception, communication.
- Include at least one explicit verification word: verify/check/confirm/measure/cross-check.
- Include at least one explicit pH measurement action (measure/read pH).

Output format:
Use exactly these headings, in this order:
GLOBAL_SUMMARY:
INTEGRATED_SCENE_UNDERSTANDING:
PRECONDITIONS_FOR_ROBOT:
SUCCESS_CRITERIA:
ORDERED_ROBOT_ACTION_STEPS:
SUBTASK_STORY:

Formatting:
- Blank line after each heading.
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
    steps = []
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


def validate(text: str) -> List[str]:
    issues = []
    for h in HEADINGS:
        if h not in text:
            issues.append(f"missing_heading:{h}")

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
        issues.append("missing_explicit_ph_reference")

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
- Do not use banned visual wording.
- Do not assume a specific environment; use "workspace" or "task area".
- Ensure {MIN_STEPS}-{MAX_STEPS} non-duplicate, actionful steps.
- Include explicit pH measurement and at least one verification.
"""


def sanitize_subtask_story(text: str) -> str:
    """
    If SUBTASK_STORY contains banned terms, replace story with a safe template derived from steps.
    """
    if "SUBTASK_STORY:" not in text:
        return text

    parts = text.split("SUBTASK_STORY:", 1)
    head = parts[0]
    story = parts[1]

    if not BANNED.search(story):
        return text

    steps = extract_steps(text)
    # Build a generic, non-visual story
    story_lines = [
        "",
        "The robot moves to the task area, gathers the required tools, and collects a representative soil sample.",
        "It performs the pH measurement using the available test device, verifies the reading for consistency,",
        "then communicates and logs the result before returning materials and tools to their proper locations.",
        "",
    ]
    return head + "SUBTASK_STORY:" + "\n".join(story_lines)


def generate_with_validate_repair(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    out = run_chat(tokenizer, model, TEXT_ONLY_SYSTEM, prompt, max_new_tokens=max_new_tokens)
    issues = validate(out)
    if issues:
        out = run_chat(tokenizer, model, REWRITE_SYSTEM, out + "\n\n" + defect_report(issues), max_new_tokens=max_new_tokens)

    # Final safety: sanitize SUBTASK_STORY even if model slips
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--in_jsonl", default=None)
    ap.add_argument("--out_jsonl", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=800)
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

    if args.prompt:
        out = generate_with_validate_repair(tokenizer, model, args.prompt, args.max_new_tokens)
        print(out)
        return

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

        if args.out_jsonl:
            write_jsonl(args.out_jsonl, out_rows)
            print(f"[OK] wrote {args.out_jsonl} ({len(out_rows)} rows)")
        else:
            for r in out_rows[:3]:
                print(json.dumps(r, ensure_ascii=False)[:2000])
        return

    raise SystemExit("Provide --prompt OR --in_jsonl (with optional --out_jsonl).")


if __name__ == "__main__":
    main()
