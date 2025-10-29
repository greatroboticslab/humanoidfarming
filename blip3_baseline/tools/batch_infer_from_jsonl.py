import os, json, re, argparse
from pathlib import Path
from tqdm import tqdm
import inference as inf

# ---------------- Helpers ----------------

def _first_user_text(messages):
    """Extract plain user text from BLIP-style messages."""
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                parts = []
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b.get("text", ""))
                return "\n".join([p for p in parts if p]).strip()
    return ""

FORMAT_RULES = (
    "Return ONLY numbered steps (1) 2) 3) ...). "
    "No headings, no 'Plan:', no role labels, no ellipses. "
    "Each step must be a complete sentence ending with a period."
)

def _wrap_prompt(prompt: str) -> str:
    return prompt.strip() + "\n\nOutput requirements: " + FORMAT_RULES + "\n"

def _clean_text(t: str) -> str:
    t = (t or "").strip().replace("\r", "")
    # Flatten JSON-ish list like [{'type': 'text', 'text': '...'}]
    if t.startswith("["):
        try:
            arr = json.loads(t)
            if isinstance(arr, list):
                t = "\n".join(item.get("text", str(item)) for item in arr)
        except Exception:
            pass
    # Strip boilerplate, keep line breaks
    t = re.sub(r"^(system|user|assistant)\s*$", "", t, flags=re.I | re.M)
    t = re.sub(r"^You are Qwen.*\n?", "", t, flags=re.I)
    t = re.sub(r"^Plan:\s*\n?", "", t, flags=re.I)
    t = t.replace("**", "")
    # Collapse spaces/tabs only
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t

def _looks_incomplete(t: str) -> bool:
    lines = [ln.strip() for ln in (t or "").replace("\r","").split("\n") if ln.strip()]
    if not lines:
        return True
    # any numbered line with empty/trivial body -> incomplete
    for ln in lines:
        m = re.match(r"^(\d+)\)\s*(.*)$", ln)
        if m:
            body = (m.group(2) or "").strip()
            if body in {"", ".", "…"} or body.startswith("...") or len(body) < 5:
                return True
    # ellipses anywhere => incomplete
    if "..." in t or "…" in t:
        return True
    # overall very short
    if sum(len(x.split()) for x in lines) < 20:
        return True
    return False

def _pretty_format_steps(t: str) -> str:
    """Ensure each step is on its own line; convert 'Step N:' or 'N.' to 'N)' and keep sub-bullets."""
    t = (t or "").replace("\r", "")
    # Normalize common patterns to line starts
    t = re.sub(r"\*\*\s*Step\s+(\d+)\s*:\s*\*\*", r"\n\1) ", t, flags=re.I)
    t = re.sub(r"\bStep\s+(\d+)\s*:", r"\n\1) ", t, flags=re.I)
    # Insert newline before any inline 'N)' or 'N.' stuck in sentences
    t = re.sub(r"(?<!^)(?<!\n)\s(?=\d+\))", "\n", t)
    t = re.sub(r"(?<!^)(?<!\n)\s(?=\d+\.)", "\n", t)
    # Turn inline hyphens into sub-bullets
    t = re.sub(r"\s-\s", "\n   - ", t)
    t = t.replace("**", "").strip()

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    out = []
    for ln in lines:
        ln = re.sub(r"^(\d+)\.", r"\1) ", ln)  # 1. -> 1)
        # Only add a period if there's a non-trivial body
        if re.match(r"^\d+\)", ln) and not re.search(r"[.!?]$", ln):
            if len(ln.split()) > 2:
                ln += "."
        out.append(ln)
    return "\n".join(out).strip()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--peft_path", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model (returns tokenizer, model, processor)
    tokenizer, model, processor = inf.load_vlm(args.vlm)

    # optional LoRA
    if args.peft_path:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.peft_path)
            try:
                model = model.merge_and_unload()
            except Exception as e:
                print("[WARN] Could not merge LoRA:", e)
        except Exception as e:
            print("[WARN] LoRA not applied:", e)

    # read jsonl
    with open(args.src, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for i, rec in enumerate(tqdm(records, desc="Processing")):
        base_prompt = _first_user_text(rec.get("messages", [])) or rec.get("prompt", "").strip()
        if not base_prompt:
            out_path = Path(args.out_dir) / f"sample_{i:06d}.txt"
            with open(out_path, "w") as fo:
                fo.write("\n")
            continue

        # Keep the task title
        header_title = base_prompt.strip()
        if not header_title.lower().startswith("task:"):
            header_title = f"Task: {header_title}"
        header = header_title + "\n" + "-"*60 + "\n"

        # First pass
        prompt = _wrap_prompt(base_prompt)
        text = inf.run_vlm(
            model=model,
            processor=processor,
            prompt=prompt,
            image_path=None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        text = _clean_text(text)

        # Retry if incomplete (up to 2 retries total)
        if _looks_incomplete(text):
            prompt2 = _wrap_prompt(base_prompt + "\nRewrite the steps WITHOUT ellipses. Each step must be informative, 12–25 words, and end with a period.")
            text2 = inf.run_vlm(
                model=model,
                processor=processor,
                prompt=prompt2,
                image_path=None,
                max_new_tokens=max(400, args.max_new_tokens * 2),
                temperature=min(0.2, args.temperature),
            )
            text2 = _clean_text(text2)
            if len(text2.split()) > len(text.split()):
                text = text2

        if _looks_incomplete(text):
            prompt3 = _wrap_prompt(base_prompt + "\nProvide COMPLETE steps. No placeholders. Each step must describe a concrete action and end with a period.")
            text3 = inf.run_vlm(
                model=model,
                processor=processor,
                prompt=prompt3,
                image_path=None,
                max_new_tokens=max(500, int(1.5 * args.max_new_tokens)),
                temperature=0.1,
            )
            text3 = _clean_text(text3)
            if len(text3.split()) > len(text.split()):
                text = text3

        # Format and save
        text = _pretty_format_steps(text)
        out_path = Path(args.out_dir) / f"sample_{i:06d}.txt"
        with open(out_path, "w") as fo:
            fo.write(header + text + "\n")

if __name__ == "__main__":
    main()
