#!/usr/bin/env python3
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/qwen_train/ckpts/qwen2p5-7b-lora"

OUT_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/qwen_train/inference_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def postprocess(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    out, buf = [], ""

    def is_header(l):
        return bool(re.match(r"^[A-Z0-9_ ]{3,}:\s*$", l.strip()))

    def is_list(l):
        s = l.strip()
        return s.startswith("- ") or bool(re.match(r"^\d+\.\s+", s))

    for line in lines:
        s = line.rstrip()
        if not s.strip():
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append("")
            continue

        if is_header(s) or is_list(s):
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append(s)
            continue

        # de-wrap weird "word-per-line" outputs
        if len(s.strip()) <= 18:
            buf += (" " if buf else "") + s.strip()
        else:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append(s)

    if buf:
        out.append(buf.strip())

    return "\n".join(out).strip()

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()

    system = (
        "You are an expert robot planning assistant. "
        "Generate ONLY robot-centric guidance. "
        "Do NOT describe human actions or human step sequences. "
        "Output must be cleanly formatted with clear headings, bullets, and numbered steps. "
        "No word-per-line formatting."
    )

    user = """TASK:
Highlight the role of soil testing in assessing nutrient content, pH, and organic matter.

FRAMES:
1. A person using soil testing equipment
2. A soil sample in a petri dish

Generate structured guidance with EXACT headings:
GLOBAL_SUMMARY:
FRAME_BASED_OBSERVATIONS:
ORDERED_ROBOT_ACTION_STEPS:
PRECONDITIONS_FOR_ROBOT:
SUCCESS_CRITERIA:
POSTCONDITIONS:
"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    cleaned = postprocess(decoded)

    out_path = OUT_DIR / "example_soil_testing_guidance_no_human.txt"
    out_path.write_text(cleaned, encoding="utf-8")

    print("\n=== INFERENCE OUTPUT ===\n")
    print(cleaned)
    print(f"\n[OK] Saved inference result to:\n  {out_path}")

if __name__ == "__main__":
    main()
