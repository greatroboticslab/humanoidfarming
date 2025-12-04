import json, os, re, argparse, random
from pathlib import Path

def first_text_from_role(messages, role):
    for m in messages or []:
        if m.get("role") == role:
            c = m.get("content", [])
            if isinstance(c, list):
                parts = [blk.get("text","") for blk in c if isinstance(blk, dict) and blk.get("type")=="text"]
                t = "\n".join(p for p in parts if p).strip()
                if t: return t
            elif isinstance(c, str):
                return c.strip()
    return ""

def set_assistant_text(messages, new_text):
    for m in messages:
        if m.get("role") == "assistant":
            c = m.get("content", [])
            if isinstance(c, list) and c:
                placed = False
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        blk["text"] = new_text
                        placed = True
                        break
                if not placed:
                    c.append({"type":"text","text":new_text})
                m["content"] = c
            else:
                m["content"] = [{"type":"text","text":new_text}]
            return
    # if no assistant message, append one
    messages.append({"role":"assistant","content":[{"type":"text","text":new_text}]})

def extract_body_from_txt(txt):
    txt = txt.replace("\r","")
    lines = txt.split("\n")
    body = "\n".join(lines)
    for i,ln in enumerate(lines):
        if re.match(r"^-{5,}$", ln.strip()):
            body = "\n".join(lines[i+1:]).strip()
            break
    return body.strip()

AUG_USER_TAILS = [
    " Provide numbered subtasks with concise justifications.",
    " Focus on safety, measurement units, and necessary tools.",
    " Emphasize precision, sustainability, and error checks.",
    " Include validation steps and fallback procedures.",
]

AUG_ASSISTANT_HINTS = [
    "\nNotes: Add safety checks, tool prep, and validation criteria.",
    "\nNotes: Include measurement units and environmental safeguards.",
    "\nNotes: Mention common pitfalls and how to avoid them.",
]

def augment_user(u):
    u = u.strip()
    # keep existing Task:... intact; just add a small requirement
    return u + random.choice(AUG_USER_TAILS)

def augment_assistant(a):
    a = a.strip()
    # If it's already a good step list, just append a small note
    return a + random.choice(AUG_ASSISTANT_HINTS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/train_blip.jsonl")
    ap.add_argument("--out", dest="out", default="data/train_blip_aug.jsonl")
    ap.add_argument("--fill_from", default="results/base_for_fill",
                    help="Directory with base outputs (sample_XXXXXX.txt) to replace placeholder assistants.")
    args = ap.parse_args()

    fill_dir = Path(args.fill_from)
    use_fill = fill_dir.exists()

    total, filled = 0, 0
    with open(args.inp, "r", encoding="utf-8", errors="ignore") as fi, \
         open(args.out, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            uid = rec.get("id","")

            user_text = first_text_from_role(msgs, "user")
            asst_text = first_text_from_role(msgs, "assistant")

            # If we have base outputs, try to replace placeholder assistant like "Plan:\n1) ..."
            if use_fill:
                # prefer exact id match: sample_XXXXXX.txt
                cand = None
                if uid:
                    # direct
                    p = fill_dir / f"{uid}.txt"
                    if p.exists(): cand = p
                    # or standard naming
                    if not cand:
                        m = re.search(r"(\d{6})", uid)
                        if m:
                            p2 = fill_dir / f"sample_{m.group(1)}.txt"
                            if p2.exists(): cand = p2
                # also try sequentially named files if id missing
                if cand and cand.exists():
                    body = extract_body_from_txt(cand.read_text(encoding="utf-8", errors="ignore"))
                    if body and ("1)" in body or "2)" in body):
                        set_assistant_text(msgs, body)
                        asst_text = body
                        filled += 1

            # Light augmentation
            if user_text:
                new_user = augment_user(user_text)
                # replace the user text in-place
                for m in msgs:
                    if m.get("role") == "user":
                        c = m.get("content", [])
                        if isinstance(c, list):
                            replaced = False
                            for blk in c:
                                if isinstance(blk, dict) and blk.get("type")=="text":
                                    blk["text"] = new_user
                                    replaced = True
                                    break
                            if not replaced:
                                c.append({"type":"text","text":new_user})
                            m["content"] = c
                        else:
                            m["content"] = [{"type":"text","text":new_user}]

            if asst_text:
                set_assistant_text(msgs, augment_assistant(asst_text))

            rec["messages"] = msgs
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1

    print(f"✅ Wrote {args.out}  | total={total}, assistant_filled_from_base={filled}")

if __name__ == "__main__":
    main()
