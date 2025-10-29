import json, os, re
from pathlib import Path

root = Path(".")
src_jsonl = root/"data/train_blip.jsonl"
res_dir   = root/"results/base_for_fill"
dst_jsonl = root/"data/train_blip_filled.jsonl"

def extract_steps_from_txt(txt: str) -> str:
    # remove header lines up to dashed separator
    txt = txt.replace("\r", "")
    parts = txt.split("\n")
    try:
        sep_idx = next(i for i,l in enumerate(parts) if re.match(r"^-{5,}$", l.strip()))
        body = "\n".join(parts[sep_idx+1:]).strip()
    except StopIteration:
        # fallback: keep everything
        body = "\n".join(parts).strip()
    return body

def main():
    n_in, n_ok, n_miss = 0,0,0
    with open(src_jsonl, "r", encoding="utf-8", errors="ignore") as fi, \
         open(dst_jsonl, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip(): continue
            rec = json.loads(line)
            sid = rec.get("id") or ""
            step_file = res_dir / f"{sid}.txt"
            if not step_file.exists():
                # also support sample_000123 style names
                step_file = res_dir / f"{sid if sid.endswith('.txt') else sid}.txt"
                if not step_file.exists():
                    # last try: match the numeric suffix
                    m = re.search(r"(\d{6})", sid)
                    if m:
                        step_file = res_dir / f"sample_{m.group(1)}.txt"
            n_in += 1
            if step_file.exists():
                body = extract_steps_from_txt(step_file.read_text(encoding="utf-8", errors="ignore"))
                # Replace assistant content with real steps
                msgs = rec.get("messages", [])
                for m in msgs:
                    if m.get("role") == "assistant":
                        c = m.get("content", [])
                        if isinstance(c, list) and c:
                            # overwrite first text block or append
                            found = False
                            for blk in c:
                                if isinstance(blk, dict) and blk.get("type")=="text":
                                    blk["text"] = body
                                    found = True
                                    break
                            if not found:
                                c.append({"type":"text","text":body})
                        elif isinstance(c, str):
                            m["content"] = body
                        else:
                            m["content"] = [{"type":"text","text":body}]
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
                n_ok += 1
            else:
                # keep original if missing (still useful)
                fo.write(json.dumps(rec, ensure_ascii=False)+"\n")
                n_miss += 1
    print(f"Filled dataset saved to: {dst_jsonl}")
    print(f"Total: {n_in}, replaced: {n_ok}, missing: {n_miss}")

if __name__ == "__main__":
    main()
