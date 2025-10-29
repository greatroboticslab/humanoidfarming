#!/usr/bin/env python3
import os, json, csv, argparse, glob, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

SYS_PROMPT = (
    "You are BLIP-3, a robotics planning assistant for farm operations.\n"
    "Always respond with a clear, numbered list of subtasks first, then optional notes.\n"
    "Be concise, deterministic, and safety-aware."
)
def as_str(x):
    if x is None: return ""
    if isinstance(x, str): return x
    if isinstance(x, (list, tuple)):
        try: return "\n".join(str(s).strip() for s in x if str(s).strip())
        except Exception: return "\n".join(map(str,x))
    if isinstance(x, dict):
        for k in ("text","value","content","answer","plan","name","title"):
            if k in x and x[k]: return as_str(x[k])
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def pull_candidates(rec):
    if not isinstance(rec, dict): return "", ""
    kl = {k.lower(): k for k in rec.keys()}
    # task-ish
    task = ""
    for key in ("task","task_name","instruction","prompt","question","query","goal","title"):
        if key in kl:
            task = as_str(rec[kl[key]]); 
            if task: break
    if not task and "task" in kl and isinstance(rec[kl["task"]], dict):
        task = as_str(rec[kl["task"]].get("text","") or rec[kl["task"]].get("name",""))
    # plan-ish
    ans = ""
    for key in ("plan","answer","output","response","steps","subtasks","solution"):
        if key in kl:
            ans = as_str(rec[kl[key]]); 
            if ans: break
    if not ans and "plan" in kl and isinstance(rec[kl["plan"]], dict):
        if "steps" in rec[kl["plan"]]: ans = as_str(rec[kl["plan"]]["steps"])
        elif "text" in rec[kl["plan"]]: ans = as_str(rec[kl["plan"]]["text"])
    return task.strip(), ans.strip()

def build_sample(idx, task_text, answer_text):
    user_text = f"Task: {task_text}\nPlease break it into step-by-step subtasks."
    a = (answer_text or "").strip()
    if a and not a.lstrip().startswith(("1)","1.","Step 1","- ")):
        if "\n" not in a and len(a.split()) < 6: a = f"Plan:\n1) {a}"
        else: a = "Plan:\n" + a
    if not a: a = "Plan:\n1) ..."
    return {
        "id": f"sample_{idx:06d}",
        "system": SYS_PROMPT,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]}
        ]
    }

def load_json(p):
    try:
        with open(p, "r") as f: return json.load(f)
    except Exception: return None

def iter_json_files(src, debug=False):
    for p in glob.glob(str(src / "**/*.json"), recursive=True):
        obj = load_json(p)
        if obj is None: continue
        if isinstance(obj, list):
            for rec in obj: yield p, rec
        elif isinstance(obj, dict):
            yield p, obj
        if debug: print(f"[DEBUG] JSON {p}", file=sys.stderr)

def iter_jsonl_files(src, debug=False):
    for p in glob.glob(str(src / "**/*.jsonl"), recursive=True):
        try:
            with open(p, "r") as f:
                for line in f:
                    try: yield p, json.loads(line)
                    except Exception: continue
            if debug: print(f"[DEBUG] JSONL {p}", file=sys.stderr)
        except Exception: continue

def iter_csv_tsv_files(src, debug=False):
    for ext, dialect in (("*.csv","excel"),("*.tsv","excel-tab")):
        for p in glob.glob(str(src / f"**/{ext}"), recursive=True):
            try:
                with open(p, newline="") as f:
                    reader = csv.DictReader(f, dialect=dialect)
                    for row in reader: yield p, row
                if debug: print(f"[DEBUG] {ext.upper()} {p}", file=sys.stderr)
            except Exception: continue

def iter_yaml_files(src, debug=False):
    if yaml is None: return
    for ext in ("*.yaml","*.yml"):
        for p in glob.glob(str(src / f"**/{ext}"), recursive=True):
            try:
                with open(p, "r") as f: obj = yaml.safe_load(f)
                if isinstance(obj, list):
                    for rec in obj: yield p, rec
                elif isinstance(obj, dict):
                    yield p, obj
                if debug: print(f"[DEBUG] YAML {p}", file=sys.stderr)
            except Exception: continue

def parse_md_text(block):
    txt = as_str(block)
    task, plan = "", ""
    lines = [l.rstrip() for l in txt.splitlines()]
    cur = None
    acc = {"task":[],"plan":[]}
    for l in lines:
        L = l.strip().lower()
        if L.startswith("task:"):
            cur = "task"; acc[cur].append(l.split(":",1)[1].strip()); continue
        if L.startswith(("plan:","subtasks:","steps:")):
            cur = "plan"; tail = l.split(":",1)[1].strip()
            if tail: acc[cur].append(tail); continue
        if cur: acc[cur].append(l)
    task = "\n".join(acc["task"]).strip()
    plan = "\n".join(acc["plan"]).strip()
    return task, plan

def iter_text_md_files(src, debug=False):
    for ext in ("*.md","*.txt"):
        for p in glob.glob(str(src / f"**/{ext}"), recursive=True):
            try:
                with open(p, "r") as f: payload = f.read()
                t, a = parse_md_text(payload)
                if t or a: yield p, {"_md_task": t, "_md_plan": a}
                if debug: print(f"[DEBUG] {ext.upper()} {p}", file=sys.stderr)
            except Exception: continue

def iter_pair_files(src, debug=False):
    pairs = [("*.task","*.plan"), ("*.instr","*.out")]
    for patt_in, patt_out in pairs:
        ins = {Path(p).stem: p for p in glob.glob(str(src / f"**/{patt_in}"), recursive=True)}
        outs = {Path(p).stem: p for p in glob.glob(str(src / f"**/{patt_out}"), recursive=True)}
        for stem, pin in ins.items():
            if stem in outs:
                try:
                    with open(pin) as fi, open(outs[stem]) as fo:
                        t = fi.read().strip(); a = fo.read().strip()
                        yield pin+"+"+outs[stem], {"task": t, "plan": a}
                    if debug: print(f"[DEBUG] Pair {pin} + {outs[stem]}", file=sys.stderr)
                except Exception: continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    total_in = total_ok = 0; seen=set()
    with open(out,"w") as fout:
        streams = (
            list(iter_json_files(src,args.debug)) +
            list(iter_jsonl_files(src,args.debug)) +
            list(iter_csv_tsv_files(src,args.debug)) +
            list(iter_yaml_files(src,args.debug)) +
            list(iter_text_md_files(src,args.debug)) +
            list(iter_pair_files(src,args.debug))
        )
        for p, rec in streams:
            total_in += 1
            if isinstance(rec, dict) and ("_md_task" in rec or "_md_plan" in rec):
                task, ans = rec.get("_md_task",""), rec.get("_md_plan","")
            else:
                task, ans = pull_candidates(rec)
            if not task:
                if args.debug: print(f"[DEBUG] No task parsed from {p}", file=sys.stderr)
                continue
            key = (task, ans)
            if key in seen: continue
            seen.add(key)
            sample = build_sample(total_ok, task, ans)
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            total_ok += 1

    print(f"Wrote {total_ok} samples to {out}")
    if args.debug:
        print(f"[DEBUG] Inspected items: {total_in}, unique kept: {total_ok}", file=sys.stderr)

if __name__ == "__main__":
    main()
