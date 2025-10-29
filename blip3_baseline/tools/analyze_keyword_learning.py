import json, re, os, glob, argparse, math
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def extract_user_text(rec):
    # BLIP-style: rec["system"], rec["messages"] [{role, content:[{type:text,text}]}]
    msgs = rec.get("messages", [])
    parts=[]
    for m in msgs:
        if m.get("role")=="user":
            content = m.get("content", [])
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type")=="text":
                        parts.append(blk.get("text",""))
            elif isinstance(content, str):
                parts.append(content)
    return "\n".join(parts).strip()

def normalize_id(rec):
    rid = rec.get("id","")
    # Prefer sample_XXXXXX if present
    m = re.search(r"(\d{6})", rid)
    if m:
        return f"sample_{m.group(1)}"
    # Fallback: sequential counter will be handled by caller if needed
    return rid or None

def read_txt(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

def count_keywords(text, keywords):
    toks = tokenize(text)
    bag = Counter(toks)
    counts = {}
    for kw in keywords:
        kws = kw.lower().split()
        if len(kws)==1:
            counts[kw] = bag.get(kws[0], 0)
        else:
            # simple n-gram count
            joined = " ".join(toks)
            counts[kw] = len(re.findall(r"\b"+re.escape(" ".join(kws))+r"\b", joined))
    return counts

def distinct_n(tokens, n=2):
    if len(tokens)<n: return 0.0
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / (len(tokens)-n+1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_blip_aug.jsonl")
    ap.add_argument("--base_dir", default="results/base_run_full_fp16")
    ap.add_argument("--trained_dir", default="results/trained_run_full_fp16")
    ap.add_argument("--keywords_file", default="", help="Optional newline-separated keywords to track")
    ap.add_argument("--top_auto_keywords", type=int, default=60, help="Auto-extract top N keywords from tasks if no file")
    ap.add_argument("--out_dir", default="results/analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load dataset & build id -> index map
    records = list(load_jsonl(args.data))
    rows = []
    id_to_idx = {}
    for i,rec in enumerate(records):
        sid = normalize_id(rec)
        if not sid:
            sid = f"sample_{i:06d}"
        id_to_idx[sid]=i
        rows.append({"sample_id":sid, "task_text": extract_user_text(rec)})

    df = pd.DataFrame(rows)

    # 2) Load generations
    def find_txts(d):
        files = {}
        for p in glob.glob(os.path.join(d, "*.txt")):
            name = Path(p).stem  # sample_XXXXXX
            files[name]=p
        return files

    base_files = find_txts(args.base_dir)
    tr_files   = find_txts(args.trained_dir)

    df["base_path"]    = df["sample_id"].map(base_files)
    df["trained_path"] = df["sample_id"].map(tr_files)
    df["base_text"]    = df["base_path"].apply(lambda p: read_txt(p) if p else "")
    df["trained_text"] = df["trained_path"].apply(lambda p: read_txt(p) if p else "")

    # 3) Keywords to track
    if args.keywords_file and Path(args.keywords_file).exists():
        keywords = [ln.strip() for ln in Path(args.keywords_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        # auto-extract: top content words from tasks
        all_task_toks = [t for s in df["task_text"].tolist() for t in tokenize(s)]
        stop = set("""the a an to and or for of in on at from with without by into over under than as is are be been being
                      your you we they i he she it this that these those can should would could drone drones djis dji
                      task plan step steps list numbered provide please""".split())
        freq = Counter([t for t in all_task_toks if t not in stop and not t.isdigit()])
        keywords = [w for w,_ in freq.most_common(args.top_auto_keywords)]
        # Also seed some domain terms if present
        seeds = ["spray","nozzle","payload","radar","mapping","pesticide","runoff","soil","ph","moisture",
                 "transplant","irrigation","sensor","autonomous","battery","coverage","hectare","safety",
                 "monitor","fertilizer","yield","greenhouse","precision","calibration"]
        for s in seeds:
            if s not in keywords: keywords.append(s)

    # 4) Compute per-sample metrics
    per_rows = []
    for _,r in df.iterrows():
        sid = r["sample_id"]
        task = r["task_text"]; base = r["base_text"]; tr = r["trained_text"]
        base_tokens = tokenize(base); tr_tokens = tokenize(tr)

        base_kw = count_keywords(base, keywords)
        tr_kw   = count_keywords(tr, keywords)

        base_hits = sum(1 for k,v in base_kw.items() if v>0)
        tr_hits   = sum(1 for k,v in tr_kw.items() if v>0)

        per_rows.append({
            "sample_id": sid,
            "task_len_tokens": len(tokenize(task)),
            "base_len_tokens": len(base_tokens),
            "trained_len_tokens": len(tr_tokens),
            "base_distinct2": distinct_n(base_tokens,2),
            "trained_distinct2": distinct_n(tr_tokens,2),
            "base_keyword_hits": base_hits,
            "trained_keyword_hits": tr_hits,
            "delta_keyword_hits": tr_hits - base_hits
        })

    per_df = pd.DataFrame(per_rows).sort_values("delta_keyword_hits", ascending=False)
    per_df.to_csv(os.path.join(args.out_dir, "per_sample_metrics.csv"), index=False)

    # 5) Aggregate per-keyword counts
    kw_rows=[]
    for kw in keywords:
        base_total = 0
        tr_total   = 0
        base_docs=0; tr_docs=0
        for _,r in df.iterrows():
            b = count_keywords(r["base_text"], [kw])[kw]
            t = count_keywords(r["trained_text"], [kw])[kw]
            base_total += b; tr_total += t
            base_docs += 1 if b>0 else 0
            tr_docs   += 1 if t>0 else 0
        kw_rows.append({
            "keyword": kw,
            "base_total": base_total,
            "trained_total": tr_total,
            "delta_total": tr_total - base_total,
            "base_doc_freq": base_docs,
            "trained_doc_freq": tr_docs,
            "delta_doc_freq": tr_docs - base_docs
        })
    kw_df = pd.DataFrame(kw_rows).sort_values(["delta_total","delta_doc_freq"], ascending=False)
    kw_df.to_csv(os.path.join(args.out_dir, "keyword_aggregates.csv"), index=False)

    # 6) Global summary
    summary = {
        "num_samples": len(df),
        "have_base_outputs": int((df["base_text"].str.len()>0).sum()),
        "have_trained_outputs": int((df["trained_text"].str.len()>0).sum()),
        "avg_len_base": per_df["base_len_tokens"].mean(),
        "avg_len_trained": per_df["trained_len_tokens"].mean(),
        "avg_distinct2_base": per_df["base_distinct2"].mean(),
        "avg_distinct2_trained": per_df["trained_distinct2"].mean(),
        "avg_keyword_hits_base": per_df["base_keyword_hits"].mean(),
        "avg_keyword_hits_trained": per_df["trained_keyword_hits"].mean(),
        "avg_keyword_hits_delta": per_df["delta_keyword_hits"].mean(),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    # 7) Examples table (top 10 with biggest positive keyword delta)
    top = per_df.head(10).merge(df[["sample_id","task_text","base_text","trained_text"]], on="sample_id", how="left")
    # keep shorter previews for markdown
    def preview(s, n=350):
        s = re.sub(r"\s+", " ", (s or "")).strip()
        return (s[:n] + "…") if len(s)>n else s
    top["task_preview"]=top["task_text"].apply(lambda s: preview(s, 160))
    top["base_preview"]=top["base_text"].apply(lambda s: preview(s, 300))
    top["trained_preview"]=top["trained_text"].apply(lambda s: preview(s, 300))
    cols = ["sample_id","delta_keyword_hits","task_preview","base_preview","trained_preview"]
    top_md = top[cols].copy()

    md_path = os.path.join(args.out_dir, "examples_top_delta.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| sample_id | Δ keyword hits | task (preview) | base (preview) | trained (preview) |\n")
        f.write("|---|---:|---|---|---|\n")
        for _,r in top_md.iterrows():
            f.write(f"| {r['sample_id']} | {r['delta_keyword_hits']} | {r['task_preview']} | {r['base_preview']} | {r['trained_preview']} |\n")

    print("✅ Wrote:")
    print(" -", os.path.join(args.out_dir, "summary.csv"))
    print(" -", os.path.join(args.out_dir, "per_sample_metrics.csv"))
    print(" -", os.path.join(args.out_dir, "keyword_aggregates.csv"))
    print(" -", md_path)
