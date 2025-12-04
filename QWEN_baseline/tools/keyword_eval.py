import argparse, json, re, os, csv
from pathlib import Path
from collections import Counter
import difflib

STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","to","of","in","on","at","by","with",
    "as","is","are","was","were","be","being","been","it","its","this","that","these","those",
    "from","into","over","under","up","down","out","about","than","you","your","we","our","they",
    "their","he","she","his","her","them","i","me","my","mine","yours","ours","theirs","will",
    "can","may","might","should","would","could","must","do","does","did","done","have","has","had",
    "not","no","yes","true","false","very","more","most","less","least"
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")

def tokenize(text):
    toks = [t.lower() for t in WORD_RE.findall(text)]
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]

def bigrams(tokens):
    return [tokens[i] + " " + tokens[i+1] for i in range(len(tokens)-1)]

def read_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def get_ref_text_from_messages(rec):
    msgs = rec.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant":
            content = m.get("content", [])
            if isinstance(content, list):
                texts = [c.get("text","") for c in content if isinstance(c, dict)]
                return "\n".join(texts).strip()
            elif isinstance(content, str):
                return content.strip()
    return ""

def keyword_set(text, vocab_items):
    toks = tokenize(text or "")
    toks_set = set(toks)
    bi_set = set(bigrams(toks))
    present = set()
    for kind, kw, _ in vocab_items:
        if kind=="uni" and kw in toks_set:
            present.add(kw)
        elif kind=="bi" and kw in bi_set:
            present.add(kw)
    return present

def build_vocab(data_path, max_keywords=300, use_bigrams=True, min_df=10):
    from collections import Counter
    df_uni, df_bi = Counter(), Counter()
    n = 0
    for rec in read_jsonl(data_path):
        ref = get_ref_text_from_messages(rec)
        if not ref: continue
        n+=1
        toks = tokenize(ref)
        df_uni.update(set(toks))
        if use_bigrams:
            df_bi.update(set(bigrams(toks)))
    uni = [(t,c) for t,c in df_uni.items() if c>=min_df]
    bi  = [(t,c) for t,c in df_bi.items() if c>=max(2,min_df//2)]
    uni.sort(key=lambda x:(-x[1],x[0]))
    bi.sort(key=lambda x:(-x[1],x[0]))
    vocab=[]
    i=j=0
    while len(vocab)<max_keywords and (i<len(uni) or j<len(bi)):
        if i<len(uni):
            vocab.append(("uni",uni[i][0],uni[i][1])); i+=1
        if len(vocab)>=max_keywords: break
        if j<len(bi):
            vocab.append(("bi",bi[j][0],bi[j][1])); j+=1
    return vocab,n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_blip_aug.jsonl")
    ap.add_argument("--base_dir", default="results/base_run_full_fp16")
    ap.add_argument("--trained_dir", default="results/trained_run_full_fp16")
    ap.add_argument("--out_dir", default="document/texttotext_analysis")
    ap.add_argument("--max_keywords", type=int, default=300)
    ap.add_argument("--min_df", type=int, default=10)
    ap.add_argument("--use_bigrams", action="store_true")
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vocab_items, n_docs = build_vocab(args.data, args.max_keywords, args.use_bigrams, args.min_df)
    vocab=[kw for _,kw,_ in vocab_items]

    # Save vocab
    with open(out/"vocab.csv","w",newline="",encoding="utf-8") as f:
        import csv; w=csv.writer(f); w.writerow(["rank","type","keyword","doc_freq"])
        for i,(k,kw,df) in enumerate(vocab_items,1):
            w.writerow([i,k,kw,df])

    # Evaluate
    from collections import Counter
    base_hit, tr_hit = Counter(), Counter()
    rows=[]
    for rec in read_jsonl(args.data):
        sid=rec.get("id","")
        ref=get_ref_text_from_messages(rec)
        ref_kw=keyword_set(ref,vocab_items)
        if not ref_kw: continue
        base_path=f"{args.base_dir}/{sid}.txt"
        tr_path=f"{args.trained_dir}/{sid}.txt"
        btxt=open(base_path).read() if os.path.exists(base_path) else ""
        ttxt=open(tr_path).read() if os.path.exists(tr_path) else ""
        bkw, tkw = keyword_set(btxt,vocab_items), keyword_set(ttxt,vocab_items)
        cov_b, cov_t = len(ref_kw & bkw)/len(ref_kw), len(ref_kw & tkw)/len(ref_kw)
        rows.append([sid, len(ref_kw), len(ref_kw & bkw), len(ref_kw & tkw), cov_b, cov_t, cov_t-cov_b])
        for kw in ref_kw & bkw: base_hit[kw]+=1
        for kw in ref_kw & tkw: tr_hit[kw]+=1

    # Save results
    with open(out/"per_sample.csv","w",newline="",encoding="utf-8") as f:
        import csv; w=csv.writer(f)
        w.writerow(["sample_id","ref_kw","base_hits","trained_hits","cov_base","cov_trained","delta"])
        w.writerows(rows)

    # Keyword lift
    lifts=[]
    for kw in vocab:
        lifts.append([kw, base_hit.get(kw,0), tr_hit.get(kw,0), tr_hit.get(kw,0)-base_hit.get(kw,0)])
    lifts.sort(key=lambda x:(-x[3],x[0]))
    with open(out/"keyword_lift.csv","w",newline="",encoding="utf-8") as f:
        import csv; w=csv.writer(f)
        w.writerow(["keyword","doc_hits_base","doc_hits_trained","lift"])
        w.writerows(lifts)

    avg_b = sum(r[4] for r in rows)/len(rows)
    avg_t = sum(r[5] for r in rows)/len(rows)
    avg_d = sum(r[6] for r in rows)/len(rows)
    summary = f"""Keyword Analysis Summary
=========================
Data: {args.data}
Base: {args.base_dir}
Trained: {args.trained_dir}
Output folder: {args.out_dir}

Samples evaluated: {len(rows)}
Average coverage (base): {avg_b:.3f}
Average coverage (trained): {avg_t:.3f}
Average improvement: {avg_d:.3f}

Files written:
- {out/'vocab.csv'}
- {out/'per_sample.csv'}
- {out/'keyword_lift.csv'}
"""
    (out/"summary.txt").write_text(summary)
    print(summary)

if __name__=="__main__":
    main()
