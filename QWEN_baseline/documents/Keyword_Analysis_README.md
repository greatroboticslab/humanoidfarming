# 📘 Keyword Learning Analysis

## Overview
This script evaluates how well the **trained model** has learned domain-specific language by comparing **keyword coverage** between the **base** and **trained** text-to-text outputs.  
It extracts frequent domain keywords from the reference (assistant) responses, then measures how many of those appear in each model’s generated output.

The goal is to quantify **keyword learning improvement** across all ~1500 examples.

---

## 🧩 Inputs
- `data/train_blip_aug.jsonl` — the main training dataset  
- `results/base_run_full_fp16/` — model outputs from the base model  
- `results/trained_run_full_fp16/` — model outputs from the trained model  

Each result directory should contain per-sample text files like:
```
results/base_run_full_fp16/000001.txt
results/trained_run_full_fp16/000001.txt
```

---

## ⚙️ How to Run
From the project root (`blip3_baseline/`):

```bash
python tools/keyword_eval.py   --data data/train_blip_aug.jsonl   --base_dir results/base_run_full_fp16   --trained_dir results/trained_run_full_fp16   --out_dir document/texttotext_analysis   --max_keywords 300   --min_df 10   --use_bigrams
```

This will automatically create the output folder and generate evaluation metrics.

---

## 📂 Output Files
All outputs are saved to:  
`document/texttotext_analysis/`

| File | Description |
|------|--------------|
| `summary.txt` | High-level summary: average keyword coverage, improvement, and counts. |
| `vocab.csv` | List of top domain keywords extracted from reference data. |
| `per_sample.csv` | Per-example coverage for base vs trained model outputs. |
| `keyword_lift.csv` | Keyword-level lift — how much each keyword increased in usage after training. |

---

## 📊 Example Summary

```
Keyword Analysis Summary
=========================
Samples evaluated: 1507
Average coverage (base): 0.312
Average coverage (trained): 0.478
Average improvement: +0.166
```

This indicates that the trained model recalls domain vocabulary **16.6% better** than the base model.

---

## 🧠 Interpretation
- **High “lift”** in `keyword_lift.csv` → The trained model learned those concepts (e.g., *root ball*, *potting mix*).  
- **`per_sample.csv`** → Identify specific examples showing large improvement.  
- **`summary.txt`** → Quantifies overall progress, ideal for reporting in papers.

---

## 🪄 Optional Extension
You can easily:
- Add a **plot of top-20 keyword lifts** (matplotlib).
- Compute **precision** (extra keywords not in reference).
- Use this analysis as a section in your paper under *“Keyword Learning Evaluation.”*
