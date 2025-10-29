#!/bin/bash -l
#SBATCH -J blip_full_train_eval_8h
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=22G
#SBATCH -t 08:00:00
#SBATCH -o logs/blip_full_train_eval_8h_%j.out
#SBATCH -e logs/blip_full_train_eval_8h_%j.err
set -eo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
HF_CACHE=$PROJECT/hf_cache

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o

# --- Set Hugging Face cache ---
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export XDG_CACHE_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
mkdir -p "$HF_CACHE" "$WORKDIR"/{logs,results,outputs}

cd "$WORKDIR"
echo "=== RUNTIME INFO ==="; date; hostname; nvidia-smi || true
python - <<'PY'
import torch; print("CUDA available:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
PY
echo "===================="

# Full dataset (1507 samples)
DATA=data/train_blip_aug.jsonl

# ---------------- TRAIN ----------------
python scripts/train_fp16_lora_qwen.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data "$DATA" \
  --out outputs/lora-qwen3b-aug-fp16-full \
  --batch 1 \
  --grad_accum 4 \
  --lr 5e-5 \
  --epochs 2 \
  --max_len 1536

# ---------------- BASE INFERENCE ----------------
python -m tools.batch_infer_from_jsonl \
  --src "$DATA" \
  --vlm Qwen/Qwen2.5-3B-Instruct \
  --out_dir results/base_run_full_fp16 \
  --max_new_tokens 300 \
  --temperature 0.3

# ---------------- TRAINED INFERENCE ----------------
python -m tools.batch_infer_from_jsonl \
  --src "$DATA" \
  --vlm Qwen/Qwen2.5-3B-Instruct \
  --peft_path outputs/lora-qwen3b-aug-fp16-full \
  --out_dir results/trained_run_full_fp16 \
  --max_new_tokens 300 \
  --temperature 0.3

echo "✅ DONE FULL TRAIN + EVAL"; date
