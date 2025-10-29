#!/bin/bash -l
#SBATCH -J blip_train_eval_pretty
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/blip_train_eval_pretty_%j.out
#SBATCH -e logs/blip_train_eval_pretty_%j.err
set -eo pipefail

# --- Paths & model selection ---
PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
HF_CACHE=$PROJECT/hf_cache
MODEL_ID=Qwen/Qwen2.5-3B-Instruct

# --- Environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o
export HF_HOME=$HF_CACHE HUGGINGFACE_HUB_CACHE=$HF_CACHE XDG_CACHE_HOME=$HF_CACHE
unset TRANSFORMERS_CACHE || true

mkdir -p "$WORKDIR"/{logs,results,outputs}
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
cd "$WORKDIR"

echo "=== RUNTIME ==="; date; hostname; nvidia-smi || true
python - <<'PY'
import torch; print("CUDA avail:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
echo "==============="

############################
# 1) TRAIN: LoRA fine-tune #
############################
# This writes the adapter to outputs/lora-qwen3b-blipchat
echo "[INFO] Starting LoRA fine-tuning…"
python -u train_blip_chat.py \
  --model "$MODEL_ID" \
  --data data/train_blip.jsonl \
  --out outputs/lora-qwen3b-blipchat \
  --batch 2 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 1 \
  --max_len 2048 \
  --quant 4bit

echo "[INFO] LoRA saved to outputs/lora-qwen3b-blipchat"

#############################################
# 2) INFERENCE: Base model (no LoRA) batch #
#############################################
echo "[INFO] Running BASE model batch inference…"
python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --out_dir results/base_run \
  --max_new_tokens 400 \
  --temperature 0.3

################################################
# 3) INFERENCE: Trained model (with LoRA) batch#
################################################
echo "[INFO] Running TRAINED (LoRA) model batch inference…"
python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --peft_path outputs/lora-qwen3b-blipchat \
  --out_dir results/trained_run \
  --max_new_tokens 400 \
  --temperature 0.3

echo "✅ DONE."; date

