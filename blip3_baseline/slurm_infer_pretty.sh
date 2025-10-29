#!/bin/bash -l
#SBATCH -J blip_infer_pretty
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/blip_infer_pretty_%j.out
#SBATCH -e logs/blip_infer_pretty_%j.err
set -eo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
HF_CACHE=$PROJECT/hf_cache
MODEL_ID=Qwen/Qwen2.5-3B-Instruct

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o
export HF_HOME=$HF_CACHE HUGGINGFACE_HUB_CACHE=$HF_CACHE XDG_CACHE_HOME=$HF_CACHE
unset TRANSFORMERS_CACHE || true

mkdir -p "$WORKDIR"/{logs,results}
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
cd "$WORKDIR"

echo "=== RUNTIME ==="; date; hostname; nvidia-smi || true
python - <<'PY'
import torch; print("CUDA avail:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
PY
echo "==============="

python -m tools.batch_infer_from_jsonl \
  --src data/train_blip.jsonl \
  --vlm "$MODEL_ID" \
  --out_dir results/clean_readable \
  --max_new_tokens 400 \
  --temperature 0.3

echo "✅ DONE."; date
