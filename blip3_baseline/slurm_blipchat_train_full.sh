#!/bin/bash -l
#SBATCH -J blipchat_full
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH -t 12:00:00
#SBATCH -o logs/blipchat_full_%j.out
#SBATCH -e logs/blipchat_full_%j.err

set -euo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
HF_CACHE=$PROJECT/hf_cache

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o

export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export XDG_CACHE_HOME=$HF_CACHE

mkdir -p "$WORKDIR/logs" "$WORKDIR/outputs"
cd "$WORKDIR"

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python train_blip_chat.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train_blip.jsonl \
  --out outputs/lora-qwen3b-blipchat \
  --batch 8 \
  --grad_accum 8 \
  --lr 2e-4 \
  --epochs 3

echo "Done."
