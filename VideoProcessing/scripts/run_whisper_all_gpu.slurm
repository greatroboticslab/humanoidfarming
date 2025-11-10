#!/bin/bash -l
#SBATCH -J whisper_all
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH -t 08:00:00
#SBATCH -o logs/whisper_all_%j.out
#SBATCH -e logs/whisper_all_%j.err
set -eo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/VideoProcessing
HF_CACHE=$PROJECT/hf_cache

source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export XDG_CACHE_HOME="$HF_CACHE"
mkdir -p "$HF_CACHE" "$WORKDIR"/{logs,results}

cd "$WORKDIR"

echo "=== RUNTIME INFO ==="
date
hostname
nvidia-smi || true
echo "===================="

# Allow per-file failures without killing the whole job
set +e

while read -r VIDEO_PATH; do
  [[ -z "$VIDEO_PATH" ]] && continue

  echo ">>> Processing $VIDEO_PATH"
  python scripts/whisper_json_only.py \
    --file "$VIDEO_PATH" \
    --out_dir "results" \
    --model "small" \
    --device cpu \
    --compute_type int8 \
    --beam_size 1 \
    --overwrite

  STATUS=$?
  if [[ $STATUS -ne 0 ]]; then
    echo "!!! Failed on $VIDEO_PATH with exit code $STATUS" >&2
  fi
done < all_videos_list.txt

echo "=== DONE ==="
