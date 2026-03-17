#!/bin/bash -l
#SBATCH -J subtask_guidance
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=22G
#SBATCH -t 08:00:00
#SBATCH -o /ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/logs/subtask_guidance_%j.out
#SBATCH -e /ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/logs/subtask_guidance_%j.err

set -eo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/VideoProcessing
HF_CACHE=$PROJECT/hf_cache

export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export XDG_CACHE_HOME="$HF_CACHE"

mkdir -p "$HF_CACHE" \
         "$WORKDIR"/logs \
         "$WORKDIR"/results \
         "$WORKDIR"/results/subtask_guidance

cd "$WORKDIR"

echo "=== RUNTIME INFO ==="
date
hostname
echo "Using Python: /jet/home/byler/miniconda3/envs/blip3/bin/python"
/jet/home/byler/miniconda3/envs/blip3/bin/python -c "import sys; print('Python inside job:', sys.executable)"
echo "===================="

/jet/home/byler/miniconda3/envs/blip3/bin/python scripts/build_subtask_guidance_from_frames.py

echo "=== DONE ==="
