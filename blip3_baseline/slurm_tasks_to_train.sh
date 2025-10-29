#!/bin/bash -l
#SBATCH -J blip_from_tasks
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/blip_from_tasks_%j.out
#SBATCH -e logs/blip_from_tasks_%j.err
set -eo pipefail

PROJECT=/ocean/projects/cis240145p/byler/anusha/humanoidfarming
WORKDIR=$PROJECT/blip3_baseline
TASKS_DIR=/ocean/projects/cis240145p/byler/anusha/HumanoidRobotTrainingData/s1_baseline/output/tasks
HF_CACHE=$PROJECT/hf_cache

# --- Env ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o
export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE
export XDG_CACHE_HOME=$HF_CACHE
unset TRANSFORMERS_CACHE || true

mkdir -p "$WORKDIR"/{logs,data,outputs,results}
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
cd "$WORKDIR"

echo "Node: $(hostname)"
nvidia-smi || true

# --- Step 1: Convert tasks -> BLIP JSONL ---
python tools/to_blip_chat_jsonl.py \
  --src "$TASKS_DIR" \
  --out data/train_blip.jsonl \
  --debug 2> logs/to_blip_debug_${SLURM_JOB_ID}.log

echo "Training samples:"
wc -l data/train_blip.jsonl

# --- Step 2: Train (QLoRA 4-bit). If no bitsandbytes, switch to: --quant fp16 and lower batch. ---
python train_blip_chat.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data data/train_blip.jsonl \
  --out outputs/lora-qwen3b-fromtasks \
  --batch 8 --grad_accum 8 --lr 2e-4 --epochs 3 \
  --quant 4bit

# --- Step 3: Inference for EVERY sample -> results/<id>.txt ---
python tools/batch_infer_from_jsonl.py \
  --jsonl data/train_blip.jsonl \
  --vlm Qwen/Qwen2.5-3B-Instruct \
  --peft_path outputs/lora-qwen3b-fromtasks \
  --outdir results \
  --max_new_tokens 200 \
  --temperature 0.2

echo "✅ Training + full-dataset inference complete."
echo "Results saved under: $WORKDIR/results/"
