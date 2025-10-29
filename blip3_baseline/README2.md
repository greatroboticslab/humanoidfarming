# 🧠 BLIP3o Inference

This repository provides a unified inference and fine-tuning framework for running **BLIP3o** across multimodal tasks using **Qwen2.5-VL** for reasoning and **Stable Diffusion** for image generation or editing.

---

## Supported Tasks

| Task | Description | Mode Flag |
|------|--------------|------------|
| Text → Text (T2T) | Text reasoning or step-by-step task generation | `--mode t2t` |
| Image → Text (I2T) | Image captioning or multimodal understanding | `--mode i2t` |
| Text → Image (T2I) | Image generation from textual descriptions | `--mode t2i` |
| Image → Image (I2I) | Image editing, variation, or inpainting | `--mode i2i` |

---


## ⚙️ Setup

Create and activate the environment:

```bash
conda create -n blip3o python=3.11 -y
conda activate blip3o
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

If `diffusers` causes issues:
```bash
pip uninstall diffusers -y
pip install diffusers==0.30.0
```

---

## 🧠 Model Checkpoints

| Component | Source | Description |
|------------|---------|-------------|
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | Multimodal reasoning backbone |
| Stable Diffusion | `runwayml/stable-diffusion-v1-5` | Image generation & editing |
| LoRA Fine-tuned Adapter | `outputs/lora-qwen3b-aug-fp16-full/` | Domain-specific fine-tuning (humanoid farming tasks) |

---

## 🚀 Running Inference

### 1️⃣ Text → Text (Reasoning)
```bash
python inference.py --mode t2t --prompt "Transplanting a tomato plant to a larger pot."
```

### 2️⃣ Image → Text
```bash
python inference.py --mode i2t --image path/to/image.jpg --prompt "Describe the image."
```

### 3️⃣ Text → Image
```bash
python inference.py --mode t2i --prompt "A futuristic greenhouse on Mars" --output_dir results/
```

### 4️⃣ Image → Image
```bash
python inference.py --mode i2i --image path/to/input.jpg --prompt "Turn this into a watercolor painting."
```

---

## 💻 Running via Slurm (HPC / Bridges2)

Example `run_inference.sbatch`:

```bash
#!/bin/bash -l
#SBATCH -J blip3_infer
#SBATCH -A cis240145p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-16:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH -t 04:00:00
#SBATCH -o logs/blip3_infer_%j.out
#SBATCH -e logs/blip3_infer_%j.err
set -e
set -o pipefail

PROJECT_DIR=/ocean/projects/cis240145p/byler/anusha/humanoidfarming/blip3_baseline
HF_CACHE=/ocean/projects/cis240145p/byler/anusha/hf_cache

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip3o

export HF_HOME=$HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export XDG_CACHE_HOME=$HF_CACHE

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/results"
cd "$PROJECT_DIR"

echo "Node: $(hostname)"
nvidia-smi || true

srun python inference.py   --mode t2i   --diffusion runwayml/stable-diffusion-v1-5   --prompt "A cute robot farming in a green valley"   --output_dir results/
```

---

## 📦 Output

- Text outputs: `.txt`  
- Images: `.png`  
- All results saved in `results/` with timestamped filenames.  


---

## 🧰 Troubleshooting

| Error | Cause | Fix |
|-------|--------|-----|
| `Disk quota exceeded` | HF cache on home dir | Use `$HF_HOME` in project path |
| `tensors does not require grad` | Gradient flow disabled | Use updated FP16 trainer |
| `diffusers` mismatch | Version conflict | `pip install diffusers==0.30.0` |
| `CUDA unavailable` | CPU node job | Re-run on GPU node or use `--gres=gpu:1` |

---

## 🧩 Text → Text: Base vs Trained Model Comparison

BLIP3o supports two inference modes for text reasoning tasks:
- **Base Model:** The original **Qwen2.5-3B-Instruct** model (no fine-tuning).
- **Trained Model (LoRA):** Fine-tuned on domain-specific data using our `train_fp16_lora_qwen.py` pipeline.

---

### 🔍 Why Base and Trained Were Initially the Same

Early experiments showed nearly identical outputs between base and trained runs.  
This occurred because:

1. **Gradients were not applied:**  
   The first trainer (`train_blip_chat.py`) didn’t attach gradients to the LoRA layers, so no parameters were updated.

2. **Dataset structure mismatch:**  
   The initial dataset used a `messages` format that the trainer didn’t parse correctly, resulting in skipped examples.

3. **Tiny LoRA without optimization:**  
   Although the LoRA adapter existed (~0.4% trainable params), it stayed uninitialized since the backward pass never ran.

**Result:**  
The LoRA checkpoint contained no learned weights — producing outputs identical to the base Qwen model.

---

### ✅ Improvements Implemented

| Area | Fix Applied | Result |
|------|--------------|---------|
| **Training Script** | Switched to `scripts/train_fp16_lora_qwen.py` with proper gradient checkpointing | Actual fine-tuning occurred |
| **Dataset Format** | Rebuilt with `tools/augment_training_data` to ensure valid user/assistant pairs | Consistent supervised learning input |
| **LoRA Configuration** | Mixed-precision FP16 training, correct gradient flow | Stable optimization and weight updates |
| **Inference Separation** | Independent base vs trained runs and directories | Easy comparison of learning impact |

---

### 💡 Current Outcome

| Model | Behavior | Example Output |
|--------|-----------|----------------|
| **Base (Qwen2.5-3B-Instruct)** | Produces structured but generic plans | “1) Assess plant health. 2) Repot carefully. 3) Water and monitor.” |
| **Trained (LoRA Fine-tuned)** | Adds domain reasoning and contextual awareness | “1) Inspect roots for pests before repotting. 2) Fill with well-draining mix. 3) Monitor sunlight exposure post-transplant.” |

The trained model now produces **more domain-specific**, **better sequenced**, and **contextually richer** step-by-step reasoning.

---

## 📖 Citation

```bibtex
@article{chen2025blip3,
  title={BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset},
  author={Chen, Jiuhai and others},
  journal={arXiv preprint arXiv:2505.09568},
  year={2025}
}
```
