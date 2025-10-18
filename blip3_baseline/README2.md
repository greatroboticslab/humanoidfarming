BLIP3o Inference

This repository provides a unified inference script for running BLIP3o across multiple multimodal tasks using the Qwen2.5-VL model for understanding and Stable Diffusion for generation/editing.

Supported Tasks
Task	Description	Mode Flag
Text → Text (T2T)	Text reasoning or response generation	--mode t2t
Image → Text (I2T)	Image captioning or understanding	--mode i2t
Text → Image (T2I)	Image generation from a text prompt	--mode t2i
Image → Image (I2I)	Image editing, stylization, or inpainting	--mode i2i
️ Setup

Create and activate the environment:

conda create -n blip3o python=3.11 -y
conda activate blip3o
pip install --upgrade pip setuptools
pip install -r requirements.txt


If diffusers causes issues, you can remove it safely:

pip uninstall diffusers

Model Checkpoints

The script automatically downloads models from Hugging Face:

Qwen2.5-VL: for text and vision-language reasoning
→ Qwen/Qwen2.5-VL-7B-Instruct

Stable Diffusion (default): for image generation and editing
→ runwayml/stable-diffusion-v1-5

If you run out of home directory space, set Hugging Face caches to a larger folder before running:

export HF_HOME=/ocean/projects/<project_id>/<username>/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME

Running Inference (Manually)
1. Text → Text
python inference.py --mode t2t --prompt "Explain photosynthesis in one line."

2. Image → Text
python inference.py --mode i2t \
    --image path/to/image.jpg \
    --prompt "Describe the image in detail."

3. Text → Image
python inference.py --mode t2i \
    --prompt "A futuristic city under the stars" \
    --output_dir results/

4. Image → Image

Stylization / Variation

python inference.py --mode i2i \
    --image path/to/input.jpg \
    --prompt "Make it look like a watercolor painting."


Inpainting (with mask)

python inference.py --mode i2i \
    --image path/to/input.jpg \
    --mask path/to/mask.png \
    --prompt "Fill the blank area with a blue sky."

Running via Slurm

You can run inference as a batch job on HPC systems like Bridges2 using:

sbatch run_inference.sbatch


Example contents of run_inference.sbatch:

#!/bin/bash
#SBATCH --job-name=blip3_infer
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=logs/blip3_infer_%j.out

module load cuda/12.1
source ~/.bashrc
conda activate blip3o

export HF_HOME=/ocean/projects/$PROJECT_ID/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME

python inference.py --mode t2i --prompt "A cute robot farming in a green valley" --output_dir results/

Output

All generated files (images or text results) are saved in the folder specified by:

--output_dir


Default is results/.

Text outputs are saved as .txt, and generated/edited images are saved as .png with timestamps.

Troubleshooting
Error	Fix
Disk quota exceeded	Set HF_HOME to a project path with enough storage (see above).
CUDA is required but not available	Make sure you are on a GPU node or job (use srun or sbatch on Bridges2).
diffusers or torch_dtype mismatch	Reinstall diffusers and transformers from requirements.txt.

 Citation

If you use this work or codebase, please cite:

@article{chen2025blip3,
  title={BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset},
  author={Chen, Jiuhai and others},
  journal={arXiv preprint arXiv:2505.09568},
  year={2025}
}
