# Qwen Baseline – Multimodal Inference Pipeline

This repository provides a unified inference pipeline for **Qwen2.5-VL** multimodal models and **Stable Diffusion** image generation.  
It supports:

- **Text → Text (t2t)**
- **Image → Text (i2t)**
- **Text → Image (t2i)**
- **Image → Image (i2i)**

The project offers a clean, reproducible baseline for multimodal experiments.

## 📁 Project Structure

```
qwen_baseline/
│
├── scripts/
│   └── inference.py
│
├── data/
│   └── images/
│       ├── xyz.jpg
│       └── abc.jpg
│
├── results/                # Auto-created
│   ├── texttotext
│   ├── imagetotext
│   ├── texttoimage
│   └── imagetoimage
│ 
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/<your-username>/humanoidfarming.git
cd humanoidfarming/qwen_baseline
```

Create and activate a Conda environment:

```bash
conda create -n qwen python=3.10 -y
conda activate qwen
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> GPU with CUDA is recommended for full functionality.

## 🚀 Running Inference

All tasks use the same script:

```bash
python scripts/inference.py --mode <mode> [options...]
```

### 🟦 1. Text → Text (t2t)

```bash
python scripts/inference.py   --mode t2t   --vlm Qwen/Qwen2.5-VL-7B-Instruct   --prompt "Explain Data Science"   --max_new_tokens 160   --temperature 0.7
```

Outputs are saved to:

```
results/texttotext/
```

### 🟩 2. Image → Text (i2t)

```bash
python scripts/inference.py   --mode i2t   --vlm Qwen/Qwen2.5-VL-3B-Instruct   --image data/images/cat.jpg   --prompt "Describe what's happening in this image."   --max_new_tokens 160   --temperature 0.7
```

Saved to:

```
results/imagetotext/
```

### 🟧 3. Text → Image (t2i)

```bash
python scripts/inference.py   --mode t2i   --diffusion runwayml/stable-diffusion-v1-5   --prompt "A cute cat harvesting apples in a sunny orchard."   --steps 30   --guidance 7.5
```

Saved to:

```
results/texttoimage/
```

### 🟥 4. Image → Image (i2i)

#### Img2Img:

```bash
python scripts/inference.py   --mode i2i   --diffusion runwayml/stable-diffusion-v1-5   --image data/images/sample_17.jpg   --prompt "Make it look cinematic with golden-hour lighting."
```

#### Inpainting:

```bash
python scripts/inference.py   --mode i2i   --diffusion runwayml/stable-diffusion-v1-5   --image data/images/face.png   --mask data/images/face_mask.png   --prompt "Add sunglasses to this person."
```

Saved to:

```
results/imagetoimage/
```

## 📦 Models

Downloaded automatically via Hugging Face Hub:

- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `runwayml/stable-diffusion-v1-5`

## 💾 Output Organization

```
results/texttotext/
results/imagetotext/
results/texttoimage/
results/imagetoimage/
```


