# 🌱🤖 Humanoid Farming Multimodal Pipeline

A unified framework for **task extraction**, **visual understanding**, **graph classification**, and **multimodal reasoning** to support intelligent humanoid-robot farming workflows.  
This project brings together four major components—**Video Processing**, **S1 Task Extraction**, **BLIP3o Multimodal Inference**, and **LLaVAGraph classification**—into a coherent pipeline for analyzing farming videos, generating tasks, aligning timestamps, captioning images, and understanding actuator behavior.

---

## 🧭 High-Level Overview

### 1️⃣ Video Processing Module (VideoProcessing/)
Extracts structured representations from YouTube or MP4 videos:

- Timestamped transcripts (YouTube subtitles or Whisper)
- Speech vs. non-speech classification  
- Optional baseline alignment of tasks to timestamps  
- Main S1 pipeline that produces:
  - Tasks  
  - Subtasks  
  - Approximate timestamps (using Python SequenceMatcher)

This module provides **raw semantic structure** essential for downstream robotic planning.

---

### 2️⃣ S1 Baseline – Task Extraction (s1_baseline/)
Uses the **S1 Large Language Model** to:

- Interpret transcripts  
- Extract hierarchical tasks and subtasks  
- Produce MoMask-ready JSONs for animation  
- Shorten captions for BLIP3 image generation

This module acts as the **language-based task planner** of the pipeline.

---

### 3️⃣ BLIP3o – Multimodal Reasoning & Generation (blip3_baseline/)
A unified inference framework using **Qwen2.5-VL** + **Stable Diffusion**.

Supports:

| Mode | Description |
|------|-------------|
| Text → Text | Reasoning, planning, explanations |
| Image → Text | Captioning & visual Q&A |
| Text → Image | Image generation |
| Image → Image | Image editing & transformations |

Includes LoRA fine‑tuning for **humanoid farming–specific knowledge**.

This module provides **rich multimodal intelligence** across text and images.

---

### 4️⃣ LLaVAGraph – Graph Understanding (LlaVAGraph/)
A specialized system for analyzing **laser displacement graphs**:

- Graph captioning  
- Pattern classification (sine, square, noise, etc.)  
- LLAMA-based reasoning for deeper explanation

This module supports actuator diagnostics for robotics in precision agriculture.

---

## 🔗 How the Modules Work Together

```
📹 Video  
    → Transcript  
        → S1 Task Extraction  
            → BLIP3 Multimodal Augmentation  
                → LLaVAGraph (if graphs involved)
```

This creates a smooth flow from **raw video → structured tasks → multimodal understanding**.

---

## 🎯 Project Goals

- Achieve **semantic understanding** of farming scenes  
- Convert human demonstrations into **robot-usable task sequences**  
- Provide **grounded multimodal reasoning**  
- Enable intelligent analysis of actuator/motion graphs  
- Support data pipelines for humanoid farming research

---

## 📑 Citation

Please cite the relevant models and frameworks used within each submodule (listed in each component’s README).

