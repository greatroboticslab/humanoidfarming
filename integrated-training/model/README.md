# Model

This directory contains all model-related components used for training and inference
in the video-to-robot-guidance pipeline.

It includes both the original Qwen LoRA training setup and the updated improved
training and inference pipeline.

---

# Purpose

The `model/` directory stores:

- Training configurations
- Instruction-tuning datasets
- LoRA checkpoints
- Inference scripts
- Evaluation outputs

These models are responsible for generating structured, robot-centric guidance.

---

# Directory Structure

model/

- qwen_train (Initial Training Pipeline – Dec 12, 2025)
- qwen_training_Updated (Improved Pipeline – Jan 1, 2026)

---

# 1. qwen_train (Initial Version)

## Description

This directory contains the original Qwen LoRA training setup used to fine-tune
Qwen2.5-7B-Instruct on robot-centric guidance data generated from videos.

## Characteristics

- Direct use of raw LLM-generated guidance
- Basic formatting checks
- Limited structural validation
- Used for early experimentation

## Contents

Typical files:

- data/ – instruction tuning dataset
- scripts/ – training and inference scripts
- ckpts/ – LoRA adapter checkpoints
- inference_results/ – sanity-check outputs

## Limitations

- Occasional hallucinated steps
- Inconsistent section formatting
- Missing verification steps
- Higher rule-violation rate

---

# 2. qwen_training_Updated (Improved Pipeline)

## Description

This directory contains the improved training and inference pipeline designed to
produce high-quality, structurally valid robot guidance.

This version introduces a validation-driven data curation workflow.

## Key Improvements

### Generate → Validate → Repair

Training data is filtered using a Python-based validator:

- All required sections must exist
- No truncated outputs allowed
- Step counts are enforced
- Only allowed step types are permitted:
  - navigation
  - manipulation
  - perception
  - communication
- Robot-only actions enforced
- Hallucinated tools removed
- Explicit verification steps required

If validation fails:

LLM (draft) → Python validator → LLM rewrite → Python re-check

Only clean samples are kept.

---

## Training Setup

- Base Model: Qwen/Qwen2.5-7B-Instruct
- Method: Parameter-efficient LoRA
- Trainable parameters: ~0.26%
- Sequence length tuned to avoid GPU OOM
- Stable execution on shared GPU clusters (Slurm)

---

## Produced Artifacts

Typical structure:

qwen_training_Updated/

- data/ – validated instruction dataset
- scripts/ – training and inference utilities
- ckpts/
  - adapter_model.safetensors
  - adapter_config.json
  - trainer_state.json
- inference_results/ – post-training verification outputs

---

## Robust Inference Mode

Supports text-only robot guidance generation.

Properties:

- Structured sections guaranteed
- Allowed action types enforced
- Verification included
- No visual references required

This enables generalization beyond video-based inputs.

---

# Recommended Version

Use **qwen_training_Updated/** for:

- Training new models
- Running inference
- Generating datasets
- Research experiments

The original `qwen_train/` directory is kept for:

- Historical comparison
- Ablation studies
- Reproducibility

---

# Research Significance

The updated pipeline implements a:

Programmatic supervision framework

LLM = generator  
Python = validator  
LLM = rewriter  
Python = final sanitizer

Benefits:

- Cleaner datasets
- Lower hallucination rate
- Stronger structural consistency
- Improved generalization
