# Results Directory — Pipeline Outputs

This directory contains all generated outputs from the preprocessing stage and both pipelines.

It is organized into three main sections:

- preprocessing/ → Intermediate outputs (tasks and timestamps)  
- pipeline1_robot_guidance/ → Robot guidance outputs  
- pipeline2_structured_task_dataset/ → Structured reasoning datasets  

---

# Directory Structure

results/

├── preprocessing/  
├── pipeline1_robot_guidance/  
├── pipeline2_structured_task_dataset/  

---

# 1. Preprocessing Outputs

## Purpose

Stores structured task representations extracted from videos.

## Key Contents

- tasks_with_timestamps/<video_id>.json  
  - tasks  
  - subtasks  
  - timestamps  
  - segment IDs  

This is the shared input for both pipelines.

---

# 2. Pipeline 1 — Robot Guidance Outputs

## Purpose

Contains robot-executable procedural guidance generated from subtasks.

## Subfolders

- frame_extractions/ → Sampled frames from videos  
- frame_captions/ → Captions for extracted frames  
- subtask_guidance/ → Structured robot instructions  
- final_guidance_txt/ → Exported readable guidance  

## Output Features

- Visually grounded instructions  
- Step-by-step robot actions  
- Structured sections for execution  

---

# 3. Pipeline 2 — Structured Task Dataset Outputs

## Purpose

Stores hierarchical reasoning structures and validated datasets.

## Subfolders

- subtask_threads/ → Logical task groupings  
- thread_logic/ → LLM validation outputs  
- categorized_threads/ → Semantic annotations  
- coherent_blocks/ → Regrouped subtasks  
- coherent_blocks_with_submissions/ → Sub-missions  
- task_blueprints/ → Execution plans  
- check_reports/ → Validation reports  
- training_quality_log_submissions/ → Dataset quality logs  
- plots/ → Visualization outputs  

## Output Features

- Mission → Sub-mission → Task hierarchy  
- Logical consistency validation  
- Training-ready datasets  

---

# Summary

The results directory stores all intermediate and final outputs of the system:

- Preprocessing → structured task extraction  
- Pipeline 1 → robot execution knowledge  
- Pipeline 2 → hierarchical reasoning datasets  

These outputs support robotic learning, multimodal reasoning, and long-horizon task planning.
