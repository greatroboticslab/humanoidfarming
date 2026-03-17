# Scripts Directory — Integrated Training Pipelines

This directory contains all executable scripts used across the system for transforming instructional videos into structured robotic knowledge.

It is organized into three main modules:

- preprocessing/ → Shared pipeline for extracting tasks and timestamps from videos  
- pipeline1_robot_guidance_model/ → Generates robot-executable guidance from subtasks  
- pipeline2_structure_task_dataset/ → Builds hierarchical task reasoning datasets  

---

# Directory Structure

scripts/

├── preprocessing/  
├── pipeline1_robot_guidance_model/  
├── pipeline2_structure_task_dataset/  
└── README.md  

---

# 1. Preprocessing Module

## Purpose

Converts raw videos or transcripts into a structured intermediate representation:

tasks_with_timestamps/<video_id>.json

This serves as the shared input for both pipelines.

---

## Key Scripts

- timestamps_using_yt_subtitles.py → Extracts transcripts from YouTube captions  
- timestamps_using_whisper.py → Generates transcripts using Whisper  
- split_speech_vs_nospeech.py → Filters non-instructional videos  
- tasks_with_timestamps.py → Extracts tasks and subtasks using LLM  
- align_tasks_with_timestamp.py → Aligns tasks with timestamps  

---

# 2. Pipeline 1 — Robot Guidance Model

## Purpose

Transforms timestamped subtasks into robot-centric procedural guidance grounded in visual observations.

## Workflow

tasks_with_timestamps → frames → captions → guidance → export → dataset

## Key Scripts

- extract_frames_from_subtasks.py → Extracts frames  
- frame_captions.py → Deduplicates and captions frames  
- subtask_guidance.py → Generates structured robot instructions  
- subtask_guidance_improved.py → Improved version  
- batch_export_all_videos.py → Export to TXT/HTML/DOCX  
- generate_html_reports.py → Visualization  

---

# 3. Pipeline 2 — Structured Task Dataset

## Purpose

Builds hierarchical reasoning structures for long-horizon task understanding.

## Workflow

tasks_with_timestamps → threads → validation → regrouping → missions → blueprints → logs

## Key Scripts

- build_subtask_threads.py → Groups subtasks  
- thread_logic_check.py → Validates logic  
- categorize_tasks_and_subtasks.py → Adds semantic labels  
- regroup_subtasks_coherence.py → Coherent clustering  
- add_submissions_to_coherent_blocks.py → Sub-mission generation  
- generate_task_blueprints.py → Execution plans  
- generate_check_reports.py → Validation  
- generate_training_quality_log.py → Dataset quality  

---

# Design Philosophy

LLM → generate  
Python → validate  
LLM → repair  
Python → verify  

---

# Summary

This directory powers a system that:

- extracts structured knowledge from videos  
- generates robot-executable guidance  
- builds hierarchical reasoning datasets  

Used for:

- robotic learning  
- multimodal AI  
- long-horizon planning  
