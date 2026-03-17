# Video-to-Robot Guidance & Structured Task Dataset Pipeline

This repository implements a multi-stage system that converts instructional videos into structured robotic knowledge.

From a single video input, the system extracts timestamped tasks and subtasks, then generates two complementary outputs:

- Robot-centric procedural guidance grounded in visual observations
- Hierarchical mission-level task structures for robotic reasoning

These outputs support research in:

- robot learning
- multimodal reasoning
- long-horizon planning
- instruction-tuned LLM agents

---

# System Overview

The pipeline processes videos in three stages.

```
Video / Transcript
        ↓
Shared Processing (task extraction + timestamps)
        ↓
────────────────────────────────
Pipeline 1 → Robot Guidance Model
Pipeline 2 → Structured Task Dataset
────────────────────────────────
```

Both pipelines start from the shared intermediate representation:

```
results/tasks_with_timestamps/<video_id>.json
```

This file contains:

- extracted tasks
- subtasks
- timestamp intervals
- Whisper segment IDs

---

# Shared Preprocessing

The preprocessing stage converts raw videos into timestamp-aligned tasks and subtasks.

Steps include:

## 1. Transcript Extraction

Videos are transcribed using:

- YouTube auto captions
- Whisper ASR for local videos

## 2. Speech Filtering

Videos are classified into:

- speech videos
- non-speech videos

This helps remove non-instructional content.

## 3. Task Extraction with LLMs

An LLM analyzes the transcript and produces:

- TASK
- SUBTASK
- SEGMENT_IDS (Whisper segments)

These segment IDs are converted into exact timestamps.

Output:

```
results/tasks_with_timestamps/
```

This serves as the starting point for both pipelines.

---

# Pipeline 1 — Robot Guidance Generation

Pipeline 1 converts timestamped subtasks into robot-centric procedural guidance grounded in video frames.

Workflow:

```
tasks_with_timestamps
        ↓
frame extraction
        ↓
frame captioning
        ↓
robot guidance generation
        ↓
export guidance
        ↓
LoRA training dataset
```

The model generates structured guidance sections such as:

- GLOBAL_SUMMARY
- FRAME_BASED_OBSERVATIONS
- PRECONDITIONS_FOR_ROBOT
- SUCCESS_CRITERIA
- ORDERED_ROBOT_ACTION_STEPS

These outputs are used to train LoRA-fine-tuned Qwen models for robotic instruction generation.

Detailed documentation:

```
README_pipeline1.md
```

---

# Pipeline 2 — Structured Task Dataset

Pipeline 2 transforms timestamped subtasks into a hierarchical reasoning structure.



Workflow:

```
tasks_with_timestamps
        ↓
thread segmentation
        ↓
logical validation (LLM)
        ↓
category annotation
        ↓
graph-based regrouping
        ↓
sub-mission generation
        ↓
task blueprint generation
        ↓
training quality logging
        ↓
visualization
```

This pipeline produces:

- coherent task groupings
- mission-level reasoning structures
- robot execution blueprints
- validated training datasets

Detailed documentation:

```
README_pipeline2.md
```

---

# Repository Structure

```
integrated-training/

├── scripts/
│
│   ├── preprocessing/
│   │   Shared preprocessing scripts used by both pipelines.
│   │
│   │   ├── timestamps_using_yt_subtitles.py
│   │   │   Extract transcripts and timestamps from YouTube captions.
│   │
│   │   ├── timestamps_using_whisper.py
│   │   │   Generate transcripts and timestamps from local videos using Whisper.
│   │
│   │   ├── split_speech_vs_nospeech.py
│   │   │   Classify transcripts into speech vs non-speech videos.
│   │
│   │   ├── tasks_with_timestamps.py
│   │   │   Main LLM pipeline that extracts tasks and subtasks with timestamps.
│   │
│   │   ├── tasks_with_timestamps_using_pyseqmatcher.py
│   │   │   Legacy alignment method using Python SequenceMatcher.
│   │
│   │   ├── align_tasks_with_timestamp.py
│   │   │   Earlier baseline timestamp alignment method.
│   │
│   │   ├── prompt_for_tasks_with_timestamps.txt
│   │   │   Prompt template for task extraction.
│   │
│   │   └── prompt_for_using_pyseqmatcher.txt
│   │       Prompt template for the legacy alignment pipeline.
│
│
│   ├── pipeline1_robot_guidance_model/
│   │   Pipeline for generating robot-centric procedural guidance
│   │   grounded in video frames.
│   │
│   │   ├── extract_frames_from_subtasks.py
│   │   │   Extract representative frames from timestamp intervals.
│   │
│   │   ├── frame_captions.py
│   │   │   Remove duplicate frames and generate captions using vision models.
│   │
│   │   ├── subtask_guidance.py
│   │   │   Generate structured robot guidance from subtasks and captions.
│   │
│   │   ├── subtask_guidance_improved.py
│   │   │   Improved guidance generation version.
│   │
│   │   ├── batch_export_all_videos.py
│   │   │   Export guidance JSON into readable TXT/HTML documents.
│   │
│   │   └── generate_html_reports.py
│   │       Generate visual reports for robot guidance outputs.
│
│
│   └── pipeline2_structure_task_dataset/
│       Pipeline for constructing hierarchical task datasets
│       and reasoning structures.
│
│       ├── build_subtask_threads.py
│       │   Group subtasks into logical reasoning threads.
│
│       ├── thread_logic_check.py
│       │   LLM-based logical validation of task threads.
│
│       ├── categorize_tasks_and_subtasks.py
│       │   Annotate tasks with semantic categories.
│
│       ├── regroup_subtasks_coherence.py
│       │   Graph-based regrouping of subtasks into coherent blocks.
│
│       ├── add_submissions_to_coherent_blocks.py
│       │   Generate sub-missions from coherent blocks.
│
│       ├── generate_task_blueprints.py
│       │   Produce robot execution blueprints.
│
│       ├── generate_check_reports.py
│       │   Validate logical and structural consistency.
│
│       ├── generate_training_quality_log.py
│       │   Generate dataset quality reports.
│
│       ├── generate_training_quality_log_submissions.py
│       │   Record mission-level dataset quality.
│
│       ├── prompt_for_thread_logic.txt
│       │   LLM prompt used for reasoning validation.
│
│       └── prompt_for_boundary_judge.txt
│           Prompt used to determine coherent block boundaries.
│
│
├── results/
│
│   ├── preprocessing/
│   │   Outputs from transcript extraction and task generation.
│   │
│   │   ├── timestamps_using_whisper/
│   │   ├── timestamps_using_yt_subtitles/
│   │   ├── speech_vs_nospeech_videos/
│   │   ├── tasks_with_timestamps/
│   │   ├── tasks_with_timestamps_using_pyseqmatcher/
│   │   └── align_tasks_with_timestamp_transcript/
│
│
│   ├── pipeline1_robot_guidance/
│   │   Outputs from robot guidance generation.
│   │
│   │   ├── frame_extractions/
│   │   ├── frame_captions/
│   │   ├── subtask_guidance/
│   │   ├── subtask_guidance_improved/
│   │   └── final_guidance_txt/
│
│
│   └── pipeline2_structured_task_dataset/
│       Outputs from hierarchical task dataset generation.
│
│       ├── subtask_threads/
│       ├── thread_logic/
│       ├── categorized_threads/
│       ├── coherent_blocks/
│       ├── coherent_blocks_with_submissions/
│       ├── task_blueprints/
│       ├── check_reports/
│       ├── training_quality_log_submissions/
│       └── plots/
│
│           ├── coherent_blocks_category_trees/
│           ├── logical_map/
│           ├── mission_submissions_plots/
│           ├── task_blueprints_plots/
│           └── human_in_loop_plots/
│
│
└── README.md

---

# Research Design

The system follows a programmatic supervision framework:

```
LLM → generate structured reasoning
Python → validate structure
LLM → repair errors
Python → final verification
```

Benefits:

- consistent task structures
- lower hallucination rate
- cleaner training datasets
- scalable dataset generation

---

# Key Contributions

This repository provides a unified framework for:

- extracting structured knowledge from instructional videos
- grounding robotic instructions in visual observations
- constructing hierarchical reasoning datasets
- generating validated supervision for robotic learning systems

---

# Outputs

The system produces two complementary datasets.

## Robot Guidance Dataset

Visually grounded procedural instructions for robotic execution.

## Structured Task Dataset

Hierarchical mission-level task representations suitable for reasoning and planning.

---

# Future Extensions

Possible research directions include:

- multimodal robotic agents
- autonomous task planning
- reinforcement learning from structured video knowledge
- large-scale robotic instruction datasets
