# Full Pipeline Runner

This folder contains scripts for running the complete AI video-to-robot training pipeline.

The system processes instructional videos and generates:

- timestamps
- speech-filtered transcripts
- task/subtask extraction
- frame extraction
- frame captions
- robot guidance
- hierarchical mission/sub-mission structures
- training-quality validation outputs

---

# Configure Number of Videos

Change the number of videos to process:

```bash
NUM_VIDEOS=5
```

or

```bash
NUM_VIDEOS=30
```

depending on the experiment size.

---

# Pipeline Overview

```text
Input Videos
    ↓
Whisper ASR
    ↓
Speech Filtering
    ↓
Task/Subtask Extraction
    ↓
Pipeline 1
    ├── Frame Extraction
    ├── Frame Captioning
    └── Robot Guidance Generation
    ↓
Pipeline 2
    ├── Subtask Threads
    ├── Logic Checking
    ├── Categorization
    ├── Coherent Blocks
    ├── Sub-missions
    └── Task Blueprints
    ↓
Training Quality Logs
```

---

# Scripts

## `run_5_full_pipeline.sh`

Runs the complete pipeline on a small sample set.

Recommended for:
- debugging
- validation
- quick experiments
- development testing

Run:

```bash
bash run_5_full_pipeline.sh
```

---

## `run_10_pipeline.sh`

Runs the pipeline on a larger dataset sample.

Recommended for:
- batch experiments
- larger dataset generation
- pipeline evaluation

Run:

```bash
bash run_10_pipeline.sh
```

---

# Main Outputs

Generated outputs include:

```text
results/
├── timestamps/
├── speech_filter/
├── tasks_with_timestamps/
├── frame_extractions/
├── frame_captions/
├── subtask_guidance/
├── final_guidance/
├── subtask_threads/
├── thread_logic/
├── categorized_threads/
├── coherent_blocks/
├── coherent_blocks_with_submissions/
├── task_blueprints/
├── check_reports/
└── training_quality_log_submissions/
```

---

# Technologies Used

- Python
- Whisper ASR
- Qwen2.5-7B-Instruct
- LoRA fine-tuning
- Vision-Language Models
- InstructBLIP
- Qwen-VL
- PyTorch
- Hugging Face Transformers
- vLLM

---

# Purpose

The goal of this project is to transform instructional videos into structured robotic training datasets for long-horizon robot reasoning and instruction generation.
