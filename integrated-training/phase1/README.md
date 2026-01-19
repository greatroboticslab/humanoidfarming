# 🎬 Timestamp Extraction, Task Alignment, Frame Grounding & Robot Guidance Generation

This repository provides a **modular, end-to-end pipeline** to convert videos (YouTube or local) into:

> **Task-structured, timestamp-aligned, visually grounded, robot-centric guidance**

The pipeline combines **ASR, LLM-based task extraction, timestamp grounding, frame-level visual grounding, and instruction-style guidance generation**.  
All intermediate and final artifacts are saved under the `results/` directory unless otherwise noted.

---

## 🔍 What This Pipeline Does

- Extract timestamped transcripts from:
  - **YouTube videos** (auto-captions)
  - **Local MP4 videos** (Whisper ASR)
- Classify videos into **speech vs no-speech**
- Use an **LLM (Qwen, S1, etc.)** to extract **tasks and subtasks** from transcripts
- Assign timestamps via:
  - **Python SequenceMatcher** (legacy / deprecated)
  - **LLM-predicted Whisper segment IDs** (recommended)
- Extract representative **frames per subtask**
- **Remove near-duplicate frames** and generate **frame captions**
- Generate **robot-centric, structured subtask guidance** from text + frames
- Export results as **JSON, TXT, HTML, or DOCX**
- Prepare datasets for **Qwen LoRA training**

---

## 🧩 1. YouTube Transcript Extraction

**Script:** `scripts/timestamps_using_yt_subtitles.py`

Downloads YouTube auto-captions and saves JSON transcripts containing:
- text
- segment IDs
- start/end timestamps

### ▶️ Run
```bash
python scripts/timestamps_using_yt_subtitles.py \
  --url https://www.youtube.com/watch?v=VIDEO_ID \
  --out_dir results/timestamps_using_yt_subtitles
```

---

## 🧩 2. Local MP4 Transcription (Whisper)

**Script:** `scripts/timestamps_using_whisper.py`

Extracts transcript + timestamps from `.mp4` videos using Whisper.

### ▶️ Run on a Single Video
```bash
python scripts/timestamps_using_whisper.py \
  --file path/to/video.mp4 \
  --out_dir results/timestamps_using_whisper
```

---

## 🧩 3. Speech vs No-Speech Classification

**Script:** `scripts/split_speech_vs_nospeech.py`

Classifies transcripts based on text length to filter non-instructional videos.

### ▶️ Run
```bash
python scripts/split_speech_vs_nospeech.py \
  --transcripts_dir results/timestamps_using_whisper
```

**Output**
```
results/speech_vs_nospeech_videos/
```

---

## 🧩 4. Baseline Task Alignment (Legacy / Optional)

**Script:** `scripts/align_tasks_with_timestamp.py`

Original timestamp alignment approach using heuristic text matching.  
Kept **only for reference and comparison**.

**Output**
```
results/align_tasks_with_timestamp_transcript/
```

---

## 🧩 5. LLM-Based Task Extraction & Timestamp Assignment

### Main Scripts

| Purpose | File |
|------|------|
| Primary pipeline (recommended) | `scripts/tasks_with_timestamps.py` |
| Prompt template | `scripts/prompt_for_tasks_with_timestamps.txt` |
| Legacy SeqMatcher version | `scripts/tasks_with_timestamps_using_pyseqmatcher.py` |
| Legacy prompt | `scripts/prompt_for_using_pyseqmatcher.txt` |

---

### How It Works

The LLM:
1. Reads the **full transcript**
2. Determines whether the content is **relevant**
3. Generates **TASK / SUBTASK** structure
4. Predicts **Whisper SEGMENT_IDS** per subtask
5. Converts segment IDs → exact timestamps

These timestamps are then used downstream for frame extraction and visual grounding.

---

### ❌ Limitations of S1 + Python SequenceMatcher

#### Why S1 Fails on Long Videos
- S1 has a **4096-token context limit**
- Prompts often include:
  - Long instructions
  - 30–200 Whisper segments
  - Full transcripts
- When exceeded, vLLM throws:
```text
decoder prompt length longer than max_model_len=4096
```

When this happens:
- S1 produces no output
- `relevant = false` is set
- This **does NOT mean the video is actually irrelevant**

#### Why SequenceMatcher Is Not Ideal
- Produces **approximate timestamps**
- Degrades on long transcripts
- Fails with repeated phrases
- Breaks when the LLM paraphrases
- Not designed for semantic grounding

---

## ⭐ Recommended Method: Qwen + Explicit SEGMENT_IDS

Modern **Qwen 2.5 (7B / 14B)** models support long context (32k–128k tokens) and follow structured prompts well.

The LLM outputs:
```text
SUBTASK: <description>
SEGMENT_IDS: [3, 4]
```

This directly ties each subtask to Whisper segments, yielding:
- Exact timestamp recovery
- No heuristic alignment
- Stable long-video support
- Higher-quality frame extraction

### ▶️ Run
```bash
python scripts/tasks_with_timestamps.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompt_file scripts/prompt_for_tasks_with_timestamps.txt \
  --input_dir results/timestamps_using_whisper \
  --output_dir results/tasks_with_timestamps
```

---

## 🧩 6. Frame Extraction

**Script:** `scripts/extract_frames_from_subtasks.py`

Converts task/subtask timestamp intervals into representative frames.

### 🎯 Frame Sampling Strategy
For every subtask:
- Minimum **3 frames**
- ~**1 frame every 5 seconds**
- Maximum **20 frames**
- Uniform spacing between `(start, end)`
- Subtasks without valid timestamps are skipped

**Output Structure**
```
results/frame_extractions/<video_id>/
  taskXX_subYY_fZZ.jpg
```

---

## 🧩 7. Frame Deduplication & Captioning (Vision Grounding)

**Script:** `scripts/frame_captions.py` (or updated variants)

### What This Stage Does
1. Removes near-duplicate frames using **perceptual hashing (pHash)**
2. Generates **one-sentence captions** for each unique frame using:
   - InstructBLIP or Qwen-VL
3. Attaches frames + captions to the correct task/subtask

**Output**
```
results/frame_captions/<video_id>.json
```

Each subtask gains a `frames` field containing:
- frame index
- relative path
- caption text

---

## 🧩 8. Subtask Guidance Generation (Robot-Centric)

**Script:** `scripts/subtask_guidance.py`

Uses **Qwen-2.5-7B** to generate **structured, robot-oriented guidance** per subtask using:
- Subtask text
- Time range
- Frame captions

### Generated Sections
- `GLOBAL_SUMMARY`
- `FRAME_BASED_OBSERVATIONS`
- `INTEGRATED_SCENE_UNDERSTANDING`
- `PRECONDITIONS_FOR_ROBOT`
- `SUCCESS_CRITERIA`
- `ORDERED_ROBOT_ACTION_STEPS`
- `SUBTASK_STORY`

The output is attached to each subtask as `guidance_text`.

---

## 🧩 9. Export Final Guidance to TXT

**Script:** `batch_export_all_videos.py`

- Converts nested JSON guidance into **clean, human-readable TXT**
- One TXT per video
- Adds a reference back into the source JSON

**Output**
```
final_guidance_txt/<video_id>.txt
```

---

## 🧩 10. Training (Qwen LoRA)

Location:
```
VideoProcessing/qwen_train/
```

Contains:
- `data/` – instruction-tuning datasets
- `scripts/` – training & inference
- `ckpts/` – LoRA checkpoints
- `inference_results/` – post-training sanity checks

Training uses the **robot-centric guidance text** as supervision.

---
## 🧩 11. Updated Qwen LoRA Training & Robust Inference (NEW)

We updated the Qwen training and inference pipeline to improve **data quality, robustness, and generalization**, especially for **text-only and fallback scenarios**.

### 🔧 What Changed (Compared to Initial Training)

#### 1. Generate–Validate–Repair Training Data
Instead of directly using raw LLM outputs, we now apply a **strict Python-based validation layer** before training:

- **Completeness checks**
  - All required sections must exist
  - No truncated outputs
  - Minimum and maximum step counts enforced
- **Structural correctness**
  - Only allowed step types: `navigation`, `manipulation`, `perception`, `communication`
- **Semantic correctness**
  - Robot-only actions (no human assumptions)
  - No hallucinated tools or environments
  - Explicit verification steps enforced

If any check fails:
```
LLM (draft) → Python validator → LLM (rewrite) → Python re-check
```

Only **defect-free examples** are kept in the final training set.

---


---

####  LoRA Training Improvements
- **Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: Parameter-efficient LoRA
- **Trainable params**: ~0.26% of full model
- **Sequence length**: tuned to avoid GPU OOM
- **Stable Slurm execution** on shared GPUs

Artifacts produced:
```
qwen_training/ckpts/qwen2.5-7b-lora/
  ├── adapter_model.safetensors
  ├── adapter_config.json
  └── trainer_state.json
```

---

#### Generic Text Inference with Fallback
We added a **text-only inference mode** that produces valid robot guidance.

Example:
```bash
python qwen_training/scripts/infer_text.py   
--base Qwen/Qwen2.5-7B-Instruct  
--lora qwen_training/ckpts/qwen2.5-7b-lora  
--prompt "Generate robot-centric guidance for testing soil pH."
```

**Guaranteed properties of inference output**:
- Structured sections
- Allowed step types only
- Verification included
- Cleanup mapped to valid actions automatically
- No visual references

---

#### Alignment with Research Goal
This update directly supports:
- **High-quality instruction tuning**
- **Low-violation supervision**
- **Scalable agent training in the future**

The final pipeline matches the intended research design:
```
LLM = writer
Python = judge
LLM = rewriter
Python = final sanitizer
```

---

### Resulting Benefits
- Cleaner training data
- Lower rule-violation rate
- Robust fallback inference
- Stronger generalization beyond videos
- Ready for agent-style extensions

## 📜 Script Summary

| Script                           | Purpose |
|----------------------------------|------|
| timestamps_using_yt_subtitles.py | Download YouTube auto-captions |
| timestamps_using_whisper.py      | Local Whisper ASR |
| split_speech_vs_nospeech.py      | Speech filtering |
| align_tasks_with_timestamp.py    | Legacy alignment |
| tasks_with_timestamps.py         | LLM task + timestamp extraction |
| extract_frames_from_subtasks.py  | Frame extraction |
| frame_captions.py                | Frame deduplication & captioning |
| subtask_guidance.py              | Robot guidance generation |
| subtask_guidance(Improved).py    | Robot guidance generation |
| batch_export_all_videos.py       | Export guidance to TXT |

---

## ⭐ Recommended Workflow

```
Whisper
  → Qwen (SEGMENT_IDS)
  → Task/Subtask JSON
  → Frame Extraction
  → Frame Captioning
  → Subtask Guidance
  → TXT / HTML / DOCX
  → Qwen LoRA Training
```

### Guarantees
- Long-video support
- True segment-level timestamps
- Visual grounding
- Robot-centric, structured supervision
- Clean, reusable training data
