# Scripts

This directory contains the complete modular pipeline used to convert videos
(YouTube or local MP4) into task-structured, timestamp-aligned, visually grounded,
robot-centric guidance suitable for instruction tuning and downstream robotic use.

The scripts are organized as sequential stages that progressively transform raw
video data into structured supervision.

---

# Pipeline Overview

Video  
→ Transcript Extraction  
→ Speech Filtering  
→ Task & Subtask Extraction  
→ Timestamp Grounding  
→ Frame Extraction  
→ Frame Captioning (Visual Grounding)  
→ Robot-Centric Guidance Generation  
→ Export

Each stage produces intermediate artifacts stored in the `results/` directory.

---

# Design Principles

- Modular stage-wise processing
- Reproducible intermediate outputs
- Exact timestamp recovery via Whisper segment IDs
- Long-video support using Qwen long-context models
- Separation of generation (LLM) and validation (Python rules)
- Robot-centric structured instruction format

---

# Script Descriptions

## 1. Transcript Extraction

### timestamps_using_yt_subtitles.py

Extracts auto-generated captions from YouTube videos.

**Input**
- YouTube URL

**Output**
- JSON transcript with:
  - text
  - segment IDs
  - start and end timestamps

**Usage**

```bash
python scripts/timestamps_using_yt_subtitles.py   --url <youtube_url>   --out_dir results/timestamps_using_yt_subtitles
```

---

### timestamps_using_whisper.py

Generates transcript and timestamps from local MP4 videos using Whisper ASR.

**Input**
- Local `.mp4` video

**Output**
- JSON transcript with Whisper segments

**Usage**

```bash
python scripts/timestamps_using_whisper.py   --file path/to/video.mp4   --out_dir results/timestamps_using_whisper
```

---

## 2. Speech vs No-Speech Filtering

### split_speech_vs_nospeech.py

Filters videos based on transcript length to remove non-instructional content.

**Purpose**
- Remove music-only or silent videos
- Improve downstream task extraction quality

**Output**

results/speech_vs_nospeech_videos/

**Usage**

```bash
python scripts/split_speech_vs_nospeech.py   --transcripts_dir results/timestamps_using_whisper
```

---

## 3. Task Extraction & Timestamp Assignment

### tasks_with_timestamps.py (Primary Pipeline)

Uses an LLM (Qwen / S1) to:

- Determine content relevance
- Extract TASK / SUBTASK hierarchy
- Predict Whisper `SEGMENT_IDS` for each subtask
- Convert segment IDs → exact timestamps

**Why Segment IDs?**

- Avoids heuristic matching
- Enables precise frame grounding
- Robust to paraphrasing
- Works for long transcripts

**Usage**

```bash
python scripts/tasks_with_timestamps.py   --model Qwen/Qwen2.5-7B-Instruct   --prompt_file scripts/prompt_for_tasks_with_timestamps.txt   --input_dir results/timestamps_using_whisper   --output_dir results/tasks_with_timestamps
```

---

### Legacy Versions

#### align_tasks_with_timestamp.py

Heuristic alignment using text matching.

Limitations:
- Approximate timestamps
- Breaks on paraphrasing
- Not used in final pipeline

---

#### tasks_with_timestamps_using_pyseqmatcher.py

Uses Python `SequenceMatcher`.

Limitations:
- Poor performance on long videos
- Not semantically robust

---

## 4. Frame Extraction

### extract_frames_from_subtasks.py

Converts subtask timestamp intervals into representative frames.

**Frame Sampling Strategy**

- Minimum: 3 frames
- Maximum: 20 frames
- Approx. 1 frame every 5 seconds
- Uniform spacing within subtask time range

Subtasks without timestamps are skipped.

**Output**

results/frame_extractions/<video_id>/
  taskXX_subYY_fZZ.jpg

---

## 5. Frame Deduplication & Captioning

### frame_captions.py

Performs visual grounding.

**Steps**

1. Remove near-duplicate frames using perceptual hashing (pHash)
2. Generate one-sentence captions using:
   - InstructBLIP or
   - Qwen-VL

**Output**

results/frame_captions/<video_id>.json

Each subtask includes:

- frame index
- relative path
- caption text

---

## 6. Robot-Centric Guidance Generation

### subtask_guidance.py

Generates structured robot-oriented guidance per subtask.

**Inputs**

- Subtask description
- Timestamp range
- Frame captions

**Generated Sections**

- GLOBAL_SUMMARY
- FRAME_BASED_OBSERVATIONS
- INTEGRATED_SCENE_UNDERSTANDING
- PRECONDITIONS_FOR_ROBOT
- SUCCESS_CRITERIA
- ORDERED_ROBOT_ACTION_STEPS
- SUBTASK_STORY

Output is attached to each subtask as `guidance_text`.

---

### subtask_guidance (Improved).py

Enhanced version with:

- Stronger structure enforcement
- Reduced hallucinations
- Better robot action formatting

---

## 7. Export Utilities

### batch_export_all_videos.py

Converts nested JSON guidance into readable TXT.

**Output**

final_guidance_txt/<video_id>.txt

Adds reference back into source JSON.

---

# Recommended Execution Order

1. Transcript Extraction (Whisper or YouTube)
2. Speech Filtering
3. Task Extraction with Segment IDs
4. Frame Extraction
5. Frame Captioning
6. Subtask Guidance Generation
7. Export to TXT / HTML / DOCX

---

# Dependencies

- Whisper
- Qwen / vLLM
- Vision-language model (InstructBLIP or Qwen-VL)
- OpenCV / ffmpeg
- Python 3.9+

---

# Notes

- All scripts are designed to be run independently or as part of batch pipelines.
- Intermediate outputs are stored for reproducibility and debugging.
- The system supports long videos through long-context LLMs.
