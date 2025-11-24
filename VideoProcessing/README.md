# 🎬 Timestamp Extraction, Task Alignment & Frame Extraction from YouTube & Local Videos

This repository provides a modular pipeline to:

- Extract timestamped transcripts from:
  - YouTube videos (auto-captions)
  - Local MP4 videos (Whisper)
- Classify videos into speech vs no-speech.
- Use an LLM (e.g., **Qwen**, **S1**, etc.) to extract tasks and subtasks from transcripts.
- Assign approximate timestamps to each subtask using Python’s `SequenceMatcher` (deprecated approach).
- Preferably, instruct the LLM (Qwen recommended) to select **Whisper segment IDs** for *high-accuracy timestamps*.
- Optionally extract **key frames / images** for each subtask and generate **HTML / DOCX** visual reports.

All outputs are saved inside the `results/` directory.

---


## 🧭 High-Level Pipeline Overview

```mermaid
flowchart TD

    A[📥 Input Videos] -->|YouTube Links| B1[timestamps_using_yt_subtitles.py]
    A -->|Local MP4 Files| B2[timestamps_using_whisper.py]

    B1 --> C1[results/timestamps_using_yt_subtitles/]
    B2 --> C2[results/timestamps_using_whisper/]

    A -->|Check Audio| B3[split_speech_vs_nospeech.py]
    B3 --> C3[results/speech_vs_nospeech_videos/]

    C2 --> D1[tasks_with_timestamps.py (LLM-Based Segment-ID Alignment)]
    C2 --> D2[tasks_with_timestamps_using_pyseqmatcher.py (Optional Baseline)]

    D1 --> E1[results/timestamps_using_llm/]
    D2 --> E2[results/timestamps_using_pyseqmatcher/]

    E1 --> F[extract_frames_from_timestamps.py]
    F --> G[results/frame_extractions/]

    G --> H1[HTML Reports]
    G --> H2[DOCX Reports]

    H1 --> I1[documents/html/]
    H2 --> I2[documents/docx/]
```

### Text-Only Version

```
YouTube Subtitles        → timestamps_using_yt_subtitles.py        → results/timestamps_using_yt_subtitles/
Local MP4 (Whisper)      → timestamps_using_whisper.py             → results/timestamps_using_whisper/
Speech/No Speech Split   → split_speech_vs_nospeech.py             → results/speech_vs_nospeech_videos/

MAIN LLM PIPELINE
  tasks_with_timestamps.py                      → results/timestamps_using_llm/
  tasks_with_timestamps_using_pyseqmatcher.py   → results/timestamps_using_pyseqmatcher/

Frame Extraction          → extract_frames_from_timestamps.py       → results/frame_extractions/

Reports (HTML & DOCX)     → results/reports/html/ + results/reports/docx/
```
```
YouTube → timestamps_using_yt_subtitles.py → results/timestamps_using_yt_subtitles/
Local MP4 → timestamps_using_whisper.py → results/timestamps_using_whisper/
Speech/No-Speech Split → split_speech_vs_nospeech.py → results/speech_vs_nospeech_videos/
Main LLM Pipeline → tasks_with_timestamps.py → results/ tasks_with_timestamps/
		    tasks_with_timestamps_using_pyseqmatcher.py → results/ tasks_with_timestamps_using_pyseqmatcher/
Frame Extraction → extract_frames_from_timestamps.py → results/frame_extractions/

(Optional Baseline) → align_tasks_with_timestamp.py → results/align_tasks_with_timestamp_transcript/
Reports → HTML & DOCX → results/reports/
```

---

## 🧩 1. YouTube Transcript Extraction

**Script:** `scripts/timestamps_using_yt_subtitles.py`

Downloads YouTube auto-captions and saves JSON transcripts containing text, segments, and timestamps.

### ▶️ Run

```bash
python scripts/timestamps_using_yt_subtitles.py     --url https://www.youtube.com/watch?v=VIDEO_ID     --out_dir results/timestamps_using_yt_subtitles
```

---

## 🧩 2. Local MP4 Transcription (Whisper)

**Script:** `scripts/timestamps_using_whisper.py`

Extracts transcript + timestamps from `.mp4` videos using Whisper.

### ▶️ Run on a Single Video

```bash
python scripts/timestamps_using_whisper.py     --file path/to/video.mp4     --out_dir results/timestamps_using_whisper
```

### ▶️ Batch Mode (HPC)

```bash
sbatch scripts/run_whisper_all_gpu.slurm
```

---

## 🧩 3. Speech vs No-Speech Classification

**Script:** `scripts/split_speech_vs_nospeech.py`

Classifies transcripts based on text length.

### ▶️ Run

```bash
python scripts/split_speech_vs_nospeech.py     --transcripts_dir results/timestamps_using_whisper
```

Outputs:

- `results/speech_vs_nospeech_videos/`

---

## 🧩 4. Baseline Task Alignment (Optional)

**Script:** `scripts/align_tasks_with_timestamp.py`

This was the original timestamp alignment method—kept for reference.

Outputs:

- `results/align_tasks_with_timestamp_transcript/`

---

## 🧩 5. LLM-Based Task Extraction & Timestamp Assignment

### Main Scripts

| Purpose | File |
|--------|------|
| Primary LLM task extraction pipeline | **`scripts/tasks_with_timestamps.py`** |
| Prompt template | **`scripts/prompt_for_tasks_with_timestamps.txt`** |
| (Optional) Python approximate alignment version | `scripts/tasks_with_timestamps_using_pyseqmatcher.py` |
| (Optional) Prompt for SeqMatcher version | `scripts/prompt_for_using_pyseqmatcher.txt` |

---

### How It Works

This pipeline uses a modern LLM (Qwen, S1, etc.) to:

1. Read the full transcript.  
2. Determine whether the content is relevant.  
3. Generate grounded **MAINTASK** and **SUBTASK** blocks.  
4. Assign **Whisper segment IDs** (`SEGMENT_IDS: [id1, id2, ...]`) for each subtask.  
5. Convert segment IDs into Whisper timestamps for downstream frame extraction.

These timestamps are then used to extract representative frames and build HTML/DOCX visual summaries.

---

### ⚠️ Limitations of S1 + Python SeqMatcher

#### ❌ Why S1 Fails on Long Videos

S1 has a **4096-token context limit**.

Your prompt typically includes:

- A long instruction block  
- 30–200 Whisper segments  
- The full transcript  

Many videos exceed this limit. When the prompt length is greater than S1's maximum context, vLLM raises an error like:

```text
decoder prompt length longer than max_model_len=4096
```

When this happens:

- S1 returns no output.  
- The script sets `relevant = false`.  
- This **does not** mean the video is truly irrelevant — it only means S1 ran out of context.

#### ❌ Why Python `SequenceMatcher` Is Not Ideal

The optional `tasks_with_timestamps_using_pyseqmatcher.py` script uses Python's `difflib.SequenceMatcher` to align LLM text back to the transcript. This approach:

- Produces only **approximate** timestamps.  
- Degrades on long transcripts.  
- Gets confused by repeated phrases.  
- Struggles when the LLM paraphrases.  
- Was never designed for robust timestamp alignment.

It can work, but it is **not** as reliable or accurate as directly predicting segment IDs.

---

### ⭐ Recommended Method: Qwen with Explicit `SEGMENT_IDS`

Modern models like **Qwen 2.5 (7B/14B)** support **32k–128k token contexts** and follow structured instructions very well.

The recommended approach:

1. The LLM reads:  
   - Full transcript  
   - Full Whisper segment list  
   - Detailed task extraction instructions  

2. The LLM outputs for each subtask:

   ```text
   SUBTASK: <description>
   SEGMENT_IDS: [3, 4]
   ```

This directly ties each subtask to Whisper segment IDs, yielding **high-quality, Whisper-aligned timestamps** and removing the need for Python heuristic alignment.

#### Benefits

- Greatly improved alignment accuracy.  
- Works for long videos.  
- No timestamp matching hacks.  
- Produces better frames for HTML/DOCX reports.

---

### ▶️ Run

Basic usage with Qwen 2.5:

```bash
python scripts/tasks_with_timestamps.py   --model Qwen/Qwen2.5-7B-Instruct   --prompt_file scripts/prompt_for_tasks_with_timestamps.txt   --input_dir /path/to/json/transcripts   --output_dir /path/to/output_tasks
```


## 🧩 6. Frame Extraction 

**Script:** `scripts/extract_frames_from_timestamps.py`

Given:

- JSON with tasks/subtasks/timestamps
- Original videos

For each subtask:

1. Pick a representative timestamp (usually mid of `[start, end]`)
2. Call `ffmpeg` to extract a single frame
3. Save image as:

```
results/frame_extractions/<video_id>/taskXX_subYY.jpg
```

### ▶️ Run

```bash
python scripts/extract_frames_from_timestamps.py     --json_dir results/tasks_with_timestamps     --video_dir data/Videos_with_speech     --out_dir results/frame_extractions
```

---

## 🧩 7. HTML & DOCX Reports 

Utilities can generate:

### 📄 HTML

- Includes task → subtask → timestamp  
- Inline base64 images  
- Perfect for visual inspection  

### 📄 DOCX

- Table format:
  - Task (merged cells)
  - Subtask description
  - Timestamp
  - Embedded image thumbnails

Outputs:

```
document/<video_id>.html
document/<video_id>.docx
```

---


## 📜 Script Summary

| Script | Purpose |
|--------|---------|
| `timestamps_using_yt_subtitles.py` | Download YouTube auto-captions |
| `timestamps_using_whisper.py` | Local Whisper ASR |
| `split_speech_vs_nospeech.py` | Speech/no-speech classification |
| `align_tasks_with_timestamp.py` | Legacy baseline alignment |
| `tasks_with_timestamps_using_pyseqmatcher.py` | LLM task extraction + SequenceMatcher timestamps |
| `tasks_with_timestamps.py` | LLM task extraction |
| `extract_frames_from_timestamps.py` | Frame extraction using ffmpeg |
| `prompt_for_tasks_with_timestamps.txt` | LLM prompt template |

---

## ⚠️ Known Limitations

- Python Sequence Matcher timestamps are approximate  
- S1 fails on long prompts (4096 max context)  
- Qwen is strongly recommended for segment selection  

---

## ⭐ Recommended Workflow

```
Whisper → Qwen (segment ID selection) → JSON → Frame Extraction → HTML/DOCX reports
```

This ensures:
- Reliable task extraction  
- True segment-level timestamps  
- Accurate frame selection  
- Clean visual documentation
