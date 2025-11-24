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

## Main 
**Script:** `scripts/tasks_with_timestamps.py`  
**Prompt:** `scripts/prompt_for_tasks_with_timestamps.txt`

## (Optional)
**Script:** `scripts/tasks_with_timestamps_using_pyseqmatcher.py`  
**Prompt:** `scripts/prompt_for_using_pyseqmatcher.txt`


### How It Works

1. The LLM (Qwen, S1, etc.) reads the full transcript.
2. It determines relevance.
3. It generates **MAINTASK** and **SUBTASK** blocks grounded in the transcript.
4. Python’s `SequenceMatcher` aligns each subtask back to Whisper segments.
5. Approximate timestamps are assigned.

---

## ⚠️ Important: Limitations of S1 and Python Sequence Matcher

### ❌ Why S1 Fails on Long Videos

S1 has a **4096 token context window**.

Your prompt includes:

- Long instructions  
- 30–200 Whisper segments  
- Entire transcript  

Many videos exceed 4096 tokens, causing vLLM to throw:

```
decoder prompt length longer than max_model_len=4096
```

When this happens:

- S1 outputs nothing  
- The script sets `relevant = false`  
- ⚠️ This does **not** mean the video is irrelevant — it means **S1 ran out of context**.

### ❌ Why PySeqMatcher Is Not Ideal

`SequenceMatcher`:
- Produces **approximate** timestamps  
- Breaks when transcripts are long  
- Gets confused by repeated phrases  
- Cannot handle LLM paraphrasing  
- Is not designed for timestamp alignment  
- Leads to frame extraction that may be off

It “works” but is **not robust or accurate**.

---

## ⭐ Superior Method: Ask Qwen to Output Whisper `SEGMENT_IDS`

Models like Qwen 2.5 (7B/14B) handle **32k–128k contexts** and follow instructions extremely well.

Best approach:

### ✔ The LLM reads:
- Full transcript  
- Full Whisper segment list  
- Task extraction instructions  

### ✔ The LLM outputs:

```
SUBTASK: <description>
SEGMENT_IDS: [3, 4]
```

This gives **perfect, Whisper-aligned timestamps** and eliminates the Python heuristics entirely.

### Benefits

- Greatly improved alignment accuracy  
- Works for long videos  
- No timestamp matching hacks  
- Produces better frames for HTML/DOCX reports  

---

## 🧩 6. Frame Extraction (Optional)

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

## 🧩 7. HTML & DOCX Reports (Optional)

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
