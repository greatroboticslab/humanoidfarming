# 🎬 Timestamp Extraction and Task Alignment from YouTube & Local Videos

This repository provides a modular pipeline to:

1. Extract **timestamped transcripts** from  
   - **YouTube videos** (auto-captions)  
   - **local MP4 videos** (Whisper)

2. Classify videos into **speech vs no-speech**.

3. Use the **S1 LLM** to extract **tasks and subtasks** from transcripts.

4. Assign **approximate timestamps** to each subtask using Python’s  
   **`SequenceMatcher`**.

All outputs are saved inside the **`results/`** directory.

---

## 🧭 High-Level Pipeline Overview

1. **YouTube** → `timestamps_using_yt_subtitles.py` → `results/timestamps_using_yt_subtitles/`
2. **Local MP4** → `timestamps_using_whisper.py` → `results/timestamps_using_whisper/`
3. **Speech/No-Speech Split** → `split_speech_vs_nospeech.py` → `results/speech_vs_nospeech_videos/`
4. **(Optional Baseline)** → `align_tasks_with_timestamp.py` → `results/align_tasks_with_timestamp_transcript/`
5. **Main Pipeline** → `s1_with_timestamps.py` (S1 + SequenceMatcher) → `results/tasks_with_timestamps_using_pysqmatch/`

---

## 🧩 1. YouTube Transcript Extraction  
**Script:** `scripts/timestamps_using_yt_subtitles.py`

Downloads YouTube auto-captions and saves JSON transcripts containing  
`text`, `segments`, and `start/end` timestamps.

### ▶️ Run
```bash
python scripts/timestamps_using_yt_subtitles.py \
    --url https://www.youtube.com/watch?v=VIDEO_ID \
    --out_dir results/timestamps_using_yt_subtitles
```

### Example Output
```json
{
  "text": "Cover crop mix. What is the management goal?",
  "segments": [
    { "id": 0, "start": 0.0, "end": 3.2, "text": "Cover crop mix." },
    { "id": 1, "start": 3.2, "end": 7.8, "text": "What is the management goal?" }
  ],
  "language": "en"
}
```

Outputs stored in:
```
results/timestamps_using_yt_subtitles/
```

---

## 🧩 2. Local Video Transcription (Whisper)

**Scripts:**  
- `scripts/timestamps_using_whisper.py`  

Extracts transcript + timestamps from `.mp4` videos using Whisper.

### ▶️ Run on a Single Video
```bash
python scripts/timestamps_using_whisper.py \
    --file path/to/video.mp4 \
    --out_dir results/timestamps_using_whisper
```

### ▶️ Batch (HPC)
```bash
sbatch scripts/run_whisper_all_gpu.slurm
```

Outputs stored in:
```
results/whisper_transcripts/
```

---

## 🧩 3. Speech vs No-Speech Classification  
**Script:** `scripts/split_speech_vs_nospeech.py`

Classifies transcripts by text length.  
Outputs stored in:
```
results/speech_vs_nospeech_videos/
```

### ▶️ Run
```bash
python scripts/split_speech_vs_nospeech.py \
    --transcripts_dir results/whisper_transcripts
```

---

## 🧩 4. Baseline Task Alignment (Optional)
**Script:** `scripts/align_tasks_with_timestamp.py`

An older experimental script that aligns tasks to segments.  
Stored in:
```
results/align_tasks_with_timestamp_transcript/
```

---

## 🧩 5. S1 Task Extraction + Timestamp Assignment

**Script:** `scripts/s1_with_timestamps.py`  
**Prompt:** `scripts/promptusinfpysqmatcher.txt`

### How it works:
1. S1 LLM reads the transcript → decides relevance → generates tasks/subtasks.
2. Python’s `SequenceMatcher` aligns each subtask to Whisper segments.
3. Approximate timestamps are assigned.

⚠️ **Important:**  
S1 **does not** generate timestamps.  
All timing is assigned by **Python heuristics** → timestamps are approximate.

### ▶️ Run
```bash
python scripts/s1_with_timestamps.py \
    --prompt_file scripts/promptusinfpysqmatcher.txt \
    --input_dir results/whisper_transcripts \
    --output_dir results/tasks_with_timestamps_using_pysqmatch
```

Outputs stored in:
```
results/tasks_with_timestamps_using_pysqmatch/
```

---

## 📁 Folder Structure (Actual)

```
results/
├─ align_tasks_with_timestamp_transcript/
├─ speech_vs_nospeech_videos/
├─ tasks_with_timestamps_using_pysqmatch/
├─ timestamps_using_whisper/
├─ timestamps_using_yt_subtitles/
└─ whisper_transcripts/
```

---

## 📜 Script Summary Table

| Script | Purpose |
|--------|---------|
| `timestamps_using_yt_subtitles.py` | Fetch YouTube auto-captions → JSON |
| `timestamps_using_whisper.py` | Transcribe MP4 videos using Whisper |
| `whisper.py` | Whisper backend |
| `split_speech_vs_nospeech.py` | Split transcripts by speech |
| `align_tasks_with_timestamp.py` | Baseline alignment |
| `s1_with_timestamps.py` | S1 task extraction + SequenceMatcher timestamps |
| `run_whisper_all_gpu.slurm` | HPC Whisper batch |
| `promptusinfpysqmatcher.txt` | Prompt for S1 |

---

## ⚠️ Known Limitations / Notes

- SequenceMatcher produces **approximate** timestamp alignment.  
- For pixel-perfect keyframe extraction, a better approach is needed.  
- Future work: ask **S1 directly** to select segment IDs from transcript.

---

