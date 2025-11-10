# 🎬 Timestamp Extraction from YouTube and Local Videos

This workflow extracts **timestamps** and **speech segments** from both **YouTube videos** and **local MP4 files**, converting audio to structured JSON files with start/end times.

---

## 🧩 1. YouTube Caption Extraction

This step automatically extracts **timestamps** for each task video using **YouTube auto-captions**.

- Each task file in `s1_baseline/output/tasks/` contains a YouTube link.  
- The script **`fetch_captions_from_tasks.py`** reads those links and downloads only the **English subtitles** (no video files).  
- Subtitles are converted into structured JSON files stored in `data/transcripts/`, containing text segments with precise start and end times.

### 🧾 Example Output
```json
{
  "text": "Cover crop mix. What is the management goal?",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "text": "Cover crop mix."},
    {"id": 1, "start": 3.2, "end": 7.8, "text": "What is the management goal?"}
  ]
}
```

---

## 🧩 2. Local Video Transcription (Whisper)

For locally stored MP4s (e.g., from robot runs), timestamps are extracted using **OpenAI Whisper** via the **Faster-Whisper** implementation.

- Input videos → `VideoProcessing/data/Videos/`
- Transcripts → `VideoProcessing/results/`
- Each JSON contains `text`, and a list of `segments` with `start`, `end`, and `text`.

### ⚙️ Setup
```bash
# Environment setup
conda create -n whisper python=3.10
conda activate whisper
pip install faster-whisper ffmpeg-python
```

### ▶️ Run (Single Video)
```bash
python scripts/whisper_json_only.py   --file data/Videos/_1k9XR8ZFTk.mp4   --out_dir results   --model small   --device cpu   --compute_type int8
```

### ▶️ Run (Batch / Cluster)
```bash
sbatch run_whisper_all_gpu.slurm
```

---

## 🧩 3. Speech vs No-Speech Classification

Once transcripts are generated, you can automatically separate videos based on **whether they contain speech**.

### ▶️ Run
```bash
python scripts/split_speech_vs_nospeech.py
```

This creates:
- `videos_with_speech.txt`
- `videos_no_speech.txt`

and separates both videos and their corresponding JSONs into:
```
data/Videos_with_speech/
data/Videos_no_speech/
results/json_with_speech/
results/json_no_speech/
```

### ✅ Classification Rule
A video is labeled **“with speech”** if total transcribed text exceeds 30 characters.

---

## 🧩 4. Aligning Tasks and Subtasks

After timestamps are extracted (from YouTube or Whisper),  
`align_subtasks.py` links **tasks** and **subtasks** to the corresponding transcript segments.

### 📂 Folder Structure
```
s1_baseline/output/tasks/                     → input task JSONs
Transcript_with_timestamp/transcripts/        → transcript JSONs
Transcript_with_timestamp/tasks_with_timestamps/ → aligned outputs
```

### ▶️ Run
```bash
python align_tasks.py
```

### ✅ Example Output
```json
{
  "task": "Animal care and maintenance",
  "task_start": 87.2,
  "task_end": 121.5,
  "task_time_method": "subtasks_span",
  "subtasks": [
    {"text": "Clean animal bedding", "start": 110.2, "end": 111.9}
  ]
}
```

---

## 💾 Summary of Key Scripts

| Script | Purpose |
|--------|----------|
| `fetch_captions_from_tasks.py` | Fetch auto-captions from YouTube tasks |
| `whisper_json_only.py` | Extract timestamps from local MP4s using Whisper |
| `split_speech_vs_nospeech.py` | Separate videos and transcripts by speech content |
| `align_subtasks.py` | Match task/subtask descriptions to transcript timestamps |

---

## 🧠 Optional: HPC Job Examples

### Single-Video Test
```bash
sbatch run_whisper_one_test_gpu.slurm
```

### Full Batch Processing
```bash
sbatch run_whisper_all_gpu.slurm
```

### Resume Remaining Files
```bash
sbatch run_whisper_remaining_gpu.slurm
```

Each job logs progress to:
```
VideoProcessing/logs/
```
