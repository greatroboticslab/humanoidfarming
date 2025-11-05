## 🎬 Timestamp Extraction from YouTube Videos

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

## 🧩 align_subtasks.py

This script links each **task** and **subtask** from  
`s1_baseline/output/tasks/*.json` to timestamps in YouTube **transcripts**.

---

## 📂 Folders
```
s1_baseline/output/tasks/                     → input task JSONs  
Transcript_with_timestamp/transcripts/        → transcript JSONs  
Transcript_with_timestamp/tasks_with_timestamps/ → output (auto-created)
```

---

## ⚙️ Setup
```bash
pip install rapidfuzz unidecode
```

---

## ▶️ Run
```bash
python align_subtasks.py
```

---

## 🧠 What it does
- Reads each video ID from the task file  
- Finds its transcript JSON  
- Fuzzy-matches **subtasks** and **tasks** to captions  
- Adds start/end times and saves a new JSON

---

## ✅ Example Output
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
