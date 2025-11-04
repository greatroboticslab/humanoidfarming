## 🎬 Timestamp Extraction from YouTube Videos (`fetch_captions_from_tasks.py`)

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
