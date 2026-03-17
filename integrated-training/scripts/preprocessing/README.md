# Preprocessing Module — Task & Timestamp Extraction

This module converts raw instructional videos or transcripts into a structured intermediate representation:

tasks_with_timestamps/<video_id>.json

This output is the foundation for both pipelines:

- Pipeline 1 → Robot Guidance Generation  
- Pipeline 2 → Structured Task Dataset  

---

# Purpose

The preprocessing stage transforms unstructured video content into:

- tasks  
- subtasks  
- timestamp intervals  
- segment mappings  

---

# Workflow

Video / Transcript
        ↓
Transcript Extraction
        ↓
Speech Filtering
        ↓
Task Extraction (LLM)
        ↓
Timestamp Alignment
        ↓
tasks_with_timestamps.json

---

# Key Scripts

## Transcript Extraction

- timestamps_using_yt_subtitles.py → Extract transcripts from YouTube  
- timestamps_using_whisper.py → Transcribe local videos using Whisper  

## Speech Filtering

- split_speech_vs_nospeech.py → Filters non-instructional videos  

## Task Extraction

- tasks_with_timestamps.py → Extract tasks & subtasks using LLM  
- tasks_with_timestamps_using_pyseqmatcher.py → Legacy method  

## Timestamp Alignment

- align_tasks_with_timestamp.py → Align tasks with timestamps  

---

# Output

results/tasks_with_timestamps/<video_id>.json

---

# Summary

This module converts raw videos into structured, timestamp-aligned tasks that serve as the foundation for downstream robotic reasoning and guidance systems.
