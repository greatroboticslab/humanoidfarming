#!/bin/bash
set -e

# =========================
# CONFIG
# =========================
NUM_VIDEOS=30

SOURCE_VIDEO_DIR="/ocean/projects/cis240145p/byler/largevideos/HumanoidRobotTrainingData/video_processing/rawvideos"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_DIR="$BASE_DIR/videos_sample"
RESULTS="$BASE_DIR/results"

TIMESTAMP_DIR="$RESULTS/timestamps"
SPEECH_DIR="$RESULTS/speech_filter/speech"
TASK_DIR="$RESULTS/tasks_with_timestamps"

echo "=== Preparing ${NUM_VIDEOS} videos ==="

rm -rf "$VIDEO_DIR"
mkdir -p "$VIDEO_DIR"

find "$SOURCE_VIDEO_DIR" -type f -name "*.mp4" | sort | head -n ${NUM_VIDEOS} | while read -r f; do
  cp "$f" "$VIDEO_DIR/"
done

echo "Copied videos:"
ls "$VIDEO_DIR" | wc -l

mkdir -p "$RESULTS"

# =========================
# STEP 1: WHISPER
# =========================
echo "=== Step 1: Whisper timestamps ==="
python "$BASE_DIR/scripts/preprocessing/timestamps_using_whisper.py" \
  --input_dir "$VIDEO_DIR" \
  --out_dir "$TIMESTAMP_DIR"

# =========================
# STEP 1.5: SPEECH FILTER
# =========================
echo "=== Step 1.5: Speech vs No-Speech ==="
python "$BASE_DIR/scripts/preprocessing/split_speech_vs_nospeech.py" \
  --input_dir "$TIMESTAMP_DIR" \
  --output_dir "$RESULTS/speech_filter"

echo "Speech timestamp files:"
find "$SPEECH_DIR" -name "*.json" | wc -l

# =========================
# STEP 2: TASK EXTRACTION
# =========================
echo "=== Step 2: Task extraction ==="
python "$BASE_DIR/scripts/preprocessing/tasks_with_timestamps.py" \
  --input_dir "$SPEECH_DIR" \
  --output_dir "$TASK_DIR" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_tasks_with_timestamps.txt" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 1 \
  --tokens 12000

echo "Task JSON files:"
find "$TASK_DIR" -name "*.json" | wc -l

# =========================
# PIPELINE 1
# =========================
echo "=== Step 3: Frame extraction ==="
python "$BASE_DIR/scripts/pipeline1/extract_frames_from_subtasks.py" \
  --input_dir "$TASK_DIR" \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$RESULTS/frame_extractions"

echo "=== Step 4: Frame captions ==="
python "$BASE_DIR/scripts/pipeline1/frame_captions.py" \
  --frames_dir "$RESULTS/frame_extractions" \
  --tasks_dir "$TASK_DIR" \
  --output_dir "$RESULTS/frame_captions"

echo "=== Step 5: Robot guidance ==="
python "$BASE_DIR/scripts/pipeline1/subtask_guidance_generate.py" \
  --input_dir "$RESULTS/frame_captions" \
  --output_dir "$RESULTS/subtask_guidance" \
  --model Qwen/Qwen2.5-7B-Instruct

echo "=== Step 6: Export guidance ==="
python "$BASE_DIR/scripts/pipeline1/batch_export_all_videos.py" \
  --input_dir "$RESULTS/subtask_guidance" \
  --output_dir "$RESULTS/final_guidance"

# =========================
# PIPELINE 2
# =========================
echo "=== Step 7: Threads ==="
python "$BASE_DIR/scripts/pipeline2/build_subtask_threads.py" \
  --input_dir "$TASK_DIR" \
  --output_dir "$RESULTS/subtask_threads"

echo "=== Step 8: Logic ==="
python "$BASE_DIR/scripts/pipeline2/thread_logic_check.py" \
  --input_dir "$RESULTS/subtask_threads" \
  --output_dir "$RESULTS/thread_logic" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_thread_logic.txt" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 1 \
  --tokens 12000

echo "=== Step 9: Categorize ==="
python "$BASE_DIR/scripts/pipeline2/categorize_tasks_and_subtasks.py" \
  --input_dir "$RESULTS/thread_logic" \
  --output_dir "$RESULTS/categorized_threads"

echo "=== Step 10: Coherence ==="
python "$BASE_DIR/scripts/pipeline2/regroup_subtasks_coherence.py" \
  --input_dir "$RESULTS/categorized_threads" \
  --output_dir "$RESULTS/coherent_blocks" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_boundary_judge.txt"

echo "=== Step 11: Sub-missions ==="
python "$BASE_DIR/scripts/pipeline2/add_submissions_to_coherent_blocks.py" \
  --input_dir "$RESULTS/coherent_blocks" \
  --output_dir "$RESULTS/coherent_blocks_with_submissions" \
  --topk 3

echo "=== Step 12: Blueprints ==="
python "$BASE_DIR/scripts/pipeline2/generate_task_blueprints.py" \
  --input_dir "$RESULTS/thread_logic" \
  --output_dir "$RESULTS/task_blueprints"

echo "=== Step 13: Checks ==="
python "$BASE_DIR/scripts/pipeline2/generate_check_reports.py" \
  --thread_logic_dir "$RESULTS/thread_logic" \
  --coherent_blocks_dir "$RESULTS/coherent_blocks_with_submissions" \
  --output_dir "$RESULTS/check_reports"

echo "=== Step 14: Training log ==="
python "$BASE_DIR/scripts/pipeline2/generate_training_quality_log_submissions.py" \
  --check_reports_dir "$RESULTS/check_reports" \
  --thread_logic_dir "$RESULTS/thread_logic" \
  --coherent_blocks_dir "$RESULTS/coherent_blocks_with_submissions" \
  --task_blueprints_dir "$RESULTS/task_blueprints" \
  --output_dir "$RESULTS/training_quality_log_submissions"

echo "✅ DONE: sample run completed"
