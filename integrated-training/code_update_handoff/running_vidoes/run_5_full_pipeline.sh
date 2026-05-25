#!/bin/bash
set -e

NUM_VIDEOS=5

SOURCE_VIDEO_DIR="/ocean/projects/cis240145p/byler/largevideos/HumanoidRobotTrainingData/video_processing/rawvideos"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

VIDEO_DIR="$BASE_DIR/videos_5"
RESULTS="$BASE_DIR/results_5"

TIMESTAMP_DIR="$RESULTS/timestamps"
SPEECH_DIR="$RESULTS/speech_filter/speech"
TASK_DIR="$RESULTS/tasks_with_timestamps"

FRAME_DIR="$RESULTS/frame_extractions"
CAPTION_DIR="$RESULTS/frame_captions"
GUIDANCE_DIR="$RESULTS/subtask_guidance"

export FRAME_ROOT="$FRAME_DIR"
export TASK_JSON_DIR="$TASK_DIR"
export JSON_OUT_DIR="$CAPTION_DIR"
export FRAME_CAPTION_DIR="$CAPTION_DIR"
export OUTPUT_DIR="$GUIDANCE_DIR"

echo "=== FULL PIPELINE: ${NUM_VIDEOS} videos ==="

rm -rf "$VIDEO_DIR" "$RESULTS" results
mkdir -p "$VIDEO_DIR" "$RESULTS" "$SPEECH_DIR"

echo "=== Step 0: Copy videos ==="
find "$SOURCE_VIDEO_DIR" -type f -name "*.mp4" | sort | head -n "$NUM_VIDEOS" | while read -r f; do
  cp "$f" "$VIDEO_DIR/"
done
ls "$VIDEO_DIR"

echo "=== Step 1: Whisper ==="
python "$BASE_DIR/scripts/preprocessing/timestamps_using_whisper.py" \
  --input_dir "$VIDEO_DIR" \
  --out_dir "$TIMESTAMP_DIR"

echo "=== Step 2: Speech filter ==="
python - <<PY
import json, shutil
from pathlib import Path
timestamp_dir = Path("$TIMESTAMP_DIR")
speech_dir = Path("$SPEECH_DIR")
speech_dir.mkdir(parents=True, exist_ok=True)
for p in timestamp_dir.glob("*.json"):
    d = json.load(open(p))
    if d.get("text", "").strip() and len(d.get("segments", [])) > 0:
        shutil.copy(p, speech_dir / p.name)
print("Speech files:", len(list(speech_dir.glob("*.json"))))
PY

echo "=== Step 3: Task extraction ==="
python "$BASE_DIR/scripts/preprocessing/tasks_with_timestamps.py" \
  --input_dir "$SPEECH_DIR" \
  --output_dir "$TASK_DIR" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_tasks_with_timestamps.txt" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 1 \
  --tokens 12000

echo "=== Step 4: Patch frame extraction paths ==="
python - <<PY
from pathlib import Path
import re
p = Path("$BASE_DIR/scripts/pipeline1/extract_frames_from_subtasks.py")
s = p.read_text()
s = re.sub(r'JSON_DIR\s*=\s*Path\(".*?"\)', 'JSON_DIR = Path("$TASK_DIR")', s)
s = re.sub(r'VIDEO_DIR\s*=\s*Path\(".*?"\)', 'VIDEO_DIR = Path("$VIDEO_DIR")', s)
s = re.sub(r'OUTPUT_DIR\s*=\s*Path\(".*?"\)', 'OUTPUT_DIR = Path("$FRAME_DIR")', s)
p.write_text(s)
print("patched frame extraction")
PY

echo "=== Step 5: Frame extraction ==="
python "$BASE_DIR/scripts/pipeline1/extract_frames_from_subtasks.py"

echo "=== Step 6: Patch frame captions paths ==="
python - <<PY
from pathlib import Path
import re
p = Path("$BASE_DIR/scripts/pipeline1/frame_captions.py")
s = p.read_text()
if "import os" not in s:
    s = "import os\n" + s
s = re.sub(r'FRAME_ROOT\s*=\s*Path\(".*?"\)', 'FRAME_ROOT = Path(os.environ.get("FRAME_ROOT", "$FRAME_DIR"))', s)
s = re.sub(r'JSON_OUT_DIR\s*=\s*Path\(".*?"\)', 'JSON_OUT_DIR = Path(os.environ.get("JSON_OUT_DIR", "$CAPTION_DIR"))', s)
s = re.sub(r'TASK_JSON_DIR\s*=\s*Path\(".*?"\)', 'TASK_JSON_DIR = Path(os.environ.get("TASK_JSON_DIR", "$TASK_DIR"))', s)
p.write_text(s)
print("patched frame captions")
PY

echo "=== Step 7: Frame captions ==="
python "$BASE_DIR/scripts/pipeline1/frame_captions.py"

echo "=== Step 8: Patch guidance paths ==="
python - <<PY
from pathlib import Path
import re
p = Path("$BASE_DIR/scripts/pipeline1/subtask_guidance_generate.py")
s = p.read_text()
s = re.sub(
    r'DEFAULT_FRAME_CAPTION_DIR\s*=\s*".*?"',
    'DEFAULT_FRAME_CAPTION_DIR = "$CAPTION_DIR"',
    s
)
s = re.sub(
    r'OUTPUT_DIR\s*=\s*Path\(os\.environ\.get\("OUTPUT_DIR",\s*".*?"\)\)',
    'OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "$GUIDANCE_DIR"))',
    s
)
p.write_text(s)
print("patched guidance")
PY

echo "=== Step 9: Robot guidance ==="
python "$BASE_DIR/scripts/pipeline1/subtask_guidance_generate.py"

echo "=== Step 10: Export guidance ==="
mkdir -p "$RESULTS/final_guidance"
python "$BASE_DIR/scripts/pipeline1/batch_export_all_videos.py" "$GUIDANCE_DIR" "$RESULTS/final_guidance" || \
python "$BASE_DIR/scripts/pipeline1/batch_export_all_videos.py" "$GUIDANCE_DIR"

echo "=== Step 11: Build threads ==="
python "$BASE_DIR/scripts/pipeline2/build_subtask_threads.py" \
  --input_dir "$TASK_DIR" \
  --output_dir "$RESULTS/subtask_threads"

echo "=== Step 12: Thread logic ==="
python "$BASE_DIR/scripts/pipeline2/thread_logic_check.py" \
  --input_dir "$RESULTS/subtask_threads" \
  --output_dir "$RESULTS/thread_logic" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_thread_logic.txt" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpus 1 \
  --tokens 12000

echo "=== Step 13: Categorize ==="
python "$BASE_DIR/scripts/pipeline2/categorize_tasks_and_subtasks.py" \
  --input_dir "$RESULTS/thread_logic" \
  --output_dir "$RESULTS/categorized_threads"

echo "=== Step 14: Coherent blocks ==="
python "$BASE_DIR/scripts/pipeline2/regroup_subtasks_coherence.py" \
  --input_dir "$RESULTS/categorized_threads" \
  --output_dir "$RESULTS/coherent_blocks" \
  --prompt_file "$BASE_DIR/prompts/prompt_for_boundary_judge.txt"

echo "=== Step 15: Sub-missions ==="
python "$BASE_DIR/scripts/pipeline2/add_submissions_to_coherent_blocks.py" \
  --input_dir "$RESULTS/coherent_blocks" \
  --output_dir "$RESULTS/coherent_blocks_with_submissions" \
  --topk 3

echo "=== Step 16: Blueprints ==="
python "$BASE_DIR/scripts/pipeline2/generate_task_blueprints.py" \
  --input_dir "$RESULTS/thread_logic" \
  --output_dir "$RESULTS/task_blueprints"

echo "=== Step 17: Check reports ==="
python "$BASE_DIR/scripts/pipeline2/generate_check_reports.py" \
  --thread_logic_dir "$RESULTS/thread_logic" \
  --coherent_blocks_dir "$RESULTS/coherent_blocks_with_submissions" \
  --output_dir "$RESULTS/check_reports"

echo "=== Step 18: Training quality log ==="
python "$BASE_DIR/scripts/pipeline2/generate_training_quality_log_submissions.py" \
  --check_reports_dir "$RESULTS/check_reports" \
  --thread_logic_dir "$RESULTS/thread_logic" \
  --coherent_blocks_dir "$RESULTS/coherent_blocks_with_submissions" \
  --task_blueprints_dir "$RESULTS/task_blueprints" \
  --output_dir "$RESULTS/training_quality_log_submissions"

echo "DONE. Results saved in: $RESULTS"
