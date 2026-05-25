# Quickstart: Pipeline 2 → Pipeline 1 Integration

## Goal

Connect:

```text
Pipeline 2 structure
↓
Pipeline 1 guidance
↓
Unified training-ready dataset
```

## Commands

### 1. Build Pipeline-2-guided Pipeline-1 input

```bash
python scripts/integration/build_pipeline2_guided_pipeline1_input.py \
  --pipeline2_json results/pipeline2_structured_task_dataset/coherent_blocks_with_submissions/_1k9XR8ZFTk.json \
  --tasks_json results/tasks_with_timestamps/_1k9XR8ZFTk.json \
  --output_dir results/unified_pipeline/pipeline2_guided_tasks
```

### 2. Run frame extraction

Edit:

```python
JSON_DIR = Path("results/unified_pipeline/pipeline2_guided_tasks")
```

Then:

```bash
python scripts/pipeline1_robot_guidance_model/extract_frames_from_subtasks.py
```

### 3. Run frame captioning

```bash
python scripts/integration/frame_captions_unified.py
```

### 4. Run guidance generation

```bash
python scripts/integration/subtask_guidance_unified.py _1k9XR8ZFTk.json
```

### 5. Validate

```bash
python scripts/integration/validate_unified_pipeline_json.py \
  --input_json results/unified_pipeline/subtask_guidance/_1k9XR8ZFTk.json \
  --output_dir results/unified_pipeline/validation_reports
```

## Expected result

```text
results/unified_pipeline/subtask_guidance/_1k9XR8ZFTk.json
```

contains:

```text
mission
sub-mission
subtasks
frames
captions
guidance
```
