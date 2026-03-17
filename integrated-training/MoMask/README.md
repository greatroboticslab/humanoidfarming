# MoMask Action Workflow

This module adds **motion grounding** to each subtask by converting validated robot action steps into **MoMask-compatible motion prompts**, with optional visualization of generated motions.

The design is **modular and non-blocking**: motion can be deferred, placeholders are generated when needed, and real MoMask outputs can be plugged in later without changing downstream code.

---

## Overview

### Main Pipeline

Video → transcript → task/subtask + timestamps → frame grounding → validated robot guidance

From **validated robot guidance**, the pipeline branches into **MoMask workflows**.

```
Robot Action Steps
    ↓
MoMask Motion Prompt
    ↓
BVH → Motion Frame Visualization
```

---

## 1. Input: Validated Robot Actions

Each subtask contains a `guidance_text` section with structured steps:

```
ORDERED_ROBOT_ACTION_STEPS:
1. [type=navigation] ...
2. [type=perception] ...
3. [type=manipulation] ...
```

These steps are guaranteed by earlier validation to be:
- Imperative and robot-centric
- Semantically typed (`navigation`, `manipulation`, etc.)
- Free of truncation and hallucinated tools
- Ordered and bounded in length

---

## 2. Action → MoMask Prompt Conversion

Script:
```
add_momask_to_subtaskguidance.py
```

### Selection Rules
- **Keep**: `navigation`, `manipulation`
- **Drop**: perception-only and communication-only steps
- **Filter out**:
  - Truncated or low-signal actions
  - Non-motion verbs (read, observe, verify, report)
- **Always append** a cleanup / return-to-standby action

### Fallback Behavior
If no usable motion actions remain, a neutral idle / gesture motion is generated to keep the pipeline consistent.

### Example Prompt
```
Perform these actions smoothly in sequence:
Navigate to the relevant scene;
Adjust position behind the object;
Return to a safe standby position;
then return to a stable neutral stance.
```

---

## 3. Motion Placeholder Injection

Each subtask receives a `motion` block:

```json
"motion": {
  "engine": "momask",
  "prompt": "...",
  "duration_s": 4.0,
  "fps": 30,
  "seed": 0,
  "bvh_path": "results/momask/<video>/taskXX/subYY/motion.bvh",
  "status": "pending_momask_execution"
}
```

At this stage:
- MoMask is **not executed**
- Output paths and parameters are pre-declared
- Downstream stages remain functional

---

## 4. MoMask Execution

An external MoMask runner consumes:
- `motion.prompt`
- `duration_s`, `fps`, `seed`

and produces:
```
motion.bvh
```

at the specified output path.

No downstream code changes are required.

---

## 5. BVH → Motion Frame Visualization

Script:
```
render_momask_bvh_to_images.py
```

### Behavior
- If `motion.bvh` exists → render skeleton motion frames
- If missing → generate labeled placeholder frames

Example:
```bash
python render_momask_bvh_to_images.py \
  --bvh_file results/momask/.../motion.bvh \
  --out_dir results/momask/.../frames \
  --num_frames 12 \
  --placeholder_if_missing
```

---

## Key Properties

- **Decoupled**: motion does not block text or visual grounding
- **Deterministic**: prompts are derived from validated robot actions
- **Replaceable**: MoMask can be swapped with another motion model
- **Auditable**: every motion traces back to specific robot steps

---

## One-Line Summary

Robot action steps are automatically distilled into MoMask motion prompts, enabling optional, modular motion grounding per subtask.
