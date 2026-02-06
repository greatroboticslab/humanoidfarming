
## MoMask Prompt Generation (Dataset-Oriented)

This path produces **standalone MoMask prompt datasets** suitable for large-scale motion generation.

### Purpose

Robot guidance is semantic; MoMask requires **low-level, physically plausible humanoid motion descriptions**.

This script acts as a **motion compiler**:
```
Robot reasoning steps
 → physical motion primitives
 → MoMask-ready text prompts
```

---

## Script: momask_prompt_from_guidance.py

### What it does

This script converts robot guidance + frame captions into **clean, non-repetitive, dataset-safe MoMask prompts**.

It:
- Extracts navigation / manipulation / perception steps
- Rewrites them into physically grounded humanoid motions
- Uses frame captions when actions are weak
- Enforces a consistent structure:
  - walk in → scan/look → act → walk back → neutral
- Removes abstract, cognitive, or non-physical semantics
- Deterministically rotates variants to reduce repetition
- Ensures prompts never start with passive observation

---

### Outputs

For each input video:
```
<out_dir>/
├── <video_id>_momask_prompts.txt
└── <video_id>_momask_index.json
```

- One prompt per subtask
- Index file maps prompts back to `(task_i, sub_i)`

---

### One-Line Summary

> Robot action guidance and frame captions are compiled into structured, physically grounded MoMask motion prompts, enabling modular motion grounding and dataset generation.

