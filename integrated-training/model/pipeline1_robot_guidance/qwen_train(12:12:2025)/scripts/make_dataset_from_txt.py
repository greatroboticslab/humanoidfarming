import re
import json
from pathlib import Path

TXT_DIR = Path("/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/final_guidance_txt")
OUT_PATH = Path("qwen_train/data/train.jsonl")

# How to split a .txt into subtask blocks:
# We assume your exporter wrote lines like:
#   TASK 0: ...
#   SUBTASK 1: ...
#   Guidance:
#     GLOBAL_SUMMARY:
#     ...
TASK_RE = re.compile(r"^TASK\s+(\d+):\s*(.*)\s*$")
SUBTASK_RE = re.compile(r"^\s*SUBTASK\s+(\d+):\s*(.*)\s*$")
TR_RE = re.compile(r"^\s*Time Range:\s*(.*)\s*→\s*(.*)\s*$")
GUIDE_START_RE = re.compile(r"^\s*Guidance:\s*$")

def strip_indent(lines):
    # Your exporter indented guidance by 4 spaces; normalize safely
    out = []
    for ln in lines:
        out.append(re.sub(r"^\s{4}", "", ln.rstrip("\n")))
    return "\n".join(out).strip() + "\n"

def parse_txt(path: Path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(True)

    video = None
    title = None

    cur_task = None
    cur_task_time = None
    cur_sub = None
    cur_sub_time = None

    i = 0
    samples = []

    # Read VIDEO/TITLE from header if present
    for ln in lines[:20]:
        if ln.startswith("VIDEO:"):
            video = ln.split("VIDEO:", 1)[1].strip()
        if ln.startswith("TITLE:"):
            title = ln.split("TITLE:", 1)[1].strip()

    while i < len(lines):
        ln = lines[i].rstrip("\n")

        m = TASK_RE.match(ln)
        if m:
            cur_task = m.group(2).strip()
            cur_task_time = None
            i += 1
            continue

        if ln.startswith("Time Range:") and cur_task and not cur_sub:
            # task time range line
            cur_task_time = ln.split("Time Range:", 1)[1].strip()
            i += 1
            continue

        m = SUBTASK_RE.match(ln)
        if m:
            cur_sub = m.group(2).strip()
            cur_sub_time = None
            i += 1
            continue

        if ln.strip().startswith("Time Range:") and cur_sub and not cur_sub_time:
            cur_sub_time = ln.split("Time Range:", 1)[1].strip()
            i += 1
            continue

        if GUIDE_START_RE.match(ln):
            # collect guidance lines until blank line separating subtasks OR next SUBTASK/TASK/==== line
            i += 1
            g_lines = []
            while i < len(lines):
                peek = lines[i].rstrip("\n")
                if TASK_RE.match(peek) or SUBTASK_RE.match(peek) or peek.startswith("="*10):
                    break
                # stop at a totally empty line only if we already captured content and next looks like a new subtask
                g_lines.append(lines[i])
                i += 1

            guidance = strip_indent(g_lines)

            # Build prompt/response
            prompt = (
                "You are generating robot subtask guidance for a video.\n\n"
                f"VIDEO_ID: {video or path.stem}\n"
                f"TITLE: {title or ''}\n"
                f"TASK: {cur_task or ''}\n"
                f"TASK_TIME_RANGE: {cur_task_time or ''}\n"
                f"SUBTASK: {cur_sub or ''}\n"
                f"SUBTASK_TIME_RANGE: {cur_sub_time or ''}\n\n"
                "Write the guidance in this exact sectioned format:\n"
                "GLOBAL_SUMMARY:\n"
                "FRAME_BASED_OBSERVATIONS:\n"
                "INTEGRATED_SCENE_UNDERSTANDING:\n"
                "PRECONDITIONS_FOR_ROBOT:\n"
                "SUCCESS_CRITERIA:\n"
                "ORDERED_ROBOT_ACTION_STEPS:\n"
                "SUBTASK_STORY:\n"
            )

            samples.append({
                "prompt": prompt,
                "response": guidance
            })

            # reset subtask so next SUBTASK line is required
            cur_sub = None
            cur_sub_time = None
            continue

        i += 1

    return samples

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_samples = []
    for p in sorted(TXT_DIR.glob("*.txt")):
        ss = parse_txt(p)
        if ss:
            all_samples.extend(ss)

    # Shuffle deterministically (optional)
    # (Keeping order is fine too; leaving as-is)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in all_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_samples)} samples to {OUT_PATH}")

if __name__ == "__main__":
    main()
