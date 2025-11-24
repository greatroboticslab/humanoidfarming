import argparse
import json
import os
from os import listdir
from os.path import isfile, join
from difflib import SequenceMatcher

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

irrelevantToken = "!!!TRANSCRIPT_IRRELEVANT:"
relevantToken = ">>>ACCEPT:"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def ExtractTask(line: str) -> str:
    """Classify a line from the model output."""
    if irrelevantToken in line:
        return irrelevantToken
    if relevantToken in line:
        return relevantToken

    stripped = line.strip()
    upper = stripped.upper()

    # Be tolerant: accept MAINTASK:/SUBTASK: anywhere in the line, any casing
    if "MAINTASK:" in upper or "SUBTASK:" in upper:
        return line

    return "null"


def IsSubtask(line: str) -> bool:
    """Return True if this line is a SUBTASK line."""
    return "SUBTASK:" in line.upper()


def TaskToMoMask(line: str) -> str:
    """Strip the MAINTASK:/SUBTASK: prefix and return the text after the first colon."""
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.strip()


def SubstituteTokens(text: str) -> str:
    """Replace placeholders in the prompt template with actual tokens."""
    return text.replace("relevantToken", relevantToken).replace("irrelevantToken", irrelevantToken)


# ---------------------------------------------------------------------
# Forbidden verb validation (DISABLED NOW)
# ---------------------------------------------------------------------

FORBIDDEN_VERBS = [
    "evaluate",
    "analyze",
    "assess",
    "compare",
    "investigate",
    "predict",
    "estimate",
    "determine",
    "infer",
    "optimize",
    "calculate",
    "measure",
    "research",
    "study",
    "extrapolate",
    "plan",
    "propose",
    "expand",
]

def contains_forbidden(text: str) -> bool:
    # Validation is disabled; rely on prompt instructions instead.
    return False


# ---------------------------------------------------------------------
# Alignment:
# 1) Align subtasks to best segment (Whisper).
# 2) Build raw task ranges from subtasks.
# 3) Turn task ranges into NON-OVERLAPPING sequential blocks.
# 4) Reassign subtasks to the task window where their midpoint falls.
# 5) Inside each task, partition [start,end] among subtasks so no duplicate ranges.
# ---------------------------------------------------------------------

def align_timestamps(tasks, segments, sim_threshold: float = 0.30):
    """
    Align tasks/subtasks to transcript segments using fuzzy matching.

    Steps:
      1) For each subtask, align to best segment (by similarity).
      2) Derive raw task ranges from subtasks.
      3) Convert raw task ranges into NON-OVERLAPPING sequential blocks.
      4) Reassign subtasks to the task whose time block their midpoint falls into.
      5) Inside each task, repartition the task's [start,end] into consecutive,
         non-overlapping subtask windows in order (no repeated timestamps).
    """
    enriched = []

    # ---------- 1) Align subtasks to best segment ----------
    for t in tasks:
        if not t:
            continue

        main_task = t[0]
        subtasks = t[1:]
        sub_entries = []

        for sub in subtasks:
            best_seg = None
            best_score = 0.0

            for seg in segments:
                score = similarity(sub, seg.get("text", ""))
                if score > best_score:
                    best_score = score
                    best_seg = seg

            if best_seg is not None and best_score >= sim_threshold:
                sub_entries.append(
                    {
                        "text": sub,
                        "start": best_seg.get("start", None),
                        "end": best_seg.get("end", None),
                        "segment_ids": [best_seg.get("id", None)] if best_seg.get("id", None) is not None else [],
                    }
                )
            else:
                # no good match – timestamps will be fixed later if possible
                sub_entries.append(
                    {
                        "text": sub,
                        "start": None,
                        "end": None,
                        "segment_ids": [],
                    }
                )

        # Fallback: fill missing timestamps from nearest neighbor in the same task
        valid_idx = [i for i, s in enumerate(sub_entries) if s["start"] is not None]
        if valid_idx:
            for i, s in enumerate(sub_entries):
                if s["start"] is None or s["end"] is None:
                    nearest = min(valid_idx, key=lambda j: abs(j - i))
                    s["start"] = sub_entries[nearest]["start"]
                    s["end"] = sub_entries[nearest]["end"]
                    s["segment_ids"] = []  # low-confidence, no specific segment
        # else: leave as None if no subtask matched at all

        # sort subtasks by start time
        sub_entries.sort(key=lambda s: float("inf") if s["start"] is None else s["start"])

        # raw task range from subtasks (before non-overlap adjustment)
        starts = [s["start"] for s in sub_entries if s["start"] is not None]
        ends = [s["end"] for s in sub_entries if s["end"] is not None]
        if starts and ends:
            raw_start = min(starts)
            raw_end = max(ends)
        else:
            raw_start = None
            raw_end = None

        enriched.append(
            {
                "task": main_task,
                "raw_start": raw_start,   # keep raw range for segmentation
                "raw_end": raw_end,
                "start": raw_start,       # will be adjusted below
                "end": raw_end,
                "subtasks": sub_entries,
            }
        )

    # ---------- 2) Enforce NON-OVERLAPPING sequential task ranges ----------
    time_tasks = [t for t in enriched if t["raw_start"] is not None and t["raw_end"] is not None]
    time_tasks.sort(key=lambda t: t["raw_start"])

    if time_tasks:
        m = len(time_tasks)
        boundaries = [0.0] * (m + 1)
        # first boundary = first task raw_start
        boundaries[0] = time_tasks[0]["raw_start"]

        # internal boundaries are midpoints between consecutive raw ranges
        for i in range(m - 1):
            cur_end = time_tasks[i]["raw_end"]
            next_start = time_tasks[i + 1]["raw_start"]
            if cur_end is None and next_start is None:
                mid = boundaries[i]
            elif cur_end is None:
                mid = next_start
            elif next_start is None:
                mid = cur_end
            else:
                mid = (cur_end + next_start) / 2.0
            boundaries[i + 1] = mid

        # last boundary = last task raw_end
        boundaries[m] = time_tasks[-1]["raw_end"]

        # assign adjusted non-overlapping ranges
        for i, t in enumerate(time_tasks):
            t["start"] = boundaries[i]
            t["end"] = boundaries[i + 1]

    # ---------- 3) Reassign subtasks to task windows by midpoint ----------
    task_intervals = []
    for i, t in enumerate(enriched):
        ts = t["start"]
        te = t["end"]
        if ts is not None and te is not None:
            task_intervals.append((i, ts, te))

    if task_intervals:
        new_subtasks_per_task = [[] for _ in enriched]
        unassigned = []

        for task_idx, task in enumerate(enriched):
            for sub in task["subtasks"]:
                if sub["start"] is None or sub["end"] is None:
                    unassigned.append((task_idx, sub))
                    continue

                mid = 0.5 * (sub["start"] + sub["end"])
                best_task = None
                best_dist = float("inf")

                for idx, ts, te in task_intervals:
                    center = 0.5 * (ts + te)
                    if ts <= mid <= te:
                        best_task = idx
                        best_dist = 0.0
                        break
                    dist = abs(mid - center)
                    if dist < best_dist:
                        best_dist = dist
                        best_task = idx

                if best_task is None:
                    best_task = task_idx

                new_subtasks_per_task[best_task].append(sub)

        # Unassigned (no original time): attach to original task and give center time
        for orig_idx, sub in unassigned:
            ts = enriched[orig_idx]["start"]
            te = enriched[orig_idx]["end"]
            if ts is not None and te is not None:
                mid = 0.5 * (ts + te)
                sub["start"] = mid
                sub["end"] = mid
            new_subtasks_per_task[orig_idx].append(sub)

        # ---------- 4) Partition each task's [start,end] among its subtasks ----------
        for i, task in enumerate(enriched):
            ts = task["start"]
            te = task["end"]
            subs = new_subtasks_per_task[i]

            if ts is not None and te is not None and subs:
                # sort by original midpoint for stable order
                for s in subs:
                    if s["start"] is None or s["end"] is None:
                        s["_mid"] = 0.5 * (ts + te)
                    else:
                        s["_mid"] = 0.5 * (s["start"] + s["end"])
                subs.sort(key=lambda s: s["_mid"])

                n = len(subs)
                # partition the task interval into n consecutive chunks
                boundaries = [ts + (te - ts) * k / n for k in range(n + 1)]

                for k, s in enumerate(subs):
                    s["start"] = boundaries[k]
                    s["end"] = boundaries[k + 1]
                    s.pop("_mid", None)
            else:
                # just sort by start if no proper task window
                subs.sort(key=lambda s: float("inf") if s["start"] is None else s["start"])

            task["subtasks"] = subs

    # ---------- 5) Finalize and clean up ----------
    enriched.sort(key=lambda t: float("inf") if t["start"] is None else t["start"])
    for t in enriched:
        t.pop("raw_start", None)
        t.pop("raw_end", None)

    return enriched


# ---------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------

def clean_title_line(first_line: str) -> str:
    """
    Clean up the model's first line of title text:
    - remove 'Title:' fragments
    - strip extra punctuation
    """
    lower = first_line.lower()
    idx = lower.find("title:")
    if idx != -1:
        first_line = first_line[:idx].strip()

    first_line = first_line.strip(" .:-\t\n\r")
    return first_line


def generate_title(model, sampling_params_title, transcript: str) -> str:
    """Generate a short, descriptive title (5–8 words) from transcript."""
    max_chars = 4000
    short_transcript = transcript[:max_chars]

    prompt = (
        "You are an assistant that generates ONE short, descriptive video title.\n"
        "Create a concise title in 5 to 8 words based on the transcript.\n"
        "Do NOT include the words 'Title' or 'Title:'.\n"
        "Do NOT include quotes or extra commentary. Only output the title text.\n\n"
        "Transcript:\n"
        f"{short_transcript}\n\n"
        "Title:"
    )

    try:
        outputs = model.generate(prompt, sampling_params=sampling_params_title)
        raw = outputs[0].outputs[0].text.strip()
        first_line = raw.splitlines()[0].strip()
        first_line = clean_title_line(first_line)
        if len(first_line) > 120:
            first_line = first_line[:117].rstrip() + "..."
        if not first_line:
            first_line = "Untitled Video"
        return first_line
    except Exception as e:
        print(f"[WARN] Title generation failed: {e}")
        return "Untitled Video"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Task extraction with timestamp alignment")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or HuggingFace repo ID for vLLM (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=12000,
        help="Max model context length (max_model_len). Increase for long transcripts.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to prompt.txt template.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/timestamps_using_whisper",
        help="Directory with JSON transcripts (Whisper-style with segments).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ocean/projects/cis240145p/byler/anusha/humanoidfarming/VideoProcessing/results/tasks_with_timestamps",
        help="Where to save enriched JSON task files.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into file list.")
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive) into file list; -1 means process all.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra debug information for the first file.",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------ Load model ------------------
    print(f"[INFO] Loading model: {args.model}")
    model = LLM(
        args.model,
        max_model_len=args.tokens,
        tensor_parallel_size=args.gpus,
        dtype="float16",
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    stop_token_ids = tok("<|im_end|>")["input_ids"]

    sampling_params_tasks = SamplingParams(
        max_tokens=1024,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
    )

    sampling_params_title = SamplingParams(
        max_tokens=32,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
    )

    # ------------------ Collect files ------------------
    all_files = [
        f for f in listdir(args.input_dir)
        if isfile(join(args.input_dir, f)) and f.lower().endswith(".json")
    ]
    all_files.sort()

    if args.end < 0 or args.end > len(all_files):
        end_idx = len(all_files)
    else:
        end_idx = args.end

    files = all_files[args.start:end_idx]

    print(f"[INFO] Found {len(files)} JSON transcripts in {args.input_dir}")
    if not files:
        print("[WARN] No files to process. Check input_dir / start / end.")
        return

    # ------------------ Load prompt template ------------------
    with open(args.prompt_file, "r", encoding="utf-8") as pf:
        prompt_template_raw = pf.read()
    prompt_template = SubstituteTokens(prompt_template_raw)

    # ------------------ Process each file ------------------
    for idx, fname in enumerate(files):
        if fname.startswith("."):
            continue

        json_path = join(args.input_dir, fname)
        video_id = os.path.splitext(fname)[0]

        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
        except Exception as e:
            print(f"[ERROR] Failed to read {json_path}: {e}")
            continue

        segments = data.get("segments", [])
        if not segments:
            print(f"[WARN] No segments in {fname}, skipping.")
            continue

        transcript_lines = [seg.get("text", "") for seg in segments]
        transcript = "\n".join(transcript_lines)

        # approximate video length in minutes (for prompt hint)
        video_seconds = segments[-1].get("start", 0.0)
        video_minutes = video_seconds / 60.0

        # ---------- Generate title ----------
        title = generate_title(model, sampling_params_title, transcript)

        url = data.get("url", f"https://www.youtube.com/watch?v={video_id}")
        category = data.get("category", "Unknown Category")

        # ---------- Build prompt for tasks ----------
        meta_hint = (
            f"\n\n[NOTE FOR ASSISTANT: This video is approximately {video_minutes:.1f} minutes long. "
            "You should extract one or more MAINTASK groups that cover different topics or phases in the transcript. "
            "Each MAINTASK should have 3–7 concrete SUBTASK lines grounded in the transcript.]\n\n"
        )

        prompt = prompt_template + meta_hint + transcript + "\n"
        prompt += "<|im_end|>\n<|im_start|>assistant\nFinal Answer:\n"

        if args.debug and idx == 0:
            print("=" * 60)
            print(f"[DEBUG] First prompt for {fname} (truncated):")
            print(prompt[:1000])
            print("=" * 60)

        try:
            outputs = model.generate(prompt, sampling_params=sampling_params_tasks)
        except Exception as e:
            print(f"[ERROR] LLM generation failed for {fname}: {e}")
            continue

        text_out = outputs[0].outputs[0].text
        if args.debug and idx == 0:
            print(f"[DEBUG] Model output (truncated) for {fname}:")
            print(text_out[:1000])
            print("=" * 60)

        lines = text_out.splitlines()

        tasks = []
        curTask = -1
        relevant = True
        reason = "Relevant transcript."
        rejectionReason = "Marked as irrelevant."

        # ---------- Parse model output ----------
        for line in lines:
            t = ExtractTask(line)

            if t == "null":
                continue

            if t == irrelevantToken:
                relevant = False
                rejectionReason = line.replace(irrelevantToken, "").strip()
                break

            if t == relevantToken:
                reason = line.replace(relevantToken, "").strip()
                continue

            motion = TaskToMoMask(line)
            if IsSubtask(line):
                if curTask >= 0 and curTask < len(tasks):
                    tasks[curTask].append(motion)
            else:
                tasks.append([motion])
                curTask += 1

        if not tasks:
            aligned_tasks = []
        else:
            aligned_tasks = align_timestamps(tasks, segments)

        # If model didn't follow format at all or no subtasks, mark irrelevant
        if relevant and (not aligned_tasks or all(len(t.get("subtasks", [])) == 0 for t in aligned_tasks)):
            relevant = False
            rejectionReason = "No MAINTASK/SUBTASK lines were found in the model output."

        result = {
            "index": video_id,
            "title": title,
            "url": url,
            "category": category,
            "relevant": relevant,
            "reason": reason if relevant else rejectionReason,
            "tasks": aligned_tasks,
        }

        out_path = join(args.output_dir, f"{video_id}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as of:
                json.dump(result, of, indent=2)
        except Exception as e:
            print(f"[ERROR] Failed to write {out_path}: {e}")
            continue

        print(f"[OK] {fname} -> {out_path} (relevant={relevant})")

    print("[INFO] Done processing all files.")


if __name__ == "__main__":
    main()
