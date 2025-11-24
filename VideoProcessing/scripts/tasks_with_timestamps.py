import argparse
import json
import os
from os import listdir
from os.path import isfile, join

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


irrelevantToken = "!!!TRANSCRIPT_IRRELEVANT:"
relevantToken = ">>>ACCEPT:"


def SubstituteTokens(text: str) -> str:
    return text.replace("relevantToken", relevantToken).replace("irrelevantToken", irrelevantToken)


def build_segment_block(segments):
    """
    Build the human-readable segment list that we insert into the prompt.
    Format: [id] start-end: text
    """
    lines = []
    for seg in segments:
        sid = seg.get("id")
        st = seg.get("start")
        en = seg.get("end")
        txt = seg.get("text", "").replace("\n", " ")
        lines.append(f"[{sid}] {st:.2f}-{en:.2f}: {txt}")
    return "\n".join(lines)


def clean_title_line(first_line: str) -> str:
    lower = first_line.lower()
    idx = lower.find("title:")
    if idx != -1:
        first_line = first_line[:idx].strip()
    first_line = first_line.strip(" .:-\t\n\r")
    return first_line


def generate_title(model, sampling_params_title, transcript: str) -> str:
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


def parse_segment_ids(line: str):
    """
    Parse a line like: SEGMENT_IDS: [10, 11]
    Return a list of ints.
    """
    line = line.strip()
    if not line.upper().startswith("SEGMENT_IDS"):
        return []
    # split at first colon
    parts = line.split(":", 1)
    if len(parts) < 2:
        return []
    payload = parts[1].strip()
    # very simple parser for [1, 2, 3]
    if not (payload.startswith("[") and payload.endswith("]")):
        return []
    inner = payload[1:-1].strip()
    if not inner:
        return []
    ids = []
    for tok in inner.split(","):
        tok = tok.strip()
        try:
            ids.append(int(tok))
        except ValueError:
            continue
    return ids


def make_subtask_timestamps_unique(aligned_tasks):
    """
    For each task, ensure subtasks do NOT all share identical start/end
    when they refer to the same interval.

    Strategy:
    - For each task, group subtasks by (start, end).
    - If a group has k > 1 subtasks and a valid interval,
      split [start, end] into k equal sub-intervals in the
      original order of those subtasks.
    - This keeps all times inside the same base interval
      but gives distinct ranges for image extraction.
    """
    for task in aligned_tasks:
        subs = task.get("subtasks", [])
        if not subs:
            continue

        # group indices by (start, end)
        groups = {}
        for idx, s in enumerate(subs):
            st = s.get("start")
            en = s.get("end")
            key = (st, en)
            groups.setdefault(key, []).append(idx)

        for (st, en), idxs in groups.items():
            if st is None or en is None:
                continue
            if len(idxs) <= 1:
                continue

            base_start = st
            base_end = en
            length = base_end - base_start
            if length <= 0:
                # degenerate interval; leave as-is
                continue

            k = len(idxs)
            step = length / k

            # keep their original order
            idxs_sorted = sorted(idxs)
            for i, sub_idx in enumerate(idxs_sorted):
                new_start = base_start + step * i
                new_end = base_start + step * (i + 1)
                subs[sub_idx]["start"] = new_start
                subs[sub_idx]["end"] = new_end

    return aligned_tasks


def main():
    parser = argparse.ArgumentParser(description="Task extraction with LLM-based segment ID alignment")

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
        help="Path to prompt_with_segments.txt template.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with JSON transcripts (Whisper-style with segments).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
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

    # Collect files
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

    # Load prompt template
    with open(args.prompt_file, "r", encoding="utf-8") as pf:
        prompt_template_raw = pf.read()
    prompt_template = SubstituteTokens(prompt_template_raw)

    # Process each file
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

        # Title
        title = generate_title(model, sampling_params_title, transcript)

        url = data.get("url", f"https://www.youtube.com/watch?v={video_id}")
        category = data.get("category", "Unknown Category")

        # Build segment block
        segment_block = build_segment_block(segments)

        # Build prompt
        meta_hint = (
            "\n\n[NOTE FOR ASSISTANT: Use the SEGMENT IDS from the list above. "
            "For EVERY SUBTASK, you MUST output a SEGMENT_IDS line that references one or more ids from the list. "
            "Do NOT invent ids.]\n\n"
        )

        prompt = prompt_template.replace("[The calling script will insert the segments list here.]", segment_block)
        prompt = prompt + meta_hint + transcript + "\n"
        prompt += "<|im_end|>\n<|im_start|>assistant\nFinal Answer:\n"

        if args.debug and idx == 0:
            print("=" * 60)
            print(f"[DEBUG] Prompt for {fname} (truncated):")
            print(prompt[:1200])
            print("=" * 60)

        try:
            outputs = model.generate(prompt, sampling_params=sampling_params_tasks)
        except Exception as e:
            print(f"[ERROR] LLM generation failed for {fname}: {e}")
            continue

        text_out = outputs[0].outputs[0].text
        if args.debug and idx == 0:
            print(f"[DEBUG] Model output (truncated) for {fname}:")
            print(text_out[:1200])
            print("=" * 60)

        lines = text_out.splitlines()

        relevant = True
        reason = "Relevant transcript."
        rejectionReason = "Marked as irrelevant."

        tasks = []
        cur_task = None
        last_subtask = None

        # Parse output
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if irrelevantToken in stripped:
                relevant = False
                rejectionReason = stripped.replace(irrelevantToken, "").strip()
                tasks = []
                break

            if relevantToken in stripped:
                reason = stripped.replace(relevantToken, "").strip()
                continue

            upper = stripped.upper()
            if upper.startswith("MAINTASK:"):
                main_text = stripped.split(":", 1)[1].strip()
                cur_task = {
                    "task": main_text,
                    "subtasks": []
                }
                tasks.append(cur_task)
                last_subtask = None
                continue

            if upper.startswith("SUBTASK:"):
                if cur_task is None:
                    # ignore malformed
                    continue
                sub_text = stripped.split(":", 1)[1].strip()
                sub_entry = {
                    "text": sub_text,
                    "segment_ids": []
                }
                cur_task["subtasks"].append(sub_entry)
                last_subtask = sub_entry
                continue

            if upper.startswith("SEGMENT_IDS"):
                if last_subtask is None:
                    continue
                seg_ids = parse_segment_ids(stripped)
                last_subtask["segment_ids"] = seg_ids
                continue

        # Build aligned_tasks using segment_ids
        aligned_tasks = []
        if relevant and tasks:
            # Map id -> segment
            seg_map = {seg["id"]: seg for seg in segments}

            for t in tasks:
                task_text = t["task"]
                subs_in = t["subtasks"]
                subs_out = []

                starts_for_task = []
                ends_for_task = []

                for s in subs_in:
                    seg_ids = s.get("segment_ids", []) or []
                    sub_text = s.get("text", "")

                    sub_starts = []
                    sub_ends = []
                    for sid in seg_ids:
                        seg = seg_map.get(sid)
                        if seg is not None:
                            st = seg.get("start", None)
                            en = seg.get("end", None)
                            if st is not None and en is not None:
                                sub_starts.append(st)
                                sub_ends.append(en)

                    if sub_starts and sub_ends:
                        sub_start = min(sub_starts)
                        sub_end = max(sub_ends)
                        starts_for_task.append(sub_start)
                        ends_for_task.append(sub_end)
                    else:
                        sub_start = None
                        sub_end = None

                    subs_out.append(
                        {
                            "text": sub_text,
                            "start": sub_start,
                            "end": sub_end,
                            "segment_ids": seg_ids,
                        }
                    )

                if starts_for_task and ends_for_task:
                    t_start = min(starts_for_task)
                    t_end = max(ends_for_task)
                else:
                    t_start = None
                    t_end = None

                aligned_tasks.append(
                    {
                        "task": task_text,
                        "start": t_start,
                        "end": t_end,
                        "subtasks": subs_out,
                    }
                )

        # Ensure subtasks within the same interval get distinct time slices
        if aligned_tasks:
            aligned_tasks = make_subtask_timestamps_unique(aligned_tasks)

        if relevant and (not aligned_tasks or all(len(t.get("subtasks", [])) == 0 for t in aligned_tasks)):
            relevant = False
            rejectionReason = "No MAINTASK/SUBTASK lines were parsed from model output."

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
