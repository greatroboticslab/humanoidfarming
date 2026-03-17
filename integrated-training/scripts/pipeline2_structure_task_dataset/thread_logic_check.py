import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def build_thread_input(thread: Dict[str, Any]) -> str:
    # Compact representation for LLM
    lines = []
    subs: List[Dict[str, Any]] = thread.get("subtasks", []) or []
    for i, s in enumerate(subs):
        st = s.get("start", None)
        en = s.get("end", None)
        t = s.get("text", "")
        lines.append(f"{i}. [{st}-{en}] {t}")
    return "\n".join(lines)


def extract_first_json(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction: find first {...} block.
    """
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"_parse_error": True, "_raw": text}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return {"_parse_error": True, "_raw": text, "_json_candidate": blob}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="e.g., Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--prompt_file", required=True, help="scripts/prompt_for_thread_logic.txt")
    ap.add_argument("--input_dir", required=True, help="results/subtask_threads")
    ap.add_argument("--output_dir", required=True, help="results/thread_logic")
    ap.add_argument("--max_threads_per_video", type=int, default=9999)
    ap.add_argument("--max_tokens_out", type=int, default=900)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8").strip()

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

    sp = SamplingParams(
        max_tokens=args.max_tokens_out,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        temperature=0.2,
    )

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No json files found in {in_dir}")
        return

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))
        threads = data.get("threads", []) or []
        threads = threads[: args.max_threads_per_video]

        out_threads = []
        for ti, thread in enumerate(threads):
            thread_text = build_thread_input(thread)

            prompt = (
                prompt_template
                + "\n\nSUBTASKS_ORDERED_BY_TIME:\n"
                + thread_text
                + "\n\n<|im_end|>\n<|im_start|>assistant\n"
            )

            outputs = model.generate(prompt, sampling_params=sp)
            raw = outputs[0].outputs[0].text.strip()

            parsed = extract_first_json(raw)
            out_threads.append({
                "thread_index": ti,
                "time_start": thread.get("time_start"),
                "time_end": thread.get("time_end"),
                "subtasks": thread.get("subtasks", []),
                "logic": parsed,
                "_raw": raw if parsed.get("_parse_error") else None
            })

        out = {
            "index": data.get("index", p.stem),
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "num_threads": len(out_threads),
            "threads_with_logic": out_threads,
        }

        (out_dir / f"{out['index']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[OK] {p.name} -> logic_threads={len(out_threads)}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
