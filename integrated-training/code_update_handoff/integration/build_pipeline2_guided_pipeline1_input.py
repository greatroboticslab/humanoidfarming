#!/usr/bin/env python3
"""
build_pipeline2_guided_pipeline1_input.py

Bridge Pipeline 2 -> Pipeline 1.

It converts:
  Pipeline 2 output JSON + original tasks_with_timestamps JSON

Into:
  Pipeline 1-compatible JSON grouped by Pipeline 2 structure.

This is the first missing connection:
  Pipeline 2 structure -> Pipeline 1 execution input
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_original_subtask(tasks_data: Dict[str, Any], task_index: int, sub_index: int):
    tasks = tasks_data.get("tasks", [])
    if task_index >= len(tasks):
        raise IndexError(f"task_index {task_index} out of range")

    task = tasks[task_index]
    subtasks = task.get("subtasks", [])
    if sub_index >= len(subtasks):
        raise IndexError(f"sub_index {sub_index} out of range for task {task_index}")

    return task, subtasks[sub_index]


def build_pipeline2_guided_tasks(p2_data: Dict[str, Any], tasks_data: Dict[str, Any]) -> Dict[str, Any]:
    video_id = tasks_data.get("index") or p2_data.get("index") or p2_data.get("mission_id")
    title = tasks_data.get("title") or p2_data.get("title") or p2_data.get("mission_title")

    sub_mission_by_block = {
        sm.get("block_id"): sm
        for sm in p2_data.get("sub_missions", [])
        if "block_id" in sm
    }

    unified_tasks: List[Dict[str, Any]] = []

    for block in p2_data.get("blocks", []):
        block_id = block.get("block_id")
        sm = sub_mission_by_block.get(block_id, {})

        task_name = (
            sm.get("sub_mission_title")
            or block.get("sub_mission_title")
            or f"Sub-mission {block_id}"
        )

        new_task = {
            "task": task_name,
            "start": block.get("time_start"),
            "end": block.get("time_end"),
            "pipeline2": {
                "mission_id": p2_data.get("mission_id"),
                "mission_title": p2_data.get("mission_title"),
                "sub_mission_id": sm.get("sub_mission_id") or block.get("sub_mission_id"),
                "sub_mission_title": task_name,
                "block_id": block_id,
                "dominant_category": block.get("dominant_category"),
                "category_distribution": block.get("category_distribution", {}),
                "block_preview_text": block.get("block_preview_text"),
            },
            "subtasks": [],
        }

        for ref in block.get("subtask_refs", []):
            task_index = int(ref.get("task_index"))
            sub_index = int(ref.get("sub_index"))

            try:
                original_task, original_subtask = get_original_subtask(
                    tasks_data, task_index, sub_index
                )
            except Exception as e:
                print(f"[WARN] Could not resolve ref {ref}: {e}")
                continue

            new_subtask = {
                "text": original_subtask.get("text"),
                "start": original_subtask.get("start"),
                "end": original_subtask.get("end"),
                "segment_ids": original_subtask.get("segment_ids", []),
                "pipeline2_ref": {
                    "block_id": block_id,
                    "sub_mission_id": sm.get("sub_mission_id") or block.get("sub_mission_id"),
                    "sub_mission_title": task_name,
                    "dominant_category": block.get("dominant_category"),
                    "original_task_index": task_index,
                    "original_sub_index": sub_index,
                    "original_task_text": original_task.get("task"),
                },
            }

            new_task["subtasks"].append(new_subtask)

        unified_tasks.append(new_task)

    return {
        "index": video_id,
        "video_index": video_id,
        "title": title,
        "url": tasks_data.get("url") or p2_data.get("url"),
        "category": tasks_data.get("category"),
        "relevant": tasks_data.get("relevant", True),
        "reason": tasks_data.get("reason"),
        "integration_note": (
            "Pipeline-2-guided Pipeline-1 input. "
            "Pipeline 2 provides structure; Pipeline 1 should add frames, captions, and robot guidance."
        ),
        "mission": {
            "mission_id": p2_data.get("mission_id"),
            "mission_title": p2_data.get("mission_title") or title,
        },
        "tasks": unified_tasks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline2_json", required=True)
    parser.add_argument("--tasks_json", required=True)
    parser.add_argument("--output_dir", default="results/unified_pipeline/pipeline2_guided_tasks")
    args = parser.parse_args()

    p2_data = load_json(Path(args.pipeline2_json))
    tasks_data = load_json(Path(args.tasks_json))

    out_data = build_pipeline2_guided_tasks(p2_data, tasks_data)

    video_id = out_data.get("index") or Path(args.pipeline2_json).stem
    out_path = Path(args.output_dir) / f"{video_id}.json"
    save_json(out_data, out_path)

    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

