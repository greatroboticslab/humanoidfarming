#!/usr/bin/env python3
"""
Plot: Mission → Sub-missions → Tasks (with pass/fail from training-quality log)

UPDATE (per Hongbo):
- Task labels are displayed as 1..N (human-friendly) instead of 0..N-1.
- We still keep the original task_index as an internal ID shown in parentheses:
    "Task 4 (id=3)"
  so the plot remains a mirror of JSON while being reviewer-friendly.

Input (per mission):
  results/training_quality_log_submissions/missions/<video_id>.json

Output:
  results/mission_submissions_plots/<video_id>.png
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ----------------------------
# Text helpers
# ----------------------------
def wrap_text(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "\n".join(
        textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False)
    )


def fmt_time(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.1f}s"
    except Exception:
        return str(x)


# ----------------------------
# Drawing helpers (auto-fit)
# ----------------------------
def add_box_autofit(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    fontsize: int = 12,
    bold: bool = False,
    text_color: str = "#0B0B0B",
):
    pad_frac = 0.10

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.7,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)

    t = ax.text(
        x + w / 2, y + h / 2,
        text,
        ha="center", va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        family="DejaVu Sans",
        color=text_color,
        zorder=3,
    )

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    (x0p, y0p) = ax.transData.transform((x, y))
    (x1p, y1p) = ax.transData.transform((x + w, y + h))

    pad_x = (x1p - x0p) * pad_frac
    pad_y = (y1p - y0p) * pad_frac

    allowed_w = max(1.0, (x1p - x0p) - 2 * pad_x)
    allowed_h = max(1.0, (y1p - y0p) - 2 * pad_y)

    min_font = 8
    fs = fontsize
    while fs >= min_font:
        t.set_fontsize(fs)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        if bbox.width <= allowed_w and bbox.height <= allowed_h:
            break
        fs -= 1

    t.set_fontsize(max(min_font, fs))


def add_ortho_edge(ax, parent_center, parent_bottom_y, child_center, child_top_y, lw=1.2, color="#555555"):
    px, _ = parent_center
    cx, _ = child_center
    mid_y = (parent_bottom_y + child_top_y) / 2.0
    offset = 0.10  # stop slightly above child top edge

    ax.plot([px, px], [parent_bottom_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([px, cx], [mid_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([cx, cx], [mid_y, child_top_y + offset], color=color, linewidth=lw, zorder=1)


# ----------------------------
# Status colors
# ----------------------------
def sub_mission_border_color(final_status: str) -> str:
    if final_status == "successful_entry":
        return "#2F855A"  # green border
    if final_status == "pending_entry":
        return "#B7791F"  # orange border
    if final_status == "redo_entry":
        return "#6B46C1"  # purple border
    if final_status == "give_up_entry":
        return "#C53030"  # red border
    return "#2D3748"


def task_row_fill(status: str) -> str:
    if status == "pass":
        return "#D4EDDA"  # light green
    if status == "fail":
        return "#F8D7DA"  # light red
    return "#EDF2F7"


# ----------------------------
# Task label mapping: display 1..N
# ----------------------------
def build_task_display_map(sub_missions: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Collect all unique task_index values across the mission and map them to
    display indices 1..N in ascending order of task_index.
    """
    ids = set()
    for sm in sub_missions:
        for t in (sm.get("tasks", []) or []):
            if isinstance(t.get("task_index"), int):
                ids.add(t["task_index"])
    ordered = sorted(ids)
    return {tid: (i + 1) for i, tid in enumerate(ordered)}  # 1..N


# ----------------------------
# Render one mission
# ----------------------------
def render_one(mission_json: Path, out_dir: Path, dpi: int = 300) -> Path:
    data = json.loads(mission_json.read_text(encoding="utf-8"))
    vid = data.get("mission_id", mission_json.stem)
    title = (data.get("mission_title") or vid).strip()

    sub_missions: List[Dict[str, Any]] = data.get("sub_missions", []) or []
    if not sub_missions:
        raise ValueError("No sub_missions found in mission log JSON.")

    # Map internal task_index -> displayed Task 1..N
    display_map = build_task_display_map(sub_missions)

    # Layout sizing
    ROOT_W, ROOT_H = 9.2, 1.15
    SUB_W, SUB_H = 6.4, 1.40

    TASK_W, TASK_H = 6.0, 0.85
    TASK_GAP_Y = 0.22

    max_tasks = max(1, max(len(sm.get("tasks", []) or []) for sm in sub_missions))

    margin_x = 1.0
    gap_sub_x = 1.4
    gap_level_y = 1.4
    gap_tasks_y = 0.6

    n = len(sub_missions)
    subs_total_w = n * SUB_W + (n - 1) * gap_sub_x
    total_w = margin_x * 2 + max(ROOT_W, subs_total_w)

    tasks_stack_h = max_tasks * TASK_H + max(0, max_tasks - 1) * TASK_GAP_Y
    total_h = (
        1.1 + ROOT_H +
        gap_level_y + SUB_H +
        gap_tasks_y + tasks_stack_h +
        1.1
    )

    fig_w = max(14, total_w * 1.05)
    fig_h = max(7, total_h * 0.85)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Root box (Mission)
    root_x = (total_w - ROOT_W) / 2
    root_y = total_h - 1.0 - ROOT_H
    add_box_autofit(
        ax, root_x, root_y, ROOT_W, ROOT_H,
        "Mission:\n" + wrap_text(title, 60),
        fc="#2F6B3F", ec="#2F6B3F",
        fontsize=18, bold=True, text_color="white"
    )
    root_center = (root_x + ROOT_W / 2, root_y + ROOT_H / 2)
    root_bottom_y = root_y

    # Sub-mission row
    sub_y = root_y - gap_level_y - SUB_H
    tasks_top_y = sub_y - gap_tasks_y

    subs_total_w = n * SUB_W + (n - 1) * gap_sub_x
    sub_x0 = (total_w - subs_total_w) / 2

    for j, sm in enumerate(sub_missions):
        sx = sub_x0 + j * (SUB_W + gap_sub_x)

        sub_id = sm.get("sub_mission_id", "")
        sub_title = sm.get("sub_mission_title", "")
        t0, t1 = sm.get("time_start"), sm.get("time_end")

        final_status = sm.get("final_status", "pending_entry")
        border = sub_mission_border_color(final_status)

        sub_label = (
            "Sub-mission:\n"
            + wrap_text(sub_title, 42) + "\n"
            + wrap_text(str(sub_id), 42) + "\n"
            + f"({fmt_time(t0)}–{fmt_time(t1)})\n"
            + f"final_status: {final_status}"
        )

        add_box_autofit(
            ax, sx, sub_y, SUB_W, SUB_H,
            sub_label,
            fc="#F7FAFC", ec=border,
            fontsize=13, bold=True, text_color="#0B0B0B"
        )

        sub_center = (sx + SUB_W / 2, sub_y + SUB_H / 2)
        sub_top_y = sub_y + SUB_H

        add_ortho_edge(
            ax,
            parent_center=root_center,
            parent_bottom_y=root_bottom_y,
            child_center=sub_center,
            child_top_y=sub_top_y,
            lw=1.2
        )

        tasks = sm.get("tasks", []) or []
        # keep stable ordering by internal task_index
        tasks = sorted(tasks, key=lambda z: (z.get("task_index", 10**9)))

        if not tasks:
            tasks = [{"task_index": -1, "task_text": "(no tasks)", "status": "pass"}]

        for k, t in enumerate(tasks):
            tid = t.get("task_index", -1)
            ttext = (t.get("task_text") or "").strip()
            status = t.get("status", "pass")

            # Display index is 1..N; keep internal id for traceability
            disp = display_map.get(tid, None)
            if disp is None or tid < 0:
                header = "Task: "
            else:
                header = f"Task {disp} (id={tid}): "

            task_label = header + wrap_text(ttext, 46) + f"\nstatus: {status}"

            tx = sx + (SUB_W - TASK_W) / 2
            ty = tasks_top_y - (k + 1) * TASK_H - k * TASK_GAP_Y

            add_box_autofit(
                ax, tx, ty, TASK_W, TASK_H,
                task_label,
                fc=task_row_fill(status),
                ec="#2D3748",
                fontsize=11,
                bold=False,
                text_color="#0B0B0B"
            )

            task_center = (tx + TASK_W / 2, ty + TASK_H / 2)
            task_top = ty + TASK_H
            add_ortho_edge(
                ax,
                parent_center=sub_center,
                parent_bottom_y=sub_y,
                child_center=task_center,
                child_top_y=task_top,
                lw=1.0
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vid}.png"
    plt.savefig(out_path, dpi=dpi, transparent=True)  # don't bbox_inches="tight"
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/training_quality_log_submissions/missions")
    ap.add_argument("--out_dir", required=True, help="results/mission_submissions_plots")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No mission JSON files found in {in_dir}")
        return

    for p in files:
        try:
            out = render_one(p, out_dir, dpi=args.dpi)
            print(f"[OK] {p.name} -> {out}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
