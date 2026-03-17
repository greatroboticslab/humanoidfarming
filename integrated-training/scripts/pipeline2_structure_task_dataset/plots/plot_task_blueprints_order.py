#!/usr/bin/env python3
"""
Plot "Task Blueprint" execution order (timestamped robot to-do list) with:
- Root labeled as Mission
- "text" renamed to "Action"
- Time labeled as "Task Duration"
- Gradient-based background colors per Task (no hardcoded palette)
- Still strictly timestamp-ordered (global execution_order)

Reads : results/task_blueprints/*.json
Writes: results/task_blueprints_plots/<video_id>.png

Mirror of JSON (no paraphrasing): Mission -> Task headers -> Actions (timestamp-ordered)
"""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ----------------------------
# Text helpers
# ----------------------------
def wrap_text(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))


def fmt_time(x) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.1f}s"
    except Exception:
        return str(x)


def safe_float(x, default=1e18) -> float:
    try:
        return float(x)
    except Exception:
        return default


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# ----------------------------
# Drawing primitives (auto-fit)
# ----------------------------
def add_box(ax, x, y, w, h, text, fc, ec, fontsize=11, bold=False, text_color="#0B0B0B"):
    pad_frac = 0.10
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1.0,
        zorder=2,
    )
    ax.add_patch(box)

    t = ax.text(
        x + w / 2, y + h / 2, text,
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

    min_font = 7
    fs = fontsize
    while fs >= min_font:
        t.set_fontsize(fs)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer)
        if bbox.width <= allowed_w and bbox.height <= allowed_h:
            break
        fs -= 1
    t.set_fontsize(max(min_font, fs))


def add_ortho_edge(ax, parent_center, parent_bottom_y, child_center, child_top_y,
                   lw=1.1, color="#555555"):
    px, _ = parent_center
    cx, _ = child_center
    mid_y = (parent_bottom_y + child_top_y) / 2.0
    offset = 0.10
    ax.plot([px, px], [parent_bottom_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([px, cx], [mid_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([cx, cx], [mid_y, child_top_y + offset], color=color, linewidth=lw, zorder=1)


# ----------------------------
# Color helpers (gradient-based)
# ----------------------------
def blend_with_white(rgba: Tuple[float, float, float, float], amount: float) -> Tuple[float, float, float, float]:
    """
    amount in [0,1]: 0 => original, 1 => white
    """
    r, g, b, a = rgba
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return (r, g, b, a)


def rgba_to_hex(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def task_color_map(task_ids_in_order: List[int]):
    """
    Returns dict task_index -> base RGBA from a colormap (gradient).
    """
    cmap = plt.get_cmap("viridis")  # gradient, not hardcoded palette list
    n = max(1, len(task_ids_in_order))
    mapping = {}
    for i, tid in enumerate(task_ids_in_order):
        v = 0.5 if n == 1 else i / (n - 1)
        mapping[tid] = cmap(v)
    return mapping


# ----------------------------
# Renderer
# ----------------------------
def render_one(json_path: Path, out_dir: Path, dpi: int = 300, max_rows: int = 30):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    vid = data.get("index", json_path.stem)
    title = (data.get("title", "") or vid).strip()

    # Map task_index -> task_text
    task_text_map: Dict[int, str] = {}
    for t in (data.get("tasks") or []):
        try:
            ti = int(t.get("task_index"))
        except Exception:
            continue
        task_text_map[ti] = (t.get("task_text", "") or "").strip()

    # Global execution order (timestamp-sorted)
    steps: List[Dict[str, Any]] = data.get("execution_order", []) or []
    steps.sort(key=lambda s: safe_float(s.get("start")))

    if not steps:
        steps = [{"task_index": 0, "sub_index": 0, "start": None, "end": None, "text": "(no actions)"}]

    # Build rows: insert a Task header whenever task_index changes
    rows: List[Dict[str, Any]] = []
    seen_task_order: List[int] = []
    prev_ti = None
    for s in steps:
        ti = s.get("task_index", 0)
        if ti != prev_ti:
            if ti not in seen_task_order:
                seen_task_order.append(int(ti))
            rows.append({
                "type": "task",
                "task_index": ti,
                "task_text": task_text_map.get(int(ti), ""),
            })
            prev_ti = ti

        rows.append({
            "type": "action",
            "task_index": ti,
            "sub_index": s.get("sub_index", 0),
            "start": s.get("start", None),
            "end": s.get("end", None),
            "action": (s.get("text", "") or "").strip(),
        })

    # Cap displayed rows to keep plot readable
    truncated = len(rows) > max_rows
    show_rows = rows[:max_rows]

    # Color mapping per task (gradient)
    tcolors = task_color_map(seen_task_order)

    # Layout
    ROOT_W, ROOT_H = 11.0, 1.15
    ROW_W = 11.0
    TASK_H = 0.90
    ACTION_H = 1.05
    MARGIN_X = 1.0
    GAP_Y = 0.40

    total_w = MARGIN_X * 2 + ROOT_W
    total_h = 1.0 + ROOT_H + 0.8
    for r in show_rows:
        total_h += (TASK_H if r["type"] == "task" else ACTION_H) + GAP_Y
    if truncated:
        total_h += 0.9 + GAP_Y
    total_h += 1.0

    fig = plt.figure(figsize=(max(14, total_w * 1.05), max(8, total_h * 0.70)))
    ax = plt.gca()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Root: Mission
    root_x = (total_w - ROOT_W) / 2
    root_y = total_h - 1.0 - ROOT_H
    mission_label = f"Mission: {title}"
    add_box(ax, root_x, root_y, ROOT_W, ROOT_H, wrap_text(mission_label, 95),
            fc="#2F6B3F", ec="#2F6B3F", fontsize=18, bold=True, text_color="white")
    root_center = (root_x + ROOT_W / 2, root_y + ROOT_H / 2)
    root_bottom_y = root_y

    # Rows
    y = root_y - 0.8

    first_row_center = None
    last_parent_center = root_center
    last_parent_bottom = root_bottom_y

    for idx, r in enumerate(show_rows):
        if r["type"] == "task":
            ti = int(r.get("task_index", 0))
            base = tcolors.get(ti, plt.get_cmap("viridis")(0.5))
            fc = rgba_to_hex(blend_with_white(base, 0.15))
            ec = rgba_to_hex(blend_with_white(base, 0.00))
            txt = f"Task {ti}"
            ttext = (r.get("task_text", "") or "").strip()
            if ttext:
                txt += "\n" + wrap_text(ttext, 85)

            y -= TASK_H
            add_box(ax, MARGIN_X, y, ROW_W, TASK_H, txt,
                    fc=fc, ec=ec, fontsize=12, bold=True, text_color="#0B0B0B")

            row_center = (MARGIN_X + ROW_W / 2, y + TASK_H / 2)
            add_ortho_edge(ax, last_parent_center, last_parent_bottom, row_center, y + TASK_H, lw=1.0)
            last_parent_center = row_center
            last_parent_bottom = y

            y -= GAP_Y

        else:
            ti = int(r.get("task_index", 0))
            base = tcolors.get(ti, plt.get_cmap("viridis")(0.5))
            fc = rgba_to_hex(blend_with_white(base, 0.80))  # lighter shade for actions
            ec = rgba_to_hex(blend_with_white(base, 0.30))

            st = r.get("start", None)
            en = r.get("end", None)
            action = r.get("action", "")

            label = (
                f"task_index: {ti}   sub_index: {r.get('sub_index')}\n"
                f"Task Duration: {fmt_time(st)}–{fmt_time(en)}\n"
                f"Action: {wrap_text(action, 90)}"
            )

            y -= ACTION_H
            add_box(ax, MARGIN_X, y, ROW_W, ACTION_H, label,
                    fc=fc, ec=ec, fontsize=10, bold=False, text_color="#0B0B0B")

            row_center = (MARGIN_X + ROW_W / 2, y + ACTION_H / 2)
            add_ortho_edge(ax, last_parent_center, last_parent_bottom, row_center, y + ACTION_H, lw=0.9)

            y -= GAP_Y

    if truncated:
        label = f"+{len(rows) - max_rows} more rows"
        y -= 0.9
        add_box(ax, MARGIN_X, y, ROW_W, 0.9, label,
                fc="#CBD5E0", ec="#4A5568", fontsize=12, bold=True, text_color="#0B0B0B")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vid}.png"
    plt.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/task_blueprints")
    ap.add_argument("--out_dir", required=True, help="results/task_blueprints_plots")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_rows", type=int, default=30, help="caps task headers + action rows shown")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No JSON files found in {in_dir}")
        return

    for p in files:
        try:
            out_path = render_one(p, out_dir, dpi=args.dpi, max_rows=args.max_rows)
            print(f"[OK] {p.name} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
