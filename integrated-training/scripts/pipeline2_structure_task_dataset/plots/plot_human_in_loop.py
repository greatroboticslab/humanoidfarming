#!/usr/bin/env python3
"""
Human-in-the-Loop Process Plot (colored + text-fit)

Fixes:
- Long text (video title / AI repair) is wrapped and auto-shrunk to fit inside boxes.
- Boxes are slightly taller to reduce overflow.
- Output is a clean, publication-style process diagram.

Visualizes:
Flagged Thread/Block
    → Detected Issue
    → AI Suggested Repair
    → Human Review
    → Re-validation
    → Blueprint Updated

Reads:
    results/thread_logic/*.json
    results/check_reports/*_logical_check.json

Writes:
    results/human_in_loop_plots/<video_id>.png
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Optional

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
    fc: str = "#E2E8F0",
    ec: str = "#2D3748",
    fontsize: int = 14,
    bold: bool = False,
    text_color: str = "#0B0B0B",
):
    """
    Draw rounded rectangle and shrink font until text fits.
    NOTE: avoid bbox_inches='tight' on save to keep transforms stable.
    """
    pad_frac = 0.10

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
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


def add_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#555555"),
        zorder=1,
    )


# ----------------------------
# Renderer
# ----------------------------
def render(video_id: str, thread_logic_path: Path, check_report_path: Path, out_dir: Path):
    tl = json.loads(thread_logic_path.read_text(encoding="utf-8"))
    report = json.loads(check_report_path.read_text(encoding="utf-8"))

    title = (tl.get("title") or video_id).strip()

    issues = report.get("issues", []) or []
    if not issues:
        print(f"[INFO] No issues for {video_id}, skipping human-in-loop plot.")
        return None

    # pick first flagged issue (example for paper)
    issue = issues[0]
    issue_type = issue.get("type", "unknown")
    thread_index = issue.get("thread_index", None)

    # extract AI repair suggestion (best-effort)
    ai_repair_text = "No repair suggestion found."
    if thread_index is not None:
        for th in (tl.get("threads_with_logic", []) or []):
            if th.get("thread_index") == thread_index:
                repair = ((th.get("logic") or {}).get("repair") or {})
                bridges = repair.get("bridging_sentences", []) or []
                if bridges:
                    ai_repair_text = (bridges[0].get("text") or "").strip() or ai_repair_text
                break

    # Figure canvas (data coords)
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Semantic stage colors
    stage_colors = [
        "#F8D7DA",  # flagged (red)
        "#FFE5B4",  # detected issue (orange)
        "#D6EAF8",  # AI suggested repair (blue)
        "#FFF3CD",  # human review (yellow)
        "#E8DAEF",  # re-validation (purple)
        "#D4EDDA",  # blueprint updated (green)
    ]

    # Layout
    x = 1.1
    box_w = 7.8
    box_h = 1.05  # taller to reduce overflow
    gap = 0.35
    y_top = 8.6

    # Wrapped texts (keep words from JSON, just line-break them)
    step0 = f"Flagged Thread/Block\nVideo: {wrap_text(title, 55)}"
    step1 = f"Detected Issue:\n{wrap_text(str(issue_type), 55)}"
    step2 = f"AI Suggested Repair:\n{wrap_text(ai_repair_text, 70)}"
    step3 = "Human Review:\nAccept / Modify / Reject"
    step4 = "Re-validation:\nLogical + Coherence Check"
    step5 = "Blueprint Updated"

    steps = [step0, step1, step2, step3, step4, step5]

    # Draw boxes + arrows
    ys = []
    y = y_top
    for i, txt in enumerate(steps):
        ys.append(y)
        add_box_autofit(
            ax,
            x, y, box_w, box_h,
            txt,
            fc=stage_colors[i],
            ec="#2D3748",
            fontsize=15 if i == 0 else 14,
            bold=(i == 0),
        )
        y -= (box_h + gap)

    for i in range(len(steps) - 1):
        # arrow from bottom center of box i to top center of box i+1
        x_mid = x + box_w / 2
        y1 = ys[i]
        y2 = ys[i + 1]
        add_arrow(ax, x_mid, y1, x_mid, y2 + box_h)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.png"
    plt.savefig(out_path, dpi=300, transparent=True)  # don't use bbox_inches="tight"
    plt.close(fig)

    print(f"[OK] Human-in-loop plot saved: {out_path}")
    return out_path


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thread_logic_dir", required=True)
    ap.add_argument("--check_reports_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    tl_dir = Path(args.thread_logic_dir)
    cr_dir = Path(args.check_reports_dir)
    out_dir = Path(args.out_dir)

    tl_files = sorted([p for p in tl_dir.glob("*.json") if p.is_file()])
    if not tl_files:
        print(f"[WARN] No thread_logic JSON files found in {tl_dir}")
        return

    for tl_path in tl_files:
        vid = tl_path.stem
        cr_path = cr_dir / f"{vid}_logical_check.json"
        if cr_path.exists():
            try:
                render(vid, tl_path, cr_path, out_dir)
            except Exception as e:
                print(f"[ERROR] {vid}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
