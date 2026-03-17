#!/usr/bin/env python3
"""
Category-hierarchy tree plot for coherent_blocks JSON.

Reads:  results/coherent_blocks/*.json
Writes: results/coherent_blocks_category_trees/<video_id>.png

Tree structure (uses ONLY words already in JSON):
  Root: title
    -> Block i  [time_start–time_end]  dominant_category
         -> category (from unit.category)
              -> unit_id  (category)

No preview text, no paraphrasing.
"""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ----------------------------
# Helpers
# ----------------------------
def wrap_text(s: str, width: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, replace_whitespace=False))


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


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
                   lw=1.2, color="#555555"):
    px, _ = parent_center
    cx, _ = child_center
    mid_y = (parent_bottom_y + child_top_y) / 2.0
    offset = 0.10
    ax.plot([px, px], [parent_bottom_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([px, cx], [mid_y, mid_y], color=color, linewidth=lw, zorder=1)
    ax.plot([cx, cx], [mid_y, child_top_y + offset], color=color, linewidth=lw, zorder=1)


# ----------------------------
# Colors (simple + readable)
# ----------------------------
def cat_style(cat: str) -> Tuple[str, str, str]:
    c = (cat or "").lower()
    if c == "motion":
        return ("#C6F6D5", "#2F855A", "#0B0B0B")
    if c == "perception":
        return ("#BEE3F8", "#2B6CB0", "#0B0B0B")
    if c == "planning":
        return ("#FEEBC8", "#B7791F", "#0B0B0B")
    # narration/default
    return ("#E2E8F0", "#4A5568", "#0B0B0B")


BLOCK_COLORS = [
    ("#1F77B4", "#1F77B4", "white"),
    ("#38B2AC", "#2C7A7B", "white"),
    ("#F6AD55", "#B86B2A", "#0B0B0B"),
    ("#9AE6B4", "#2F855A", "#0B0B0B"),
    ("#B794F4", "#6B46C1", "white"),
]


def block_style(i: int) -> Tuple[str, str, str]:
    return BLOCK_COLORS[i % len(BLOCK_COLORS)]


# ----------------------------
# Render
# ----------------------------
def render_one(json_path: Path, out_dir: Path, dpi: int = 300,
               max_units_per_category: int = 8):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    vid = data.get("index", json_path.stem)
    title = (data.get("title", "") or vid).strip()

    blocks: List[Dict] = data.get("blocks", []) or []
    blocks.sort(key=lambda b: safe_float(b.get("time_start")))

    # Pre-group units by category inside each block
    block_cats: List[Dict[str, List[Dict]]] = []
    for b in blocks:
        m: Dict[str, List[Dict]] = {}
        for u in (b.get("units") or []):
            cat = (u.get("category") or "narration")
            m.setdefault(cat, []).append(u)
        # stable order by unit_id
        for k in list(m.keys()):
            m[k].sort(key=lambda uu: str(uu.get("unit_id", "")))
        block_cats.append(m)

    if not blocks:
        blocks = [{
            "block_id": 0, "time_start": None, "time_end": None,
            "dominant_category": "narration", "units": []
        }]
        block_cats = [{"narration": [{"unit_id": "(no units)", "category": "narration"}]}]

    # Layout sizes
    ROOT_W, ROOT_H = 10.0, 1.1
    BLOCK_W, BLOCK_H = 6.2, 1.2
    CAT_W, CAT_H = 4.8, 1.0
    UNIT_W, UNIT_H = 4.8, 0.9

    MARGIN_X = 1.0
    GAP_BLOCK_X = 1.2
    GAP_Y = 1.3
    GAP_NODE_Y = 0.35
    GAP_CAT_X = 0.7
    GAP_UNIT_Y = 0.25

    # Determine max categories per block and max units per category for sizing
    max_cats = 1
    max_units_stack = 1
    for m in block_cats:
        cats_here = list(m.keys()) or ["narration"]
        max_cats = max(max_cats, len(cats_here))
        for cat in cats_here:
            n = len(m.get(cat, [])) or 1
            n_show = min(n, max_units_per_category) + (1 if n > max_units_per_category else 0)
            max_units_stack = max(max_units_stack, n_show)

    # Canvas width driven by number of blocks
    blocks_total_w = len(blocks) * BLOCK_W + (len(blocks) - 1) * GAP_BLOCK_X
    total_w = MARGIN_X * 2 + max(ROOT_W, blocks_total_w)

    # Height: root + blocks + cats + units
    units_stack_h = max_units_stack * UNIT_H + max(0, max_units_stack - 1) * GAP_UNIT_Y
    total_h = (
        1.0 + ROOT_H +
        GAP_Y + BLOCK_H +
        GAP_Y + CAT_H +
        0.7 + units_stack_h +
        1.0
    )

    fig_w = max(14, total_w * 1.05)
    fig_h = max(8, total_h * 0.85)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    # Root
    root_x = (total_w - ROOT_W) / 2
    root_y = total_h - 1.0 - ROOT_H
    add_box(ax, root_x, root_y, ROOT_W, ROOT_H,
            wrap_text(title, 80),
            fc="#2F6B3F", ec="#2F6B3F", fontsize=18, bold=True, text_color="white")
    root_center = (root_x + ROOT_W / 2, root_y + ROOT_H / 2)
    root_bottom_y = root_y

    # Block row
    block_y = root_y - GAP_Y - BLOCK_H
    cat_y = block_y - GAP_Y - CAT_H
    unit_top_y = cat_y - 0.7  # start of unit stacks

    blocks_total_w = len(blocks) * BLOCK_W + (len(blocks) - 1) * GAP_BLOCK_X
    block_x0 = (total_w - blocks_total_w) / 2

    for j, b in enumerate(blocks):
        bx = block_x0 + j * (BLOCK_W + GAP_BLOCK_X)

        bid = b.get("block_id", j)
        t0, t1 = b.get("time_start", None), b.get("time_end", None)
        dom = (b.get("dominant_category") or "narration")

        block_label = f"block_id: {bid}\n{fmt_time(t0)}–{fmt_time(t1)}\ndominant_category: {dom}"
        fc_b, ec_b, tc_b = block_style(j)
        add_box(ax, bx, block_y, BLOCK_W, BLOCK_H,
                wrap_text(block_label, 32),
                fc=fc_b, ec=ec_b, fontsize=12, bold=True, text_color=tc_b)

        block_center = (bx + BLOCK_W / 2, block_y + BLOCK_H / 2)
        add_ortho_edge(ax, root_center, root_bottom_y, block_center, block_y + BLOCK_H, lw=1.2)

        # Categories row under this block (centered)
        m = block_cats[j] if j < len(block_cats) else {}
        cats = list(m.keys()) if m else ["narration"]
        cats.sort(key=lambda x: str(x))  # stable

        cats_total_w = len(cats) * CAT_W + (len(cats) - 1) * GAP_CAT_X
        cats_x0 = bx + (BLOCK_W - cats_total_w) / 2

        for ci, cat in enumerate(cats):
            cx = cats_x0 + ci * (CAT_W + GAP_CAT_X)

            cat_label = f"category: {cat}"
            fc_c, ec_c, tc_c = cat_style(cat)
            add_box(ax, cx, cat_y, CAT_W, CAT_H,
                    wrap_text(cat_label, 22),
                    fc=fc_c, ec=ec_c, fontsize=12, bold=True, text_color=tc_c)

            cat_center = (cx + CAT_W / 2, cat_y + CAT_H / 2)
            add_ortho_edge(ax, block_center, block_y, cat_center, cat_y + CAT_H, lw=1.0)

            # Units stack under category
            units = m.get(cat, []) if m else []
            if not units:
                units = [{"unit_id": "(no units)", "category": cat}]

            show_units = units[:max_units_per_category]
            truncated = len(units) > max_units_per_category

            display_units = show_units + ([{"unit_id": f"+{len(units)-max_units_per_category} more", "category": cat}] if truncated else [])

            for ui, u in enumerate(display_units):
                uy = unit_top_y - (ui + 1) * UNIT_H - ui * GAP_UNIT_Y
                ux = cx  # align with category box width

                unit_id = u.get("unit_id", "")
                ucat = u.get("category", cat)
                unit_label = f"unit_id: {unit_id}\ncategory: {ucat}"

                fc_u, ec_u, tc_u = cat_style(ucat)
                add_box(ax, ux, uy, UNIT_W, UNIT_H,
                        wrap_text(unit_label, 26),
                        fc=fc_u, ec=ec_u, fontsize=10, bold=False, text_color=tc_u)

                unit_center = (ux + UNIT_W / 2, uy + UNIT_H / 2)
                add_ortho_edge(ax, cat_center, cat_y, unit_center, uy + UNIT_H, lw=0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{vid}.png"
    plt.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/coherent_blocks")
    ap.add_argument("--out_dir", required=True, help="results/coherent_blocks_category_trees")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_units_per_category", type=int, default=8)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"[WARN] No JSON files found in {in_dir}")
        return

    for p in files:
        try:
            out_path = render_one(p, out_dir, dpi=args.dpi, max_units_per_category=args.max_units_per_category)
            print(f"[OK] {p.name} -> {out_path}")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
