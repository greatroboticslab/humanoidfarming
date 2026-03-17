#!/usr/bin/env python3
"""
Regroup subtasks into coherent, logical blocks across multiple tasks/subtasks.

Reads:  results/categorized_threads/*.json
Writes: results/coherent_blocks/*.json

Update (long-horizon support):
- Uses LLM-provided logical_links to connect related subtasks across non-adjacent time.
- Lifts subtask-level links -> category_group "unit" links.
- Builds blocks as connected components in this link graph (goal/dependency-aware).
- Applies safety constraints (max units, max time span, conflict-splitting, optional boundary refinement).

You can still run pure local adjacency mode if you want (--use_links false).
Optional LLM boundary judge exists but is off by default.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


# -----------------------------
# Helpers: text + overlap
# -----------------------------
def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", norm(s)))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class GroupUnit:
    # Atomic unit (category_group) expanded with actual subtasks
    unit_id: str                   # e.g., "th3_g2"
    thread_index: int
    group_id: int
    category: str
    time_start: Optional[float]
    time_end: Optional[float]
    subtasks: List[Dict[str, Any]]  # actual subtask dicts
    has_conflicts: bool
    needs_repair: bool
    avg_gap_seconds: float


# -----------------------------
# Build units from categorized_threads JSON
# -----------------------------
def build_units(video: Dict[str, Any]) -> List[GroupUnit]:
    units: List[GroupUnit] = []

    for th in (video.get("threads") or []):
        tix = int(th.get("thread_index", -1))
        logic = th.get("logic", {}) or {}
        conflicts = bool((logic.get("conflicts") or []))
        repair = logic.get("repair", {}) or {}
        needs_repair = bool((repair.get("bridging_sentences") or []) or (repair.get("revised_subtasks") or []))

        metrics = th.get("metrics", {}) or {}
        avg_gap = float(metrics.get("avg_gap_seconds", 0.0) or 0.0)

        # Map (task_index, sub_index) -> subtask dict
        sub_lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for s in (th.get("subtasks") or []):
            try:
                key = (int(s.get("task_index")), int(s.get("sub_index")))
            except Exception:
                continue
            sub_lookup[key] = s

        for g in (th.get("category_groups") or []):
            gid = int(g.get("group_id", -1))
            cat = g.get("category", "narration")
            refs = g.get("subtask_refs") or []

            sub_list: List[Dict[str, Any]] = []
            for r in refs:
                try:
                    key = (int(r.get("task_index")), int(r.get("sub_index")))
                    if key in sub_lookup:
                        sub_list.append(sub_lookup[key])
                except Exception:
                    pass

            # Defensive time bounds from subtasks if group missing
            ts = g.get("time_start")
            te = g.get("time_end")
            starts = [s.get("start") for s in sub_list if is_num(s.get("start"))]
            ends = [s.get("end") for s in sub_list if is_num(s.get("end"))]
            if ts is None and starts:
                ts = min(starts)
            if te is None and ends:
                te = max(ends)

            units.append(GroupUnit(
                unit_id=f"th{tix}_g{gid}",
                thread_index=tix,
                group_id=gid,
                category=cat,
                time_start=float(ts) if is_num(ts) else None,
                time_end=float(te) if is_num(te) else None,
                subtasks=sub_list,
                has_conflicts=conflicts,
                needs_repair=needs_repair,
                avg_gap_seconds=avg_gap,
            ))

    # Sort by time_start when available
    units.sort(key=lambda u: u.time_start if is_num(u.time_start) else 1e18)
    return units


# -----------------------------
# Boundary features
# -----------------------------
def boundary_gap(a: GroupUnit, b: GroupUnit) -> Optional[float]:
    if is_num(a.time_end) and is_num(b.time_start):
        return float(b.time_start) - float(a.time_end)
    return None


def edge_text(a: GroupUnit, b: GroupUnit, edge_k: int = 2) -> Tuple[str, str]:
    a_texts = [s.get("text", "") for s in (a.subtasks or [])][-edge_k:]
    b_texts = [s.get("text", "") for s in (b.subtasks or [])][:edge_k]
    return "\n".join(a_texts).strip(), "\n".join(b_texts).strip()


def category_compatible(a_cat: str, b_cat: str) -> bool:
    if a_cat == b_cat:
        return True
    if {a_cat, b_cat} == {"motion", "perception"}:
        return True
    if a_cat == "planning" and b_cat in ("motion", "perception", "planning"):
        return True
    if b_cat == "planning" and a_cat in ("motion", "perception", "planning"):
        return True
    return False


# -----------------------------
# Deterministic boundary decision (used as refinement / fallback)
# -----------------------------
def deterministic_decision(
    a: GroupUnit,
    b: GroupUnit,
    max_gap_seconds: float,
    min_overlap: float,
    narration_block_merge_overlap: float,
    force_split_on_conflict: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    gap = boundary_gap(a, b)
    left_text, right_text = edge_text(a, b)
    overlap = jaccard(left_text, right_text)

    info = {
        "gap_seconds": gap,
        "overlap": round(overlap, 4),
        "cat_left": a.category,
        "cat_right": b.category,
        "has_conflicts_left": a.has_conflicts,
        "needs_repair_left": a.needs_repair,
        "has_conflicts_right": b.has_conflicts,
        "needs_repair_right": b.needs_repair,
    }

    if force_split_on_conflict and (a.has_conflicts or b.has_conflicts):
        return "split", {**info, "rule": "conflicts_present"}

    if gap is not None and gap > max_gap_seconds:
        return "split", {**info, "rule": "gap_too_large"}

    if not category_compatible(a.category, b.category):
        if "narration" in (a.category, b.category):
            if overlap >= narration_block_merge_overlap and (gap is None or gap <= max_gap_seconds):
                return "merge", {**info, "rule": "narration_merge_high_overlap"}
            return "split", {**info, "rule": "narration_incompatible"}
        if overlap >= (min_overlap * 2.0):
            return "uncertain", {**info, "rule": "incompatible_but_some_overlap"}
        return "split", {**info, "rule": "category_incompatible_low_overlap"}

    if overlap >= min_overlap:
        return "merge", {**info, "rule": "compatible_and_overlap_ok"}

    return "uncertain", {**info, "rule": "compatible_but_low_overlap"}


# -----------------------------
# Optional LLM boundary judge (vLLM) - only for uncertain boundaries
# -----------------------------
def maybe_load_vllm(model_name: str, gpus: int, max_model_len: int):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=gpus,
        dtype="float16",
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sp = SamplingParams(
        max_tokens=400,
        temperature=0.2,
        stop_token_ids=stop_token_ids,
    )
    return llm, sp


def extract_first_json(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"_parse_error": True, "_raw": text}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return {"_parse_error": True, "_raw": text, "_json_candidate": blob}


def build_boundary_prompt(prompt_template: str, a: GroupUnit, b: GroupUnit, det_info: Dict[str, Any]) -> str:
    left_text, right_text = edge_text(a, b)
    return (
        prompt_template.strip()
        + "\n\nLEFT_GROUP:\n"
        + f"category={a.category} time=[{a.time_start}-{a.time_end}]\n"
        + left_text
        + "\n\nRIGHT_GROUP:\n"
        + f"category={b.category} time=[{b.time_start}-{b.time_end}]\n"
        + right_text
        + "\n\nHEURISTICS:\n"
        + json.dumps(det_info, indent=2)
        + "\n\nReturn ONLY valid JSON.\n"
        + "\n<|im_end|>\n<|im_start|>assistant\n"
    )


def llm_judge_boundary(llm, sp, prompt_template: str, a: GroupUnit, b: GroupUnit, det_info: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_boundary_prompt(prompt_template, a, b, det_info)
    outputs = llm.generate(prompt, sampling_params=sp)
    raw = outputs[0].outputs[0].text.strip()
    parsed = extract_first_json(raw)
    if parsed.get("_parse_error"):
        return {"decision": "split", "reason": "parse_error", "_raw": raw}
    d = (parsed.get("decision") or "").strip().lower()
    if d not in ("merge", "split"):
        d = "split"
    return {"decision": d, **parsed}


# -----------------------------
# Long-horizon link-aware grouping (NEW)
# -----------------------------
def subkey(task_index, sub_index) -> Tuple[int, int]:
    return (int(task_index), int(sub_index))


def build_subtask_to_unit_index(units: List[GroupUnit]) -> Dict[Tuple[int, int], int]:
    m: Dict[Tuple[int, int], int] = {}
    for ui, u in enumerate(units):
        for s in (u.subtasks or []):
            ti = s.get("task_index", None)
            si = s.get("sub_index", None)
            if ti is None or si is None:
                continue
            try:
                m[subkey(ti, si)] = ui
            except Exception:
                continue
    return m


def extract_unit_edges_from_links(video: Dict[str, Any], units: List[GroupUnit]) -> List[Tuple[int, int]]:
    """
    Convert thread-local logical_links (from/to indices within thread.subtasks)
    into global edges between unit indices.
    """
    sub_to_unit = build_subtask_to_unit_index(units)
    edges = set()

    for th in (video.get("threads") or []):
        subs = th.get("subtasks") or []
        logic = th.get("logic", {}) or {}
        links = logic.get("logical_links", []) or []
        for lk in links:
            try:
                a_i = int(lk.get("from"))
                b_i = int(lk.get("to"))
                if a_i < 0 or b_i < 0 or a_i >= len(subs) or b_i >= len(subs):
                    continue
                a = subs[a_i]
                b = subs[b_i]
                ka = subkey(a.get("task_index"), a.get("sub_index"))
                kb = subkey(b.get("task_index"), b.get("sub_index"))
                if ka not in sub_to_unit or kb not in sub_to_unit:
                    continue
                ua = sub_to_unit[ka]
                ub = sub_to_unit[kb]
                if ua != ub:
                    x, y = (ua, ub) if ua < ub else (ub, ua)
                    edges.add((x, y))
            except Exception:
                continue

    return sorted(edges)


def connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    g = [[] for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)

    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = [i]
        while q:
            x = q.popleft()
            for y in g[x]:
                if not seen[y]:
                    seen[y] = True
                    q.append(y)
                    comp.append(y)
        comps.append(sorted(comp))
    return comps


def link_aware_unit_blocks(
    video: Dict[str, Any],
    units: List[GroupUnit],
    max_block_units: int,
    max_time_span_seconds: float,
) -> Tuple[List[List[GroupUnit]], Dict[str, Any]]:
    """
    Build unit blocks using logical-links graph connected components,
    then enforce size/time-span constraints by splitting in time order.
    """
    edges = extract_unit_edges_from_links(video, units)
    dbg = {
        "num_units": len(units),
        "num_unit_edges": len(edges),
    }

    if not edges:
        # no link structure; each unit alone (then refinement may merge locally)
        return [[u] for u in units], dbg

    comps = connected_components(len(units), edges)

    raw_blocks: List[List[GroupUnit]] = []
    for comp in comps:
        bus = [units[i] for i in comp]
        bus.sort(key=lambda u: u.time_start if is_num(u.time_start) else 1e18)
        raw_blocks.append(bus)

    # Enforce caps by splitting in time order
    final_blocks: List[List[GroupUnit]] = []

    for bus in raw_blocks:
        cur: List[GroupUnit] = []
        block_start = None

        def cur_span_ok(next_u: GroupUnit) -> bool:
            nonlocal block_start
            starts = [x.time_start for x in cur if is_num(x.time_start)]
            if block_start is None:
                if is_num(next_u.time_start):
                    block_start = float(next_u.time_start)
                elif starts:
                    block_start = float(min(starts))
            if block_start is None:
                return True
            ends = [x.time_end for x in cur if is_num(x.time_end)]
            if is_num(next_u.time_end):
                ends.append(float(next_u.time_end))
            if not ends:
                return True
            span = float(max(ends)) - float(block_start)
            return span <= max_time_span_seconds

        for u in bus:
            if not cur:
                cur = [u]
                block_start = float(u.time_start) if is_num(u.time_start) else None
                continue

            if (len(cur) + 1) > max_block_units or (not cur_span_ok(u)):
                final_blocks.append(cur)
                cur = [u]
                block_start = float(u.time_start) if is_num(u.time_start) else None
            else:
                cur.append(u)

        if cur:
            final_blocks.append(cur)

    # Sort blocks globally
    final_blocks.sort(key=lambda block: min(
        [u.time_start for u in block if is_num(u.time_start)] or [1e18]
    ))

    dbg["num_components"] = len(comps)
    dbg["num_blocks_after_caps"] = len(final_blocks)
    return final_blocks, dbg


# -----------------------------
# Local adjacency refinement within a precomputed unit-block
# -----------------------------
def refine_block_locally(
    unit_block: List[GroupUnit],
    max_gap_seconds: float,
    min_overlap: float,
    narration_merge_overlap: float,
    use_llm: bool = False,
    llm=None,
    sp=None,
    boundary_prompt: str = "",
) -> Tuple[List[List[GroupUnit]], List[Dict[str, Any]]]:
    """
    Further split a link-aware unit-block using local boundary rules.
    (This keeps blocks from becoming incoherent even if links exist.)
    """
    if len(unit_block) <= 1:
        return [unit_block], []

    refined: List[List[GroupUnit]] = []
    dbg: List[Dict[str, Any]] = []

    cur: List[GroupUnit] = [unit_block[0]]

    for u in unit_block[1:]:
        prev = cur[-1]
        det_dec, det_info = deterministic_decision(
            prev, u,
            max_gap_seconds=max_gap_seconds,
            min_overlap=min_overlap,
            narration_block_merge_overlap=narration_merge_overlap,
            force_split_on_conflict=True,
        )

        final_dec = det_dec
        final_reason = det_info.get("rule")

        if det_dec == "uncertain" and use_llm and llm is not None:
            judged = llm_judge_boundary(llm, sp, boundary_prompt, prev, u, det_info)
            final_dec = judged.get("decision", "split")
            final_reason = judged.get("reason", "llm_judge")
            det_info["llm"] = judged

        dbg.append({
            "left_unit": prev.unit_id,
            "right_unit": u.unit_id,
            "decision": final_dec,
            "reason": final_reason,
            "details": det_info,
        })

        if final_dec == "merge":
            cur.append(u)
        else:
            refined.append(cur)
            cur = [u]

    if cur:
        refined.append(cur)

    return refined, dbg


# -----------------------------
# Build output block JSON from unit list
# -----------------------------
def build_block_json(unit_list: List[GroupUnit], block_id: int) -> Dict[str, Any]:
    refs = []
    subtexts = []
    t0s = [u.time_start for u in unit_list if is_num(u.time_start)]
    t1s = [u.time_end for u in unit_list if is_num(u.time_end)]

    cat_counts: Dict[str, int] = {}
    for u in unit_list:
        cat_counts[u.category] = cat_counts.get(u.category, 0) + 1
        for s in (u.subtasks or []):
            refs.append({"task_index": s.get("task_index"), "sub_index": s.get("sub_index")})
            subtexts.append(s.get("text", ""))

    # dominant category (tie -> planning)
    items = sorted(cat_counts.items(), key=lambda x: (-x[1], x[0]))
    dom = items[0][0] if items else "narration"
    if len(items) >= 2 and items[0][1] == items[1][1]:
        dom = "planning"

    return {
        "block_id": block_id,
        "time_start": min(t0s) if t0s else None,
        "time_end": max(t1s) if t1s else None,
        "dominant_category": dom,
        "category_distribution": cat_counts,
        "units": [{"unit_id": u.unit_id, "thread_index": u.thread_index, "group_id": u.group_id, "category": u.category} for u in unit_list],
        "subtask_refs": refs,
        "block_preview_text": " ".join([t for t in subtexts if t])[:500],
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="results/categorized_threads")
    ap.add_argument("--output_dir", required=True, help="results/coherent_blocks")

    # Long-horizon link-aware grouping
    ap.add_argument("--use_links", type=str, default="true", help="true|false (default true)")
    ap.add_argument("--max_time_span", type=float, default=180.0, help="Max seconds per block (link-aware caps)")

    # Local refinement thresholds
    ap.add_argument("--max_gap", type=float, default=30.0)
    ap.add_argument("--min_overlap", type=float, default=0.06)
    ap.add_argument("--narration_merge_overlap", type=float, default=0.12)
    ap.add_argument("--max_block_units", type=int, default=12)

    # Optional LLM judge for uncertain local boundaries
    ap.add_argument("--use_llm", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=12000)
    ap.add_argument("--boundary_prompt_file", default="", help="Required if --use_llm")

    # Debug artifacts
    ap.add_argument("--save_debug", action="store_true")

    args = ap.parse_args()

    use_links = str(args.use_links).strip().lower() in ("1", "true", "yes", "y")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = sp = None
    boundary_prompt = ""
    if args.use_llm:
        if not args.boundary_prompt_file:
            raise ValueError("--use_llm requires --boundary_prompt_file")
        boundary_prompt = Path(args.boundary_prompt_file).read_text(encoding="utf-8")
        llm, sp = maybe_load_vllm(args.model, args.gpus, args.tokens)

    files = sorted([p for p in in_dir.glob("*.json") if p.is_file()])
    if not files:
        print(f"[WARN] No json files in {in_dir}")
        return

    for p in files:
        data = json.loads(p.read_text(encoding="utf-8"))

        units = build_units(data)

        # 1) Long-horizon block proposal
        if use_links:
            unit_blocks, link_dbg = link_aware_unit_blocks(
                data, units,
                max_block_units=args.max_block_units,
                max_time_span_seconds=args.max_time_span,
            )
        else:
            # fallback: each unit alone, then local refinement can merge/split
            unit_blocks = [[u] for u in units]
            link_dbg = {"num_units": len(units), "num_unit_edges": 0, "num_components": len(units), "num_blocks_after_caps": len(units)}

        # 2) Local refinement within each proposed block (safety)
        refined_blocks: List[List[GroupUnit]] = []
        boundary_debug_all: List[Dict[str, Any]] = []

        for ub in unit_blocks:
            ub_sorted = sorted(ub, key=lambda u: u.time_start if is_num(u.time_start) else 1e18)
            rb, dbg = refine_block_locally(
                ub_sorted,
                max_gap_seconds=args.max_gap,
                min_overlap=args.min_overlap,
                narration_merge_overlap=args.narration_merge_overlap,
                use_llm=args.use_llm,
                llm=llm,
                sp=sp,
                boundary_prompt=boundary_prompt,
            )
            refined_blocks.extend(rb)
            if dbg:
                boundary_debug_all.extend(dbg)

        # 3) Build final JSON blocks
        blocks_json = [build_block_json(b, i) for i, b in enumerate(refined_blocks)]

        out = {
            "index": data.get("index", p.stem),
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "params": {
                "use_links": use_links,
                "max_time_span": args.max_time_span,
                "max_gap": args.max_gap,
                "min_overlap": args.min_overlap,
                "narration_merge_overlap": args.narration_merge_overlap,
                "max_block_units": args.max_block_units,
                "use_llm": bool(args.use_llm),
                "model": args.model if args.use_llm else None,
            },
            "debug": {
                "link_grouping": link_dbg
            },
            "num_blocks": len(blocks_json),
            "blocks": blocks_json,
        }

        (out_dir / f"{out['index']}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        if args.save_debug:
            (out_dir / f"{out['index']}_boundary_debug.json").write_text(
                json.dumps(boundary_debug_all, indent=2), encoding="utf-8"
            )

        print(f"[OK] {p.name} -> blocks={len(blocks_json)} (use_links={use_links})")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
