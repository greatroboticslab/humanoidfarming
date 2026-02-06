#!/usr/bin/env python3
import os
import math
import glob
import argparse

def parse_bvh(path):
    joints = []
    parents = []
    offsets = []
    channels = []  # list of channel names per joint
    channel_start = []  # start index into motion vector
    motion = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    stack = []
    cur_parent = -1
    cur_joint = -1
    total_channels = 0

    def add_joint(name, parent):
        nonlocal total_channels
        joints.append(name)
        parents.append(parent)
        offsets.append([0.0, 0.0, 0.0])
        channels.append([])
        channel_start.append(total_channels)
        return len(joints) - 1

    # Parse hierarchy
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("MOTION"):
            i += 1
            break

        if ln.startswith("ROOT") or ln.startswith("JOINT"):
            name = ln.split()[1]
            cur_joint = add_joint(name, cur_parent)
            stack.append(cur_joint)
            cur_parent = cur_joint
            i += 1
            continue

        if ln.startswith("End Site"):
            # We ignore End Sites (no channels). Consume its block to keep parser aligned.
            i += 1
            # next should be "{"
            while i < len(lines) and "{" not in lines[i]:
                i += 1
            # consume until matching "}"
            depth = 0
            while i < len(lines):
                if "{" in lines[i]:
                    depth += 1
                if "}" in lines[i]:
                    depth -= 1
                    if depth <= 0:
                        i += 1
                        break
                i += 1
            continue

        if ln.startswith("OFFSET"):
            parts = ln.split()
            offsets[cur_joint] = [float(parts[1]), float(parts[2]), float(parts[3])]
            i += 1
            continue

        if ln.startswith("CHANNELS"):
            parts = ln.split()
            n = int(parts[1])
            ch = parts[2:2+n]
            channels[cur_joint] = ch
            total_channels += n
            i += 1
            continue

        if ln.startswith("}"):
            stack.pop()
            cur_parent = stack[-1] if stack else -1
            i += 1
            continue

        i += 1

    # Parse motion header
    frames = None
    frame_time = None
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("Frames:"):
            frames = int(ln.split(":")[1].strip())
        elif ln.startswith("Frame Time:"):
            frame_time = float(ln.split(":")[1].strip())
            i += 1
            break
        i += 1

    # Motion data
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            i += 1
            continue
        vals = [float(x) for x in ln.split()]
        if vals:
            motion.append(vals)
        i += 1

    if frames is not None and len(motion) != frames:
        # tolerate mismatch, but keep what we have
        pass

    return joints, parents, offsets, channels, channel_start, motion, frame_time


def rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[1,0,0],[0,ca,-sa],[0,sa,ca]]

def rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca,0,sa],[0,1,0],[-sa,0,ca]]

def rot_z(a):
    ca, sa = math.cos(a), math.sin(a)
    return [[ca,-sa,0],[sa,ca,0],[0,0,1]]

def matmul(A,B):
    return [[
        A[r][0]*B[0][c] + A[r][1]*B[1][c] + A[r][2]*B[2][c]
        for c in range(3)
    ] for r in range(3)]

def matvec(A,v):
    return [
        A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
        A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
        A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
    ]

def add(a,b): return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def fk_world_positions(parents, offsets, channels, channel_start, frame):
    """
    Returns world positions for each joint.
    Root uses position channels if present.
    Rotations are applied in the BVH channel order for that joint.
    """
    n = len(parents)
    world_pos = [[0.0,0.0,0.0] for _ in range(n)]
    world_rot = [[[1,0,0],[0,1,0],[0,0,1]] for _ in range(n)]

    for j in range(n):
        parent = parents[j]
        # local transform
        R = [[1,0,0],[0,1,0],[0,0,1]]
        t = offsets[j][:]

        # apply channels
        start = channel_start[j]
        ch = channels[j]
        # root may have position
        pos = [0.0,0.0,0.0]
        for k, name in enumerate(ch):
            v = frame[start + k]
            if name == "Xposition": pos[0] = v
            elif name == "Yposition": pos[1] = v
            elif name == "Zposition": pos[2] = v
            elif name == "Xrotation": R = matmul(R, rot_x(math.radians(v)))
            elif name == "Yrotation": R = matmul(R, rot_y(math.radians(v)))
            elif name == "Zrotation": R = matmul(R, rot_z(math.radians(v)))

        if parent == -1:
            # root
            world_rot[j] = R
            world_pos[j] = add(pos, t)
        else:
            # child
            world_rot[j] = matmul(world_rot[parent], R)
            world_pos[j] = add(world_pos[parent], matvec(world_rot[parent], t))

    return world_pos


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_placeholders(out_dir, num_frames, title="MISSING_BVH"):
    from PIL import Image, ImageDraw, ImageFont
    ensure_dir(out_dir)
    for fi in range(num_frames):
        img = Image.new("RGB", (512,512), (255,255,255))
        d = ImageDraw.Draw(img)
        d.text((20,20), title, fill=(0,0,0))
        d.text((20,50), f"frame {fi:03d}", fill=(0,0,0))
        img.save(os.path.join(out_dir, f"frame_{fi:03d}.png"))

def render_one_bvh(bvh_file, out_dir, num_frames=12, plane="xz", pad_frac=0.25, lw=6, dpi=160):
    # lazy import so placeholders still work without mpl
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joints, parents, offsets, channels, channel_start, motion, frame_time = parse_bvh(bvh_file)
    if not motion:
        raise RuntimeError(f"No motion frames in {bvh_file}")

    # sample frames evenly across the clip
    total = len(motion)
    if num_frames <= 1:
        idxs = [0]
    else:
        idxs = [round(i*(total-1)/(num_frames-1)) for i in range(num_frames)]

    # compute global bounds across selected frames for stable framing
    all_xy = []
    for idx in idxs:
        pos = fk_world_positions(parents, offsets, channels, channel_start, motion[idx])
        for p in pos:
            if plane == "xy": all_xy.append((p[0], p[1]))
            elif plane == "yz": all_xy.append((p[1], p[2]))
            else: all_xy.append((p[0], p[2]))  # xz

    xs = [p[0] for p in all_xy]
    ys = [p[1] for p in all_xy]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    w = max(1e-6, xmax - xmin)
    h = max(1e-6, ymax - ymin)
    pad = pad_frac * max(w, h)

    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    ensure_dir(out_dir)

    for out_i, idx in enumerate(idxs):
        pos = fk_world_positions(parents, offsets, channels, channel_start, motion[idx])

        fig = plt.figure(figsize=(4,4), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.axis("off")

        # draw bones
        for j in range(len(parents)):
            p = parents[j]
            if p == -1:
                continue
            a = pos[p]
            b = pos[j]
            if plane == "xy":
                ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=lw)
            elif plane == "yz":
                ax.plot([a[1], b[1]], [a[2], b[2]], linewidth=lw)
            else:  # xz
                ax.plot([a[0], b[0]], [a[2], b[2]], linewidth=lw)

        fig.suptitle(f"{os.path.basename(bvh_file)} frame {out_i:03d}", fontsize=10)
        fig.savefig(os.path.join(out_dir, f"frame_{out_i:03d}.png"), bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_file", default=None)
    ap.add_argument("--bvh_glob", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=12)
    ap.add_argument("--plane", choices=["xy","xz","yz"], default="xz", help="2D projection plane")
    ap.add_argument("--pad_frac", type=float, default=0.25, help="padding fraction around skeleton")
    ap.add_argument("--lw", type=float, default=6.0, help="line width")
    ap.add_argument("--placeholder_if_missing", action="store_true")
    ap.add_argument("--placeholder_title", default="MISSING_BVH")
    args = ap.parse_args()

    # collect BVHs
    bvh_files = []
    if args.bvh_file:
        if os.path.isfile(args.bvh_file):
            bvh_files = [args.bvh_file]
    elif args.bvh_glob:
        bvh_files = sorted(glob.glob(args.bvh_glob, recursive=True))

    if not bvh_files:
        if args.placeholder_if_missing:
            print("[WARN] BVH not found. Writing placeholders.")
            write_placeholders(args.out_dir, args.num_frames, title=args.placeholder_title)
            print(f"[OK] wrote {args.num_frames} placeholder frames -> {args.out_dir}")
            return
        print("[ERR] BVH not found.")
        print(f"      Tried file: {args.bvh_file}")
        print(f"      Tried glob: {args.bvh_glob}")
        raise SystemExit(2)

    # If multiple BVHs, mirror directory structure under out_dir
    if len(bvh_files) == 1:
        render_one_bvh(bvh_files[0], args.out_dir, args.num_frames, args.plane, args.pad_frac, args.lw)
        print(f"[OK] rendered {args.num_frames} frames -> {args.out_dir}")
        return

    base_out = args.out_dir
    for bf in bvh_files:
        rel = bf
        # try to drop leading "results/" if present, for nicer paths
        if rel.startswith("results/"):
            rel = rel[len("results/"):]
        # drop trailing "/motion.bvh"
        if rel.endswith("/motion.bvh"):
            rel = rel[:-len("/motion.bvh")]
        od = os.path.join(base_out, rel)
        render_one_bvh(bf, od, args.num_frames, args.plane, args.pad_frac, args.lw)
    print(f"[OK] rendered {args.num_frames} frames for {len(bvh_files)} BVHs under -> {base_out}")


if __name__ == "__main__":
    main()
