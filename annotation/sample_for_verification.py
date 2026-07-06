#!/usr/bin/env python3
"""
GroundSumm - manual verification sample generator.

For each grounded JSONL file, stratify entities across score buckets, sample
N per bucket, and produce:
  1. A CSV with one row per (entity, video) pair, columns:
     judgment (empty), entity_id, surface, type, malformed, blip_max,
     blip_argmax_frame, video_id, frame_grid_path, summary_text
  2. One 4x4 frame-grid PNG per unique video referenced

Annotator fills the `judgment` column with: P (present) / A (absent) / ? (ambiguous).

Usage:
    python annotation/sample_for_verification.py \\
        --grounded outputs/grounding/youcook2/mm_blip_blip-itm.jsonl \\
        --id-source /path/to/test.txt \\
        --video-dir data/videos/youcook2 \\
        --per-bucket 40 \\
        --output-csv outputs/annotation/youcook2_mm_blip.csv \\
        --output-frames-dir outputs/annotation/frames
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
import cv2
from PIL import Image


BUCKETS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 0.50), (0.50, 1.01)]
BUCKET_LABELS = ["very_low", "low", "mid", "high", "very_high"]


def load_video_ids(id_source: str) -> list:
    ids = []
    with open(id_source) as f:
        for line in f:
            parts = line.split(None, 1)
            ids.append(parts[0] if parts else "")
    return ids


def build_video_index(video_dir: Path) -> dict:
    extensions = (".mp4", ".mkv", ".webm", ".avi")
    index = {}
    for ext in extensions:
        for p in video_dir.rglob(f"*{ext}"):
            index[p.stem] = str(p)
    return index


def make_frame_grid(video_path: str, out_path: Path, n: int = 16):
    """Save a 4x4 grid of n evenly spaced frames as a PNG."""
    if out_path.exists():
        return True
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False
    indices = [int(i * total / n) for i in range(n)]
    pil_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    cap.release()
    if len(pil_frames) < n:
        # Pad with last frame if some failed
        if not pil_frames:
            return False
        pil_frames += [pil_frames[-1]] * (n - len(pil_frames))

    # Resize each to a thumbnail
    thumb_w, thumb_h = 320, 180
    for i, f in enumerate(pil_frames):
        pil_frames[i] = f.resize((thumb_w, thumb_h), Image.LANCZOS)

    # Compose 4x4
    grid = Image.new("RGB", (thumb_w * 4, thumb_h * 4), "black")
    for i, f in enumerate(pil_frames[:16]):
        r, c = divmod(i, 4)
        grid.paste(f, (c * thumb_w, r * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, "PNG", optimize=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounded", required=True)
    parser.add_argument("--id-source", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--per-bucket", type=int, default=40)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-frames-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    video_ids = load_video_ids(args.id_source)
    video_index = build_video_index(Path(args.video_dir))

    # Collect candidate entities with their scores and video context
    candidates_by_bucket = {label: [] for label in BUCKET_LABELS}
    with open(args.grounded) as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            vid = video_ids[idx] if idx < len(video_ids) else None
            if vid is None or vid not in video_index:
                continue
            for ent in d.get("entities", []):
                if not isinstance(ent, dict):
                    continue
                g = ent.get("grounding", {})
                if "max" not in g:
                    continue
                max_s = g["max"]
                # Find bucket
                for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS):
                    if lo <= max_s < hi:
                        candidates_by_bucket[label].append({
                            "entity_id": f"{d.get('summary_id', f'sum_{idx}')}__{ent.get('id', '?')}",
                            "surface": ent.get("surface", ""),
                            "type": ent.get("type", ""),
                            "malformed": ent.get("malformed", False),
                            "blip_max": max_s,
                            "blip_argmax_frame": g.get("argmax_frame", 0),
                            "video_id": vid,
                            "video_path": video_index[vid],
                            "summary_id": d.get("summary_id", ""),
                            "summary_text": d.get("summary_text", "")[:200],
                            "bucket": label,
                        })
                        break

    # Sample per bucket
    print("Bucket sizes:")
    for label in BUCKET_LABELS:
        print(f"  {label:10s}  available={len(candidates_by_bucket[label])}")

    selected = []
    for label in BUCKET_LABELS:
        pool = candidates_by_bucket[label]
        rng.shuffle(pool)
        take = min(args.per_bucket, len(pool))
        if take < args.per_bucket:
            print(f"  WARN: {label} bucket only has {take} candidates (asked for {args.per_bucket})")
        selected.extend(pool[:take])

    print(f"Total selected: {len(selected)}")

    # Generate frame grids for all unique videos
    unique_videos = sorted({s["video_id"]: s["video_path"] for s in selected}.items())
    frames_dir = Path(args.output_frames_dir)
    print(f"Generating {len(unique_videos)} frame grids in {frames_dir}/...")
    failures = []
    for i, (vid, vpath) in enumerate(unique_videos):
        out = frames_dir / f"{vid}.png"
        if not make_frame_grid(vpath, out):
            failures.append(vid)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unique_videos)} done")
    if failures:
        print(f"  Failed grids: {len(failures)} (first 5: {failures[:5]})")

    # Write CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "judgment",  # empty, annotator fills
        "bucket", "blip_max", "blip_argmax_frame",
        "surface", "type", "malformed",
        "video_id", "frame_grid_path",
        "summary_text",
        "entity_id", "summary_id",
    ]
    rng.shuffle(selected)  # randomize within stratification to avoid annotator order bias
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in selected:
            writer.writerow({
                "judgment": "",
                "bucket": s["bucket"],
                "blip_max": round(s["blip_max"], 4),
                "blip_argmax_frame": s["blip_argmax_frame"],
                "surface": s["surface"],
                "type": s["type"],
                "malformed": s["malformed"],
                "video_id": s["video_id"],
                "frame_grid_path": str(frames_dir / f"{s['video_id']}.png"),
                "summary_text": s["summary_text"],
                "entity_id": s["entity_id"],
                "summary_id": s["summary_id"],
            })

    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote frame grids in: {frames_dir}/")
    print(f"\nNext step: open CSV in your spreadsheet tool, fill `judgment` with P / A / ?")


if __name__ == "__main__":
    main()
