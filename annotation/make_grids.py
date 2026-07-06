#!/usr/bin/env python3
"""
GroundSumm - batch generator for 4x4 frame-grid PNGs.

For each video ID in the test-set source file, if the grid PNG doesn't already
exist in --output-dir, sample 16 frames evenly and save as a 4x4 grid.

Parallelized across CPU workers. Skips existing files.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm


def make_frame_grid(video_path: str, out_path: str, n: int = 16) -> tuple:
    """Sample n evenly-spaced frames and save as 4x4 grid. Returns (video_id, success, reason)."""
    op = Path(out_path)
    if op.exists():
        return (op.stem, True, "exists")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (op.stem, False, "cannot_open")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return (op.stem, False, "no_frames")
    indices = [int(i * total / n) for i in range(n)]
    pil_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    cap.release()
    if not pil_frames:
        return (op.stem, False, "no_frames_decoded")
    if len(pil_frames) < n:
        pil_frames += [pil_frames[-1]] * (n - len(pil_frames))
    thumb_w, thumb_h = 320, 180
    pil_frames = [f.resize((thumb_w, thumb_h), Image.LANCZOS) for f in pil_frames]
    grid = Image.new("RGB", (thumb_w * 4, thumb_h * 4), "black")
    for i, f in enumerate(pil_frames[:16]):
        r, c = divmod(i, 4)
        grid.paste(f, (c * thumb_w, r * thumb_h))
    op.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, "PNG", optimize=True)
    return (op.stem, True, "created")


def load_video_index(video_dir: Path) -> dict:
    extensions = (".mp4", ".mkv", ".webm", ".avi")
    index = {}
    for ext in extensions:
        for p in video_dir.rglob(f"*{ext}"):
            index[p.stem] = str(p)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-source", required=True, help="test-set source file (first token = video ID)")
    parser.add_argument("--video-dir", required=True, help="dir containing source videos")
    parser.add_argument("--output-dir", required=True, help="dir for output PNG grids")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.id_source) as f:
        vids = [line.split(None, 1)[0] for line in f if line.strip()]
    print(f"Test-set video IDs: {len(vids)}")
    vindex = load_video_index(Path(args.video_dir))
    print(f"Indexed videos in {args.video_dir}: {len(vindex)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    skipped_missing_video = 0
    skipped_already_exists = 0
    for vid in vids:
        if vid not in vindex:
            skipped_missing_video += 1
            continue
        out_path = out_dir / f"{vid}.png"
        if out_path.exists():
            skipped_already_exists += 1
            continue
        tasks.append((vindex[vid], str(out_path)))

    print(f"Skipped (video file missing): {skipped_missing_video}")
    print(f"Skipped (grid already exists): {skipped_already_exists}")
    print(f"Tasks to run: {len(tasks)}")

    if not tasks:
        print("Nothing to do.")
        return

    n_ok = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(make_frame_grid, vp, op): op for vp, op in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building grids"):
            vid, ok, reason = fut.result()
            if ok:
                n_ok += 1
            else:
                failures.append((vid, reason))

    print(f"\nCreated: {n_ok}")
    if failures:
        print(f"Failed:  {len(failures)} (first 10: {failures[:10]})")


if __name__ == "__main__":
    main()
