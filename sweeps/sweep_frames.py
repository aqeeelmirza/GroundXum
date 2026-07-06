#!/usr/bin/env python3
"""
SWEEP 3 (analysis) - Keyframe-count sensitivity.   [NO GPU - reads run_frames.sh output]

For each frame count F, and a fixed decision threshold, report per summarizer:
  * grounding rate at F
  * decision FLIP rate vs the paper-default run (16 frames): fraction of entities
    whose present/absent verdict changes
  * vs-VLM agreement (if VLM file passed) so you can see if accuracy saturates
And the pairwise summarizer gap at each F, to show the ranking holds as frames vary.

    python sweeps/sweep_frames.py \
        --root outputs/grounding/youcook2/frames \
        --scorer blip --ref-frames 16 --thresh 0.2 \
        --entry "S1=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl" \
        --entry "S2=outputs/annotation/vlm/youcook2_blip_base_full.jsonl" \
        --out outputs/results/sweep_frames_youcook2_blip.csv

--entry is label=vlm_file (vlm optional; omit to skip accuracy columns).
Grounding files are discovered as {root}/{label}_{scorer}_f{F}.jsonl
"""
import argparse
import csv
import glob
import os
import re
from pathlib import Path

from _common import load_scores, load_vlm, prf


def discover(root, label, scorer):
    """Return {frame_count: filepath} for a label."""
    out = {}
    for fp in glob.glob(os.path.join(root, f"{label}_{scorer}_f*.jsonl")):
        m = re.search(rf"{re.escape(label)}_{re.escape(scorer)}_f(\d+)\.jsonl$", fp)
        if m:
            out[int(m.group(1))] = fp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scorer", required=True, choices=["blip", "clip"])
    ap.add_argument("--ref-frames", type=int, default=16)
    ap.add_argument("--thresh", type=float, required=True,
                    help="decision threshold (use the calibrated best from sweep_threshold)")
    ap.add_argument("--entry", action="append", required=True,
                    help="label=vlm_file ; vlm optional (label= alone is fine)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = []
    for e in args.entry:
        p = e.split("=")
        entries.append((p[0], p[1] if len(p) > 1 and p[1] else None))

    t = args.thresh
    rows = []
    rate_by = {}  # label -> {F: rate}
    for label, vfp in entries:
        files = discover(args.root, label, args.scorer)
        if not files:
            print(f"WARNING: no files for {label} under {args.root}")
            continue
        vlm = load_vlm(vfp) if vfp else {}
        ref_pred = None
        if args.ref_frames in files:
            ref_scores = load_scores(files[args.ref_frames])
            ref_pred = {k: (v["max"] > t) for k, v in ref_scores.items()}
        rate_by[label] = {}
        for F in sorted(files):
            scores = load_scores(files[F])
            pred = {k: (v["max"] > t) for k, v in scores.items()}
            rate = sum(pred.values()) / len(pred) if pred else 0.0
            rate_by[label][F] = rate
            # flip rate vs reference
            flip = None
            if ref_pred is not None:
                shared = [k for k in pred if k in ref_pred]
                if shared:
                    flip = sum(1 for k in shared if pred[k] != ref_pred[k]) / len(shared)
            pr, rc, f1, acc, n = prf(pred, vlm) if vlm else (0, 0, 0, 0, 0)
            rows.append({
                "model": label, "frames": F, "n": len(pred),
                "grounding_rate": round(rate, 4),
                "flip_vs_ref": round(flip, 4) if flip is not None else "",
                "vlm_f1": round(f1, 4) if vlm else "",
                "vlm_acc": round(acc, 4) if vlm else "",
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nFrame-count sweep ({args.scorer}, t={t}) -> {out}\n")
    print(f"{'model':<8}{'frames':>7}{'rate':>9}{'flip':>9}{'vlmF1':>8}")
    for r in rows:
        print(f"{r['model']:<8}{r['frames']:>7}{r['grounding_rate']:>9}"
              f"{str(r['flip_vs_ref']):>9}{str(r['vlm_f1']):>8}")

    # ranking stability across frame counts
    labels = [l for l in rate_by]
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        common = sorted(set(rate_by[a]) & set(rate_by[b]))
        print(f"\n  {a} vs {b} gap by frame count:")
        for F in common:
            g = rate_by[a][F] - rate_by[b][F]
            print(f"    f={F:<3} gap={g:+.4f}")


if __name__ == "__main__":
    main()