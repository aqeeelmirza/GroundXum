#!/usr/bin/env python3
"""
SWEEP 3b - Frame-count sensitivity from CACHED scores.   [NO GPU, NO VIDEOS]

The grounding JSONL stores all 16 per-frame scores per entity. This sweep
approximates "what if we had sampled only k frames?" by selecting k of the 16
cached frames at even stride (indices int(i*n/k) for i in range(k) - the same
rule ground.py uses to pick frames), re-aggregating with max, and re-applying
the decision threshold. It then reports, per summarizer and per k:

  * grounding rate at k
  * decision FLIP rate vs the full reference (k=16): fraction of entities whose
    present/absent verdict changes
  * agreement with the (fixed 16-frame-grid) VLM judge

plus the pairwise summarizer gap at each k, to show the ranking holds as k drops.

Caveat for the writeup: this is a cached approximation of true k-frame sampling.
Because the 16 cached frames are themselves evenly spaced, the k chosen here land
near a real uniform-k sample's positions; treat agreement vs a true re-sample as
an upper bound. The conclusion (how few frames before the verdict moves) is sound.

    python sweeps/sweep_frames_cached.py --scorer clip --thresh 0.2 \
        --entry "S1=outputs/grounding/youcook2/mm_blip_clip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl" \
        --entry "S2=outputs/grounding/youcook2/blip_base_clip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl" \
        --ks 1,2,4,8,16 --ref-k 16 \
        --out outputs/results/sweep_frames_cached_youcook2_clip.csv
"""
import argparse
import csv
from pathlib import Path

from _common import load_scores, load_vlm, prf


def subsample_indices(n, k):
    """k frame indices from n cached frames, even stride (ground.py's rule)."""
    if k >= n:
        return list(range(n))
    return [min(int(i * n / k), n - 1) for i in range(k)]


def reduce_at_k(scores, k):
    """max over the k-subsampled cached per-frame scores."""
    if not scores:
        return None
    idx = subsample_indices(len(scores), k)
    sub = [scores[i] for i in idx]
    return max(sub) if sub else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["blip", "clip"])
    ap.add_argument("--entry", action="append", required=True,
                    help="label=grounding=vlm")
    ap.add_argument("--thresh", type=float, required=True)
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--ref-k", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    t = args.thresh

    models = {}
    for e in args.entry:
        p = e.split("=")
        label, gfp, vfp = p[0], p[1], p[2]
        scores = load_scores(gfp)
        vlm = load_vlm(vfp)
        # keep only entities that actually have a per-frame array
        scores = {k_: v for k_, v in scores.items() if v.get("scores")}
        models[label] = {"scores": scores, "vlm": vlm}

    rows = []
    rate_by = {label: {} for label in models}
    for label, m in models.items():
        # reference verdicts at ref-k
        ref_pred = {ek: (reduce_at_k(v["scores"], args.ref_k) or -1e9) > t
                    for ek, v in m["scores"].items()}
        for k in ks:
            pred = {ek: ((reduce_at_k(v["scores"], k) or -1e9) > t)
                    for ek, v in m["scores"].items()}
            n_all = len(pred)
            rate = sum(pred.values()) / n_all if n_all else 0.0
            rate_by[label][k] = rate
            flips = sum(1 for ek in pred if pred[ek] != ref_pred[ek])
            flip_rate = flips / n_all if n_all else 0.0
            pr, rc, f1, acc, nv = prf(pred, m["vlm"])
            rows.append({
                "scorer": args.scorer, "model": label, "k": k, "n": n_all,
                "grounding_rate": round(rate, 4),
                "flip_vs_ref": round(flip_rate, 4),
                "vlm_n": nv, "vlm_f1": round(f1, 4), "vlm_acc": round(acc, 4),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nCached frame-count sweep ({args.scorer}, t={t}, ref k={args.ref_k}) -> {out}\n")
    print(f"{'model':<6}{'k':>4}{'n':>7}{'rate':>9}{'flip':>9}{'vlmF1':>8}{'vlmAcc':>8}")
    for r in rows:
        print(f"{r['model']:<6}{r['k']:>4}{r['n']:>7}{r['grounding_rate']:>9}"
              f"{r['flip_vs_ref']:>9}{r['vlm_f1']:>8}{r['vlm_acc']:>8}")

    labels = list(models.keys())
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        print(f"\n  {a} vs {b} grounding-rate gap by k:")
        for k in ks:
            g = rate_by[a][k] - rate_by[b][k]
            print(f"    k={k:<3} gap={g:+.4f}")


if __name__ == "__main__":
    main()