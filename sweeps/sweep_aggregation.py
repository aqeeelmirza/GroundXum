#!/usr/bin/env python3
"""
SWEEP 2 - Frame-aggregation sensitivity.   [NO GPU - reads cache]

ground.py reduces a per-frame score array to a single entity score via max().
This sweep tests whether that choice matters, using the cached `scores` array:
it recomputes entity scores under several reductions and reports best-F1-vs-VLM
and grounding rate for each.

    agg in {max, mean, top3_mean, top5_mean}

If the audit's conclusions are stable across reductions, that is a clean
reproducibility result; if max() is doing heavy lifting, that is worth knowing.

    python sweeps/sweep_aggregation.py \
        --entry "S1=outputs/grounding/youcook2/mm_blip_blip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl" \
        --entry "S2=outputs/grounding/youcook2/blip_base_blip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl" \
        --grid 60 \
        --out outputs/results/sweep_aggregation_youcook2_blip.csv
"""
import argparse
import csv
from pathlib import Path

from _common import load_scores, load_vlm, prf, linspace


def reduce_scores(arr, agg):
    if not arr:
        return None
    if agg == "max":
        return max(arr)
    if agg == "mean":
        return sum(arr) / len(arr)
    if agg.startswith("top"):
        k = int(agg[3:].split("_")[0])
        top = sorted(arr, reverse=True)[:k]
        return sum(top) / len(top)
    raise ValueError(agg)


AGGS = ["max", "mean", "top3_mean", "top5_mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", action="append", required=True,
                    help="label=grounding=vlm")
    ap.add_argument("--grid", type=int, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = []
    for e in args.entry:
        p = e.split("=")
        entries.append((p[0], p[1], p[2]))

    rows = []
    for label, gfp, vfp in entries:
        scores = load_scores(gfp)
        vlm = load_vlm(vfp)
        for agg in AGGS:
            reduced = {}
            for k, v in scores.items():
                arr = v.get("scores")
                val = reduce_scores(arr, agg) if arr else v.get("max")
                if val is not None:
                    reduced[k] = val
            vals = list(reduced.values())
            if not vals:
                continue
            grid = linspace(min(vals), max(vals), args.grid)
            best = (0.0, None, 0.0, 0.0)  # f1, t, prec, rec
            for t in grid:
                pred = {k: (val > t) for k, val in reduced.items()}
                pr, rc, f1, _, n = prf(pred, vlm)
                if f1 > best[0]:
                    best = (f1, t, pr, rc)
            # grounding rate at the best-F1 threshold
            bt = best[1]
            rate = sum(1 for val in vals if val > bt) / len(vals) if bt is not None else 0.0
            rows.append({
                "model": label, "agg": agg,
                "best_f1": round(best[0], 4), "best_thresh": round(best[1], 5) if best[1] is not None else None,
                "prec": round(best[2], 4), "rec": round(best[3], 4),
                "n_entities": len(vals), "grounding_rate_at_bestF1": round(rate, 4),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nAggregation sweep -> {out}\n")
    print(f"{'model':<8}{'agg':<12}{'bestF1':>8}{'@t':>9}{'rate':>9}")
    for r in rows:
        print(f"{r['model']:<8}{r['agg']:<12}{r['best_f1']:>8}{str(r['best_thresh']):>9}{r['grounding_rate_at_bestF1']:>9}")


if __name__ == "__main__":
    main()