#!/usr/bin/env python3
"""
SWEEP 1 - Threshold sensitivity + comparison stability.   [NO GPU - reads cache]

For one scorer (blip or clip), sweep the present/absent decision threshold over a
dense grid derived from the observed score distribution, and for each threshold:

  * agreement with the VLM judge   (precision / recall / F1 / accuracy)
  * agreement with the 800 human labels, where available
  * each summarizer's grounding rate  (= fraction of its entities judged present),
    which is the metric the paper uses to compare summarizers
  * the pairwise gap between summarizers, and whether its SIGN stays stable

The headline reproducibility claim lives here: "the ranking S_a vs S_b holds for
every threshold in [t_lo, t_hi]" - or, if it doesn't, exactly where it breaks.

Entries are model-agnostic so you control naming (anonymize the TCSVT system):

    python sweeps/sweep_threshold.py \
        --scorer blip \
        --entry "S1=outputs/grounding/youcook2/mm_blip_blip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl=outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl" \
        --entry "S2=outputs/grounding/youcook2/blip_base_blip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl=outputs/annotation/judegements/youcook2_blip_base_judgments.jsonl" \
        --grid 60 \
        --out outputs/results/sweep_threshold_youcook2_blip.csv

Each --entry is  label=grounding_file=vlm_file[=human_file]  (human optional).
"""
import argparse
import csv
from pathlib import Path

from _common import load_scores, load_vlm, load_human, prf, linspace


def parse_entry(s):
    parts = s.split("=")
    if len(parts) < 3:
        raise ValueError(f"--entry must be label=grounding=vlm[=human], got: {s}")
    label, grounding, vlm = parts[0], parts[1], parts[2]
    human = parts[3] if len(parts) > 3 else None
    return label, grounding, vlm, human


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", required=True, choices=["blip", "clip"],
                    help="only used to label output; scores are read from the file")
    ap.add_argument("--entry", action="append", required=True,
                    help="label=grounding=vlm[=human]; repeat for each summarizer")
    ap.add_argument("--grid", type=int, default=60, help="number of thresholds")
    ap.add_argument("--ref-thresh", type=float, default=None,
                    help="reference threshold for sign-stability (default: grid midpoint)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    entries = [parse_entry(e) for e in args.entry]
    models = {}
    all_max = []
    for label, gfp, vfp, hfp in entries:
        scores = load_scores(gfp)
        vlm = load_vlm(vfp)
        human = load_human(hfp) if hfp else {}
        models[label] = {"scores": scores, "vlm": vlm, "human": human}
        all_max += [v["max"] for v in scores.values() if v["max"] is not None]

    if not all_max:
        raise SystemExit("No cached scores found - check grounding file paths.")
    lo, hi = min(all_max), max(all_max)
    grid = linspace(lo, hi, args.grid)
    ref_t = args.ref_thresh if args.ref_thresh is not None else grid[len(grid) // 2]

    # ---- per-threshold rows ----
    rows = []
    rate_by_model = {label: {} for label in models}
    for t in grid:
        for label, m in models.items():
            pred = {k: (v["max"] > t) for k, v in m["scores"].items() if v["max"] is not None}
            n_all = len(pred)
            rate = sum(pred.values()) / n_all if n_all else 0.0
            rate_by_model[label][t] = rate
            vp, vr, vf, va, vn = prf(pred, m["vlm"])
            hp, hr, hf, ha, hn = prf(pred, m["human"]) if m["human"] else (0, 0, 0, 0, 0)
            rows.append({
                "scorer": args.scorer, "model": label, "threshold": round(t, 5),
                "n_all": n_all, "grounding_rate": round(rate, 4),
                "vlm_n": vn, "vlm_prec": round(vp, 4), "vlm_rec": round(vr, 4),
                "vlm_f1": round(vf, 4), "vlm_acc": round(va, 4),
                "human_n": hn, "human_prec": round(hp, 4), "human_rec": round(hr, 4),
                "human_f1": round(hf, 4), "human_acc": round(ha, 4),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- ranking-stability summary for every model pair ----
    labels = list(models.keys())
    print(f"\nThreshold sweep ({args.scorer}): {args.grid} thresholds in [{lo:.4f}, {hi:.4f}]")
    print(f"Reference threshold for sign test: {ref_t:.4f}")
    print(f"Per-threshold CSV: {out}\n")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            gaps = [rate_by_model[a][t] - rate_by_model[b][t] for t in grid]
            ref_gap = rate_by_model[a][ref_t] - rate_by_model[b][ref_t]
            ref_sign = (ref_gap > 0) - (ref_gap < 0)
            same = sum(1 for g in gaps if ((g > 0) - (g < 0)) == ref_sign and ref_sign != 0)
            flips = any(((g > 0) - (g < 0)) == -ref_sign for g in gaps) if ref_sign else None
            print(f"  {a} vs {b}:")
            print(f"    gap at ref = {ref_gap:+.4f}  (sign {'+' if ref_sign>0 else '-' if ref_sign<0 else '0'})")
            print(f"    sign stable across {same}/{len(grid)} thresholds; "
                  f"sign ever flips: {flips}")
            print(f"    gap range [{min(gaps):+.4f}, {max(gaps):+.4f}]\n")


if __name__ == "__main__":
    main()