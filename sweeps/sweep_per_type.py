#!/usr/bin/env python3
"""
SWEEP 5 - Per-type human-vs-VLM agreement.   [NO GPU - reads cache]

The human pass is permissive/inference-based; the VLM applies a strict
"visible in the frames" criterion. This script shows WHERE they diverge, by
breaking the PART-6 agreement number down by entity type, and grouping types
into visual vs non-visual to give one headline figure for SS4.

Hypothesis (from the annotation protocol): agreement is high on visually
concrete types (object, person, tool, setting) and collapses on non-visual
ones (ingredient seasonings, attribute, action), which is the quantitative
signature of inference-based human annotation.

Defaults follow master_analysis.py paths. Run from repo root:

    python sweeps/sweep_per_type.py \
        --out outputs/results/sweep_per_type.csv

Override datasets/models/min-n as needed.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from _common import load_jsonl, cohens_kappa

VLM_FILE = {
    ("youcook2", "mm"): "outputs/annotation/vlm/youcook2_mm_blip_full.jsonl",
    ("youcook2", "blip"):  "outputs/annotation/vlm/youcook2_blip_base_full.jsonl",
    ("videoxum", "mm"): "outputs/annotation/vlm/videoxum_mm_blip_full.jsonl",
    ("videoxum", "blip"):  "outputs/annotation/vlm/videoxum_blip_base_full.jsonl",
}
HUMAN_FILE = {
    ("youcook2", "mm"): "outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl",
    ("youcook2", "blip"):  "outputs/annotation/judegements/youcook2_blip_base_judgments.jsonl",
    ("videoxum", "mm"): "outputs/annotation/judegements/videoxum_mm_blip_judgments.jsonl",
    ("videoxum", "blip"):  "outputs/annotation/judegements/videoxum_blip_base_judgments.jsonl",
}

# coarse grouping: which entity types are visually concrete
VISUAL = {"object", "person", "tool", "setting"}
NONVISUAL = {"ingredient", "attribute", "action"}
LABELS = {"P", "A", "?"}


def visual_group(t):
    if t in VISUAL:
        return "visual"
    if t in NONVISUAL:
        return "non-visual"
    return "other"


def load_vlm_types(fp):
    """entity_id -> (vlm_judgment, type)"""
    out = {}
    for d in load_jsonl(fp):
        if "_error" in d or "vlm_judgment" not in d:
            continue
        out[d["entity_id"]] = (d["vlm_judgment"], d.get("type", ""))
    return out


def load_human_types(fp):
    """entity_id -> (judgment, type). Coerces stringy fields."""
    out = {}
    for d in load_jsonl(fp):
        j = d.get("judgment")
        if j is None:
            continue
        out[d["entity_id"]] = (j, d.get("type", ""))
    return out


def stats(pairs):
    """pairs: list of (human, vlm). Returns dict of agreement metrics."""
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for h, v in pairs if h == v)
    # human-present-but-vlm-absent: the permissive-overcall direction
    h_p_v_a = sum(1 for h, v in pairs if h == "P" and v == "A")
    h_p = sum(1 for h, v in pairs if h == "P")
    a = {e: i for i, e in enumerate(pairs)}  # placeholder to keep linter calm
    hd = {f"h{i}": p[0] for i, p in enumerate(pairs)}
    vd = {f"h{i}": p[1] for i, p in enumerate(pairs)}
    kappa, _ = cohens_kappa(hd, vd, LABELS)
    return {
        "n": n,
        "agree_pct": round(100 * agree / n, 1),
        "kappa": round(kappa, 4) if kappa is not None else None,
        "human_P_pct": round(100 * h_p / n, 1),
        "humanP_vlmA_pct": round(100 * h_p_v_a / n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="youcook2,videoxum")
    ap.add_argument("--models", default="mm,blip")
    ap.add_argument("--min-n", type=int, default=10, help="min entities to report a type")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    models = args.models.split(",")

    rows = []          # per-type rows
    group_rows = []    # visual/non-visual rows
    for ds in datasets:
        for model in models:
            vlm = load_vlm_types(VLM_FILE[(ds, model)])
            human = load_human_types(HUMAN_FILE[(ds, model)])
            by_type = defaultdict(list)
            by_group = defaultdict(list)
            for eid, (hj, htype) in human.items():
                if eid not in vlm:
                    continue
                vj, vtype = vlm[eid]
                t = vtype or htype
                by_type[t].append((hj, vj))
                by_group[visual_group(t)].append((hj, vj))

            for t, pairs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
                s = stats(pairs)
                if s and s["n"] >= args.min_n:
                    rows.append({"dataset": ds, "model": model, "type": t,
                                 "group": visual_group(t), **s})
            for g, pairs in by_group.items():
                s = stats(pairs)
                if s:
                    group_rows.append({"dataset": ds, "model": model, "group": g, **s})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "type", "group",
                                          "n", "agree_pct", "kappa",
                                          "human_P_pct", "humanP_vlmA_pct"])
        w.writeheader()
        w.writerows(rows)
    gout = out.with_name(out.stem + "_grouped.csv")
    with open(gout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "model", "group", "n",
                                          "agree_pct", "kappa", "human_P_pct",
                                          "humanP_vlmA_pct"])
        w.writeheader()
        w.writerows(group_rows)

    print(f"\nPer-type agreement -> {out}")
    print(f"Grouped            -> {gout}\n")
    print(f"{'ds':<9}{'model':<6}{'type':<12}{'grp':<11}{'n':>5}{'agree%':>8}{'kappa':>8}{'hP%':>7}{'hP_vA%':>8}")
    for r in rows:
        print(f"{r['dataset']:<9}{r['model']:<6}{r['type']:<12}{r['group']:<11}"
              f"{r['n']:>5}{r['agree_pct']:>8}{str(r['kappa']):>8}"
              f"{r['human_P_pct']:>7}{r['humanP_vlmA_pct']:>8}")
    print(f"\n-- visual vs non-visual --")
    print(f"{'ds':<9}{'model':<6}{'group':<11}{'n':>5}{'agree%':>8}{'kappa':>8}{'hP%':>7}{'hP_vA%':>8}")
    for r in sorted(group_rows, key=lambda x: (x['dataset'], x['model'], x['group'])):
        print(f"{r['dataset']:<9}{r['model']:<6}{r['group']:<11}{r['n']:>5}"
              f"{r['agree_pct']:>8}{str(r['kappa']):>8}{r['human_P_pct']:>7}{r['humanP_vlmA_pct']:>8}")


if __name__ == "__main__":
    main()