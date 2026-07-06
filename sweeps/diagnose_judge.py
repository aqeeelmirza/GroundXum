#!/usr/bin/env python3
"""
Diagnose the low judge-vs-human agreement.

Compares, on the SAME validation entities, three label sources:
  human   : the judgment field in the judgments JSONL
  v0_fresh : this sweep's fresh v0_original run (from --pred-out preds file)
  cached  : the vlm_judgment your paper's tables used (the *_full.jsonl)

Interpretation:
  * cached-vs-human HIGH but v0_fresh-vs-human LOW  -> the two VLM runs see
    different grids for the same entity (validation-mode CSV grid vs full-mode
    positional grid). A pipeline-alignment bug; the paper number stands, the
    re-run is wrong - fixable.
  * cached-vs-human ALSO LOW -> the paper's reported agreement itself is low,
    or the 'judgment' field is not the human label. Bigger issue; check labels.
  * v0_fresh-vs-cached LOW -> confirms the two judge runs disagree (grid/mode).

    python sweeps/diagnose_judge.py \
        --judgments outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl \
        --preds outputs/results/sweep_judge_prompt_youcook2_mm_preds.jsonl \
        --cached outputs/annotation/vlm/youcook2_mm_blip_full.jsonl \
        --csv outputs/annotation/youcook2_mm_blip.csv
"""
import argparse
import csv
import json
from collections import Counter

from _common import load_jsonl, cohens_kappa


def agree(a, b, labels):
    keys = [k for k in a if k in b and a[k] in labels and b[k] in labels]
    if not keys:
        return None, 0
    return sum(1 for k in keys if a[k] == b[k]) / len(keys), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--cached", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--variant", default="v0_original")
    args = ap.parse_args()

    human = {d["entity_id"]: d.get("judgment") for d in load_jsonl(args.judgments)}
    v0 = {d["entity_id"]: d["vlm"] for d in load_jsonl(args.preds)
          if d.get("variant") == args.variant}
    cached = {d["entity_id"]: d.get("vlm_judgment") for d in load_jsonl(args.cached)
              if "vlm_judgment" in d and "_error" not in d}
    grid = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            grid[row["entity_id"]] = row.get("frame_grid_path")

    L = {"P", "A", "?"}
    print(f"counts: human={len(human)} v0_fresh={len(v0)} cached={len(cached)}")
    print(f"label dist human : {dict(Counter(human.values()))}")
    print(f"label dist v0    : {dict(Counter(v0.values()))}")
    print(f"label dist cached: {dict(Counter(v in L and v or '?' for v in cached.values()))}\n")

    for name, a, b in [
        ("v0_fresh vs human", v0, human),
        ("cached   vs human", cached, human),
        ("v0_fresh vs cached", v0, cached),
    ]:
        ag, n = agree(a, b, L)
        k, _ = cohens_kappa(a, b, L)
        print(f"{name}: agree={ag:.3f} kappa={k:.4f} (n={n})" if ag is not None
              else f"{name}: no overlap")

    # show disagreements between fresh v0 and human, with grid path to eyeball
    print("\nfirst 12 v0_fresh != human (check the grid matches the surface):")
    shown = 0
    preds_full = {d["entity_id"]: d for d in load_jsonl(args.preds)
                  if d.get("variant") == args.variant}
    for eid, hv in human.items():
        if eid in v0 and v0[eid] != hv and hv in L:
            d = preds_full.get(eid, {})
            print(f"  {eid}  surface='{d.get('surface','')}'  human={hv}  v0={v0[eid]}  "
                  f"cached={cached.get(eid,'-')}  grid={grid.get(eid,'?')}")
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    main()