#!/usr/bin/env python3
"""
GroundSumm - master analysis.

Computes every result table needed for the paper and writes them to
outputs/results/master_results.{json,txt}.

Run from the groundsumm repo root:
    python master_analysis.py
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = ["youcook2", "videoxum"]
MODELS = ["mm", "blip"]

VLM_FILE = {
    ("youcook2", "mm"): "outputs/annotation/vlm/youcook2_mm_blip_full.jsonl",
    ("youcook2", "blip"):  "outputs/annotation/vlm/youcook2_blip_base_full.jsonl",
    ("videoxum", "mm"): "outputs/annotation/vlm/videoxum_mm_blip_full.jsonl",
    ("videoxum", "blip"):  "outputs/annotation/vlm/videoxum_blip_base_full.jsonl",
}
CLIP_FILE = {
    ("youcook2", "mm"): "outputs/grounding/youcook2/mm_blip_clip.jsonl",
    ("youcook2", "blip"):  "outputs/grounding/youcook2/blip_base_clip.jsonl",
    ("videoxum", "mm"): "outputs/grounding/videoxum/mm_blip_clip.jsonl",
    ("videoxum", "blip"):  "outputs/grounding/videoxum/blip_base_clip.jsonl",
}
HUMAN_FILE = {
    ("youcook2", "mm"): "outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl",
    ("youcook2", "blip"):  "outputs/annotation/judegements/youcook2_blip_base_judgments.jsonl",
    ("videoxum", "mm"): "outputs/annotation/judegements/videoxum_mm_blip_judgments.jsonl",
    ("videoxum", "blip"):  "outputs/annotation/judegements/videoxum_blip_base_judgments.jsonl",
}


def load_jsonl(fp):
    out = []
    with open(fp) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def is_mal(v):
    return v in (True, "True", "true")


def entity_key(summary_id, ent_id):
    return f"{summary_id}__{ent_id}"


def build_merged(ds, model):
    """entity_key -> {type, surface, malformed, vlm, blip, clip}"""
    merged = {}
    for d in load_jsonl(VLM_FILE[(ds, model)]):
        if "_error" in d or "vlm_judgment" not in d:
            continue
        k = d["entity_id"]
        merged[k] = {
            "type": d.get("type", ""),
            "surface": d.get("surface", ""),
            "malformed": is_mal(d.get("malformed")),
            "vlm": d.get("vlm_judgment"),
            "blip": d.get("blip_max"),
            "clip": None,
        }
    for d in load_jsonl(CLIP_FILE[(ds, model)]):
        sid = d.get("summary_id", "")
        for ent in d.get("entities", []):
            if not isinstance(ent, dict):
                continue
            k = entity_key(sid, ent.get("id", "?"))
            if k in merged:
                merged[k]["clip"] = ent.get("grounding", {}).get("max")
    return merged


results = {"parts": {}}
lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("=" * 80)
out("GROUNDSUMM MASTER RESULTS")
out("=" * 80)

# Pre-build all merged tables once (reused across parts)
MERGED = {(ds, model): build_merged(ds, model) for ds in DATASETS for model in MODELS}

# ---- PART 1: VLM grounding rates ----
out("\n## PART 1: VLM grounding rate (ground truth)")
out(f"{'dataset':<10}{'model':<7}{'N':>7}{'P':>7}{'A':>7}{'P-rate':>8}")
p1 = {}
for ds in DATASETS:
    for model in MODELS:
        m = MERGED[(ds, model)]
        c = Counter(v["vlm"] for v in m.values())
        n = len(m)
        prate = 100 * c["P"] / n if n else 0
        p1[f"{ds}_{model}"] = {"N": n, "P": c["P"], "A": c["A"], "amb": c["?"], "p_rate": round(prate, 2)}
        out(f"{ds:<10}{model:<7}{n:>7}{c['P']:>7}{c['A']:>7}{prate:>7.1f}%")
results["parts"]["1_vlm_grounding"] = p1

# ---- PART 2: DACMF vs BLIP ----
out("\n## PART 2: MM vs TXT (grounded entity count & rate)")
p2 = {}
for ds in DATASETS:
    d = p1[f"{ds}_mm"]
    b = p1[f"{ds}_blip"]
    p2[ds] = {
        "mm_grounded": d["P"], "mm_total": d["N"], "mm_rate": d["p_rate"],
        "blip_grounded": b["P"], "blip_total": b["N"], "blip_rate": b["p_rate"],
        "count_diff": d["P"] - b["P"], "rate_diff": round(d["p_rate"] - b["p_rate"], 2),
    }
    out(f"{ds}: MM {d['P']}/{d['N']} ({d['p_rate']}%)  vs  "
        f"TXT {b['P']}/{b['N']} ({b['p_rate']}%)  rate diff {d['p_rate']-b['p_rate']:+.1f}")
results["parts"]["2_mm_vs_blip"] = p2

# ---- PART 3: malformed vs real ----
out("\n## PART 3: Malformed vs real entity grounding (VLM=P rate)")
out(f"{'dataset':<10}{'model':<7}{'mal_P%':>8}{'mal_n':>7}{'real_P%':>9}{'real_n':>8}{'gap':>7}")
p3 = {}
for ds in DATASETS:
    for model in MODELS:
        m = MERGED[(ds, model)]
        mal = [v for v in m.values() if v["malformed"]]
        real = [v for v in m.values() if not v["malformed"]]
        mp = 100 * sum(1 for v in mal if v["vlm"] == "P") / len(mal) if mal else 0
        rp = 100 * sum(1 for v in real if v["vlm"] == "P") / len(real) if real else 0
        p3[f"{ds}_{model}"] = {
            "mal_p_rate": round(mp, 2), "mal_n": len(mal),
            "real_p_rate": round(rp, 2), "real_n": len(real), "gap": round(rp - mp, 2),
        }
        out(f"{ds:<10}{model:<7}{mp:>7.1f}%{len(mal):>7}{rp:>8.1f}%{len(real):>8}{rp-mp:>+7.1f}")
results["parts"]["3_malformed"] = p3

# ---- PART 4: per-type grounding rate ----
out("\n## PART 4: Per-entity-type VLM grounding rate (types with n>=10)")
p4 = {}
for ds in DATASETS:
    for model in MODELS:
        m = MERGED[(ds, model)]
        by = defaultdict(lambda: {"P": 0, "n": 0})
        for v in m.values():
            by[v["type"]]["n"] += 1
            if v["vlm"] == "P":
                by[v["type"]]["P"] += 1
        p4[f"{ds}_{model}"] = {
            t: {"p_rate": round(100 * x["P"] / x["n"], 1), "n": x["n"]}
            for t, x in by.items() if x["n"] >= 10
        }
        out(f"\n{ds}/{model}:")
        for t, x in sorted(by.items(), key=lambda kv: -kv[1]["n"]):
            if x["n"] >= 10:
                out(f"    {t:<12}{100*x['P']/x['n']:>6.1f}%  (n={x['n']})")
results["parts"]["4_per_type"] = p4

# ---- PART 5: scorer calibration (BLIP-ITM and CLIP) vs VLM ----
def calibrate(rows, score_key):
    THRESH = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    valid = [r for r in rows if r.get(score_key) is not None and r["vlm"] in ("P", "A")]
    best = (0, None, None)
    table = []
    for t in THRESH:
        tp = sum(1 for r in valid if r[score_key] > t and r["vlm"] == "P")
        fp = sum(1 for r in valid if r[score_key] > t and r["vlm"] == "A")
        fn = sum(1 for r in valid if r[score_key] <= t and r["vlm"] == "P")
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        table.append({"t": t, "prec": round(prec, 3), "rec": round(rec, 3), "f1": round(f1, 3)})
        if f1 > best[0]:
            best = (f1, t, (round(prec, 3), round(rec, 3)))
    return table, best, len(valid)


out("\n## PART 5: Scorer calibration vs VLM (best F1 per scorer)")
out(f"{'dataset':<10}{'model':<7}{'scorer':<10}{'bestF1':>7}{'@thresh':>8}{'n':>8}")
p5 = {}
for ds in DATASETS:
    for model in MODELS:
        rows = list(MERGED[(ds, model)].values())
        for scorer in ("blip", "clip"):
            table, best, n = calibrate(rows, scorer)
            p5[f"{ds}_{model}_{scorer}"] = {
                "best_f1": round(best[0], 3), "best_thresh": best[1], "n": n, "table": table,
            }
            out(f"{ds:<10}{model:<7}{scorer:<10}{best[0]:>6.3f}{str(best[1]):>8}{n:>8}")
results["parts"]["5_calibration"] = p5

# ---- PART 6: human audit agreement (800) vs VLM ----
out("\n## PART 6: Human audit (800) vs VLM agreement")
out(f"{'dataset':<10}{'model':<7}{'N':>5}{'agree%':>8}{'both_P':>8}{'both_A':>8}{'disputed':>9}")
p6 = {}
for ds in DATASETS:
    for model in MODELS:
        human = {d["entity_id"]: d["judgment"] for d in load_jsonl(HUMAN_FILE[(ds, model)])}
        m = MERGED[(ds, model)]
        both_p = both_a = disp = agree = n = 0
        for k, hv in human.items():
            if k not in m:
                continue
            vv = m[k]["vlm"]
            n += 1
            if hv == vv:
                agree += 1
            if hv == "P" and vv == "P":
                both_p += 1
            elif hv == "A" and vv == "A":
                both_a += 1
            elif hv == "P" and vv == "A":
                disp += 1
        ag = 100 * agree / n if n else 0
        p6[f"{ds}_{model}"] = {
            "N": n, "agree_pct": round(ag, 1),
            "both_P": both_p, "both_A": both_a, "disputed_P_vs_A": disp,
        }
        out(f"{ds:<10}{model:<7}{n:>5}{ag:>7.1f}%{both_p:>8}{both_a:>8}{disp:>9}")
results["parts"]["6_human_audit"] = p6

# ---- PART 7: BLIP-ITM vs CLIP scorer agreement ----
out("\n## PART 7: BLIP-ITM vs CLIP scorer agreement (at each scorer's best threshold)")
p7 = {}
for ds in DATASETS:
    for model in MODELS:
        m = MERGED[(ds, model)]
        bt = p5[f"{ds}_{model}_blip"]["best_thresh"]
        ct = p5[f"{ds}_{model}_clip"]["best_thresh"]
        both = [v for v in m.values() if v["blip"] is not None and v["clip"] is not None]
        agree = sum(1 for v in both if (v["blip"] > bt) == (v["clip"] > ct))
        ag = 100 * agree / len(both) if both else 0
        p7[f"{ds}_{model}"] = {
            "n": len(both), "agree_pct": round(ag, 1),
            "blip_thresh": bt, "clip_thresh": ct,
        }
        out(f"{ds:<10}{model:<7}  scorer-agreement={ag:.1f}%  (n={len(both)}, blip>{bt} vs clip>{ct})")
results["parts"]["7_scorer_agreement"] = p7

# ---- Save ----
Path("outputs/results").mkdir(parents=True, exist_ok=True)
with open("outputs/results/master_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("outputs/results/master_results.txt", "w") as f:
    f.write("\n".join(lines))

out("\n" + "=" * 80)
out("Saved: outputs/results/master_results.json and outputs/results/master_results.txt")
out("=" * 80)