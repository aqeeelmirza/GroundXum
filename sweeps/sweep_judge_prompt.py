#!/usr/bin/env python3
"""
SWEEP 4 - VLM judge-prompt sensitivity.   [GPU/Ollama - validation set only]

Re-runs the Qwen2.5-VL judge on the SAME pre-rendered grid images with several
prompt variants, on the 800-entity human-labeled validation set. For each variant
it reports agreement and Cohen's kappa vs the human labels; it also reports
pairwise kappa BETWEEN variants - i.e. how much the verdict moves when you only
reword the prompt. This is the headline reproducibility result for a VLM-as-judge.

Inputs are the same files auto_annotate.py uses in `validation` mode:
  --judgments : human-judgment JSONL (entity_id + judgment + bucket/type)
  --csv       : maps entity_id -> frame_grid_path  (the 4x4 grid PNG)

Variants: v0_original (verbatim from auto_annotate.py), v1_reworded,
v2_cot (brief reasoning then a final letter), v3_no_ambiguous (forced P/A),
v4_order_swap (options listed in a different order).

    python sweeps/sweep_judge_prompt.py \
        --judgments outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl \
        --csv       outputs/annotation/youcook2_mm_blip.csv \
        --variants  v0_original,v1_reworded,v2_cot,v3_no_ambiguous,v4_order_swap \
        --out       outputs/results/sweep_judge_prompt_youcook2_mm.csv \
        --pred-out  outputs/results/sweep_judge_prompt_youcook2_mm_preds.jsonl

Cost: ~ (n_items * n_variants) VLM calls. v2_cot is slower (more tokens). Use
--limit to subsample while wiring it up, then run full for the paper numbers.
"""
import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

from _common import cohens_kappa

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("GROUNDSUMM_VLM", "qwen2.5vl:7b")


# ----------------------------------------------------------------------
# Prompt variants. Each returns (prompt_text, num_predict, parser_name).
# Keep the TASK identical across variants - only the phrasing changes - so
# kappa-between-variants measures phrasing sensitivity, not task drift.
# ----------------------------------------------------------------------

V0 = """You are a careful video annotator.

This image is a 4x4 grid of 16 frames sampled uniformly from a short video.

Question: Is "{surface}" visibly present somewhere in any of these 16 frames?

Decide based ONLY on what you can see in the frames, not on what is plausible.

Answer with ONE letter:
- P : Present. You can clearly see the entity in at least one frame.
- A : Absent. The entity is not visible in any frame.
- ? : Ambiguous. Occluded, out-of-focus, ambiguous reference, or an abstract concept that cannot be visualized.

Your answer (one letter only):"""

V1 = """Look at this 4x4 montage of 16 frames taken from one short video.

Can you actually SEE "{surface}" in any of the 16 frames? Judge only from the
pixels, not from what you would expect to be there.

Reply with a single letter:
P = yes, clearly visible in at least one frame
A = no, not visible in any frame
? = cannot tell (occluded, blurry, abstract, or unclear what is referred to)

Letter:"""

V2 = """This image is a 4x4 grid of 16 frames from a short video.

Question: Is "{surface}" visibly present in any of the 16 frames?
Judge only from what is visible.

First, in one short sentence, note what you see relevant to "{surface}".
Then give your verdict on a new line as exactly "Answer: X" where X is one of:
P (clearly visible), A (not visible), ? (occluded/blurry/abstract/unclear)."""

V3 = """You are a careful video annotator.

This image is a 4x4 grid of 16 frames sampled uniformly from a short video.

Question: Is "{surface}" visibly present somewhere in any of these 16 frames?
Decide based ONLY on what you can see in the frames.

You must answer with ONE letter:
- P : Present. Visible in at least one frame.
- A : Absent. Not visible in any frame.

Your answer (one letter only):"""

V4 = """You are a careful video annotator.

This image is a 4x4 grid of 16 frames sampled uniformly from a short video.

Question: Is "{surface}" visibly present somewhere in any of these 16 frames?
Decide based ONLY on what you can see in the frames, not on what is plausible.

Answer with ONE letter:
- A : Absent. The entity is not visible in any frame.
- ? : Ambiguous. Occluded, out-of-focus, ambiguous reference, or abstract concept.
- P : Present. You can clearly see the entity in at least one frame.

Your answer (one letter only):"""

VARIANTS = {
    "v0_original":     (V0, 10,  "first"),
    "v1_reworded":     (V1, 10,  "first"),
    "v2_cot":          (V2, 128, "final"),
    "v3_no_ambiguous": (V3, 10,  "first"),
    "v4_order_swap":   (V4, 10,  "first"),
}


def parse_first(text):
    t = text.strip().upper()
    for c in t:
        if c in ("P", "A", "?"):
            return c
    if "PRESENT" in t: return "P"
    if "ABSENT" in t:  return "A"
    if "AMBIG" in t:   return "?"
    return "?"


def parse_final(text):
    """For CoT: prefer the letter after 'Answer:'; else the last P/A/? token."""
    m = re.search(r"answer\s*[:=]\s*\(?([pa?])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = [c for c in text.upper() if c in ("P", "A", "?")]
    return letters[-1] if letters else "?"


PARSERS = {"first": parse_first, "final": parse_final}


def load_image_b64(path, cache):
    if path not in cache:
        with open(path, "rb") as f:
            cache[path] = base64.b64encode(f.read()).decode()
    return cache[path]


def call_vlm(model, img_b64, prompt, num_predict, parser, max_retries=3):
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": num_predict},
                },
                timeout=180,
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            return raw, parser(raw)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def load_items(judgments_path, csv_path):
    with open(judgments_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    grid = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            grid[row["entity_id"]] = row.get("frame_grid_path")
    for it in items:
        it["_grid_path"] = grid.get(it["entity_id"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pred-out", default=None, help="optional per-entity predictions JSONL")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant: {v} (choose from {list(VARIANTS)})")

    items = load_items(args.judgments, args.csv)
    if args.limit:
        items = items[: args.limit]
    items = [it for it in items if it.get("_grid_path") and Path(it["_grid_path"]).exists()]
    print(f"{len(items)} items with valid grids; variants: {variants}")

    human = {it["entity_id"]: it.get("judgment") for it in items}
    preds = {v: {} for v in variants}  # variant -> entity_id -> letter

    img_cache = {}
    pred_f = open(args.pred_out, "w") if args.pred_out else None
    for v in variants:
        prompt_t, npred, parser_name = VARIANTS[v]
        parser = PARSERS[parser_name]
        for it in tqdm(items, desc=v):
            b64 = load_image_b64(it["_grid_path"], img_cache)
            prompt = prompt_t.format(surface=it.get("surface", ""))
            try:
                raw, j = call_vlm(args.model, b64, prompt, npred, parser)
            except Exception as e:
                j, raw = "?", f"_error:{e}"
            preds[v][it["entity_id"]] = j
            if pred_f:
                pred_f.write(json.dumps({
                    "entity_id": it["entity_id"], "surface": it.get("surface"),
                    "type": it.get("type"), "variant": v,
                    "human": human.get(it["entity_id"]), "vlm": j, "raw": raw,
                }, ensure_ascii=False) + "\n")
    if pred_f:
        pred_f.close()

    # ---- vs-human agreement + kappa per variant ----
    summary = []
    for v in variants:
        labels = {"P", "A"} if v == "v3_no_ambiguous" else {"P", "A", "?"}
        keys = [k for k in preds[v] if human.get(k) in labels and preds[v][k] in labels]
        agree = sum(1 for k in keys if preds[v][k] == human[k])
        acc = agree / len(keys) if keys else 0.0
        kappa, n = cohens_kappa(preds[v], human, labels)
        summary.append({"variant": v, "n": n, "agree_vs_human": round(acc, 4),
                        "kappa_vs_human": round(kappa, 4) if kappa is not None else None})

    # ---- pairwise kappa between variants ----
    pair_rows = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            labels = {"P", "A"} if "v3_no_ambiguous" in (a, b) else {"P", "A", "?"}
            kappa, n = cohens_kappa(preds[a], preds[b], labels)
            pair_rows.append({"variant_a": a, "variant_b": b, "n": n,
                              "kappa": round(kappa, 4) if kappa is not None else None})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "a", "b", "n", "metric", "value"])
        for s in summary:
            w.writerow(["vs_human", s["variant"], "", s["n"], "agreement", s["agree_vs_human"]])
            w.writerow(["vs_human", s["variant"], "", s["n"], "kappa", s["kappa_vs_human"]])
        for p in pair_rows:
            w.writerow(["pairwise", p["variant_a"], p["variant_b"], p["n"], "kappa", p["kappa"]])

    print("\n== vs human ==")
    print(f"{'variant':<18}{'n':>5}{'agree':>9}{'kappa':>9}")
    for s in summary:
        print(f"{s['variant']:<18}{s['n']:>5}{s['agree_vs_human']:>9}{str(s['kappa_vs_human']):>9}")
    print("\n== pairwise kappa (judge stability under rewording) ==")
    for p in pair_rows:
        print(f"  {p['variant_a']:<16} vs {p['variant_b']:<16} kappa={p['kappa']} (n={p['n']})")
    print(f"\nCSV: {out}" + (f"  |  preds: {args.pred_out}" if args.pred_out else ""))


if __name__ == "__main__":
    main()