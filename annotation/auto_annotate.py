#!/usr/bin/env python3
"""
GroundSumm - VLM-based automated annotation.

Two modes:
  validation  - score entities listed in a human-judgment JSONL (uses CSV for grid paths)
  full        - score every entity in a grounded JSONL (uses --id-source, --video-dir, --frames-dir)
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("GROUNDSUMM_VLM", "qwen2.5vl:7b")

PROMPT_TEMPLATE = """You are a careful video annotator.

This image is a 4x4 grid of 16 frames sampled uniformly from a short video.

Question: Is "{surface}" visibly present somewhere in any of these 16 frames?

Decide based ONLY on what you can see in the frames, not on what is plausible.

Answer with ONE letter:
- P : Present. You can clearly see the entity in at least one frame.
- A : Absent. The entity is not visible in any frame.
- ? : Ambiguous. Occluded, out-of-focus, ambiguous reference, or an abstract concept that cannot be visualized.

Your answer (one letter only):"""


def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def parse_judgment(text):
    t = text.strip().upper()
    for c in t:
        if c in ("P", "A", "?"):
            return c
    if "PRESENT" in t: return "P"
    if "ABSENT"  in t: return "A"
    if "AMBIG"   in t: return "?"
    return "?"


def call_vlm(model, img_b64, surface, max_retries=3):
    prompt = PROMPT_TEMPLATE.format(surface=surface)
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10},
                },
                timeout=120,
            )
            r.raise_for_status()
            raw = r.json()["message"]["content"]
            return raw, parse_judgment(raw)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def load_csv_rows(path):
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["entity_id"]] = row
    return rows


def build_items_from_judgments(judgments_path, csv_path):
    with open(judgments_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(items)} human judgments from {judgments_path}")
    csv_rows = load_csv_rows(csv_path)
    print(f"Loaded {len(csv_rows)} rows from {csv_path}")
    for it in items:
        row = csv_rows.get(it["entity_id"])
        it["_grid_path"] = row["frame_grid_path"] if row else None
    return items


def build_items_from_grounded(grounded_path, id_source, frames_dir):
    with open(id_source) as f:
        video_ids = [line.split(None, 1)[0] if line.strip() else "" for line in f]
    items = []
    with open(grounded_path) as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            if idx >= len(video_ids):
                continue
            vid = video_ids[idx]
            grid_path = str(Path(frames_dir) / f"{vid}.png")
            if d.get("degenerate"):
                continue
            for ent in d.get("entities", []):
                if not isinstance(ent, dict):
                    continue
                items.append({
                    "entity_id": f"{d.get('summary_id', 'sum_' + str(idx))}__{ent.get('id', '?')}",
                    "summary_id": d.get("summary_id", ""),
                    "surface": ent.get("surface", ""),
                    "type": ent.get("type", ""),
                    "malformed": ent.get("malformed", False),
                    "blip_max": ent.get("grounding", {}).get("max", None),
                    "video_id": vid,
                    "_grid_path": grid_path,
                })
    print(f"Loaded {len(items)} entities from {grounded_path}")
    return items


def strip_private(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments")
    parser.add_argument("--csv")
    parser.add_argument("--grounded")
    parser.add_argument("--id-source")
    parser.add_argument("--video-dir")
    parser.add_argument("--frames-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.judgments:
        if not args.csv:
            print("ERROR: --csv required with --judgments.", file=sys.stderr); sys.exit(1)
        items = build_items_from_judgments(args.judgments, args.csv)
        mode = "validation"
    elif args.grounded:
        if not (args.id_source and args.frames_dir):
            print("ERROR: --id-source and --frames-dir required with --grounded.", file=sys.stderr); sys.exit(1)
        items = build_items_from_grounded(args.grounded, args.id_source, args.frames_dir)
        mode = "full"
    else:
        print("ERROR: pass --judgments+--csv OR --grounded+--id-source+--frames-dir.", file=sys.stderr); sys.exit(1)

    if args.limit:
        items = items[: args.limit]
        print(f"Limited to first {len(items)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model}  Mode: {mode}  Writing to: {out_path}")

    n_done = 0; n_errors = 0; agree = 0
    disagree_examples = []
    img_cache = {}

    with open(out_path, "w") as out_f:
        for it in tqdm(items, desc=f"VLM annotating ({mode})"):
            base = strip_private(it)
            grid_path = it.get("_grid_path")
            if not grid_path or not Path(grid_path).exists():
                out_f.write(json.dumps({**base, "_error": "grid_missing"}, ensure_ascii=False) + "\n")
                n_errors += 1
                continue
            if grid_path not in img_cache:
                try:
                    img_cache[grid_path] = load_image_b64(grid_path)
                except Exception as e:
                    out_f.write(json.dumps({**base, "_error": f"image_load_failed:{e}"}, ensure_ascii=False) + "\n")
                    n_errors += 1
                    continue
            try:
                raw, vlm_j = call_vlm(args.model, img_cache[grid_path], it["surface"])
            except Exception as e:
                out_f.write(json.dumps({**base, "_error": f"vlm_failed:{e}"}, ensure_ascii=False) + "\n")
                n_errors += 1
                continue

            human_j = it.get("judgment", None)
            record = {**base, "vlm_judgment": vlm_j, "vlm_raw": raw}
            if human_j is not None:
                record["agree_with_human"] = vlm_j == human_j
                if vlm_j == human_j:
                    agree += 1
                elif len(disagree_examples) < 20:
                    disagree_examples.append({
                        "entity_id": it.get("entity_id"), "surface": it.get("surface"),
                        "type": it.get("type"), "bucket": it.get("bucket"),
                        "blip_max": it.get("blip_max"),
                        "human": human_j, "vlm": vlm_j, "vlm_raw": raw,
                    })
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1

    print(f"\nDone. Processed: {n_done}, errors: {n_errors}")
    if mode == "validation" and n_done > 0:
        print(f"Agreement: {agree}/{n_done} ({100*agree/n_done:.1f}%)")
    if disagree_examples:
        print(f"\nFirst {len(disagree_examples)} disagreements:")
        for ex in disagree_examples[:10]:
            print(f"  [{ex['type']:12s} {ex['bucket']:9s} blip={ex['blip_max']}]  '{ex['surface']}'  human={ex['human']}  vlm={ex['vlm']}")


if __name__ == "__main__":
    main()
