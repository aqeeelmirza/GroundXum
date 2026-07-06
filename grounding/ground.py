#!/usr/bin/env python3
"""
GroundSumm - grounding pipeline.

For each entity extracted from a summary, compute its grounding score against
N evenly-spaced frames from the source video, using either BLIP-ITM or CLIP.

Per entity, store:
  - max score across frames
  - argmax frame index
  - mean score
  - all per-frame scores

Output: JSONL, one record per summary, augmented with grounding info per entity.

Usage:
    python grounding/ground.py \\
        --kg outputs/kg/youcook2/mm_blip.jsonl \\
        --dataset youcook2 \\
        --id-source /path/to/test/tran.tok.txt \\
        --video-dir data/videos/youcook2 \\
        --scorer blip \\
        --frames 16 \\
        --output outputs/grounding/youcook2/mm_blip_blip-itm.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Video frame extraction
import cv2


# ----------------------------------------------------------------------
# Frame sampling
# ----------------------------------------------------------------------

def sample_frames(video_path: str, n_frames: int) -> list:
    """Sample n_frames evenly spaced frames from a video. Returns list of PIL images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    n = min(n_frames, total)
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


# ----------------------------------------------------------------------
# Scorers
# ----------------------------------------------------------------------

class BlipITMScorer:
    """BLIP image-text matching head. Returns per-(entity, frame) ITM probability."""

    def __init__(self, device: str = "cuda"):
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        self.device = device
        model_name = "Salesforce/blip-itm-base-coco"
        print(f"Loading {model_name}...", file=sys.stderr)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def score(self, frames: list, texts: list) -> list:
        """Returns scores[t][f] for each text t and frame f."""
        scores = [[0.0] * len(frames) for _ in texts]
        for t_idx, text in enumerate(texts):
            inputs = self.processor(
                images=frames, text=[text] * len(frames),
                return_tensors="pt", padding=True,
            ).to(self.device)
            out = self.model(**inputs)
            # itm_score is logits over [no_match, match]; softmax then take match prob
            probs = torch.softmax(out.itm_score, dim=-1)[:, 1].cpu().tolist()
            scores[t_idx] = probs
        return scores


class ClipScorer:
    """CLIP cosine similarity between text and image embeddings."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        model_name = "openai/clip-vit-base-patch32"
        print(f"Loading {model_name}...", file=sys.stderr)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def score(self, frames: list, texts: list) -> list:
        """Returns scores[t][f] for each text t and frame f. Cosine similarity in [-1, 1]."""
        # Encode images once
        image_inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        img_feats = self.model.get_image_features(**image_inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # Encode all texts
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        txt_feats = self.model.get_text_features(**text_inputs)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        # Cosine similarity: (T, D) @ (D, F) -> (T, F)
        sims = (txt_feats @ img_feats.T).cpu().tolist()
        return sims


# ----------------------------------------------------------------------
# Video index
# ----------------------------------------------------------------------

def build_video_index(video_dir: Path, extensions: tuple) -> dict:
    """Index videos by stem (filename without extension)."""
    index = {}
    for ext in extensions:
        for p in video_dir.rglob(f"*{ext}"):
            index[p.stem] = str(p)
    return index


def load_video_ids(id_source: str) -> list:
    """Read the first whitespace token of each line."""
    ids = []
    with open(id_source) as f:
        for line in f:
            parts = line.split(None, 1)
            ids.append(parts[0] if parts else "")
    return ids


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def entity_to_text(entity) -> str:
    """Convert an entity record to a text prompt for scoring. Handles malformed records."""
    if isinstance(entity, dict):
        surface = entity.get("surface", "")
    elif isinstance(entity, str):
        surface = entity
    else:
        surface = ""
    surface = (surface or "").strip()
    return surface if surface else "object"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", required=True, help="path to KG JSONL")
    parser.add_argument("--dataset", required=True, choices=["youcook2", "videoxum"])
    parser.add_argument("--id-source", required=True, help="test-set source file (first token = video ID)")
    parser.add_argument("--video-dir", required=True, help="dir containing source videos")
    parser.add_argument("--scorer", required=True, choices=["blip", "clip"])
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="process first N (smoke test)")
    parser.add_argument("--skip-malformed", action="store_true", help="do not score malformed-flagged entities")
    args = parser.parse_args()

    # Load ID mapping
    video_ids = load_video_ids(args.id_source)
    extensions = (".mp4", ".mkv", ".webm", ".avi")
    video_index = build_video_index(Path(args.video_dir), extensions)
    print(f"Loaded {len(video_ids)} test-set IDs, indexed {len(video_index)} videos")

    # Init scorer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = BlipITMScorer(device) if args.scorer == "blip" else ClipScorer(device)
    print(f"Scorer: {args.scorer} on {device}")

    # Load KG file
    with open(args.kg) as f:
        kg_lines = [json.loads(line) for line in f]
    if args.limit:
        kg_lines = kg_lines[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_missing_video = 0
    skipped_degenerate = 0
    skipped_no_entities = 0

    with open(out_path, "w") as out_f:
        for idx, record in enumerate(tqdm(kg_lines, desc="Grounding")):
            # ID lookup by position
            if idx >= len(video_ids):
                record["_error"] = "no_video_id_for_index"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            vid = video_ids[idx]
            record["video_id"] = vid

            if vid not in video_index:
                skipped_missing_video += 1
                record["_error"] = f"video_not_found:{vid}"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            entities = record.get("entities", [])
            if record.get("degenerate") or not entities:
                if record.get("degenerate"):
                    skipped_degenerate += 1
                else:
                    skipped_no_entities += 1
                record["video_path"] = video_index[vid]
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            video_path = video_index[vid]
            record["video_path"] = video_path

            # Extract frames
            try:
                frames = sample_frames(video_path, args.frames)
            except Exception as e:
                record["_error"] = f"frame_extraction_failed:{e}"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if not frames:
                record["_error"] = "no_frames_extracted"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Decide which entities to score
            score_entities = [
                e for e in entities
                if isinstance(e, dict) and not (args.skip_malformed and e.get("malformed"))
            ]
            texts = [entity_to_text(e) for e in score_entities]

            if not texts:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            try:
                scores = scorer.score(frames, texts)
            except Exception as e:
                record["_error"] = f"scoring_failed:{e}"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            # Attach grounding info per scored entity
            score_idx = 0
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                if args.skip_malformed and ent.get("malformed"):
                    ent["grounding"] = {"skipped": "malformed"}
                    continue
                row = scores[score_idx]
                score_idx += 1
                max_s = max(row)
                argmax = row.index(max_s)
                mean_s = sum(row) / len(row)
                ent["grounding"] = {
                    "scorer": args.scorer,
                    "n_frames": len(frames),
                    "max": round(float(max_s), 4),
                    "argmax_frame": argmax,
                    "mean": round(float(mean_s), 4),
                    "scores": [round(float(s), 4) for s in row],
                }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone.")
    print(f"  Skipped (video missing):  {skipped_missing_video}")
    print(f"  Skipped (degenerate):     {skipped_degenerate}")
    print(f"  Skipped (no entities):    {skipped_no_entities}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
