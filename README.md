# GroundXum
Auditing Hallucination in Generative Video Summarization via Visual Grounding

*Author names withheld for double-blind review.*

GroundXum audits hallucination in generative video summarization by extracting the entities a summary names and testing whether each is visually observable in the source video, using BLIP-ITM and CLIP as grounding scorers and Qwen2.5-VL-7B as the visual-presence judge.

This repository accompanies two papers currently under review:

- **GroundXum** — GenAAI Workshop, ICPR 2026
- **Auditing the Auditor** — RRPR Workshop, ICPR 2026

Every table and figure in the RRPR paper regenerates from the cached outputs here, mostly on CPU in under 30 seconds.

## Repository structure

groundxum/
├── grounding/ground.py          # grounding operator: scores entities against frames
├── kg_extraction/extract.py     # extraction operator: entity graph from summary text
├── annotation/
│   ├── auto_annotate.py         # VLM judge over frame grids
│   └── make_grids.py            # render 4x4 keyframe grids
├── sweeps/                      # reproducibility sweep scripts
│   ├── _common.py
│   ├── sweep_threshold.py
│   ├── sweep_aggregation.py
│   ├── sweep_frames_cached.py
│   ├── sweep_judge_prompt.py
│   └── sweep_per_type.py
├── outputs/
│   ├── annotation/              # VLM judge verdicts and human judgment labels
│   └── results/                 # sweep CSVs
├── prompts/extract.md           # LLM entity extraction prompt
├── master_analysis.py           # reproduces all main paper tables
└── requirements.txt
## Installation

```bash
pip install -r requirements.txt

# Ollama models (required for judge and extractor)
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:14b-instruct
```

Hardware: single NVIDIA GPU with 24 GB VRAM. CPU-only sweeps (threshold, aggregation, frames, per-type) need no GPU.

## Reproducing the RRPR paper

All commands run from the repository root.

| Result | Script | CPU/GPU |
|---|---|---|
| Table 2 — threshold stability | `sweep_threshold.py` | CPU |
| Table 3 — aggregation sensitivity | `sweep_aggregation.py` | CPU |
| Fig. 1 + Table 4 — keyframe sensitivity | `sweep_frames_cached.py` | CPU |
| Table 5 — judge-prompt kappa | `sweep_judge_prompt.py` | GPU |
| Tables 6–8 — per-type human agreement | `sweep_per_type.py` | CPU |
| All main paper numbers (Parts 1–7) | `master_analysis.py` | CPU |

```bash
# Table 2
python sweeps/sweep_threshold.py --scorer clip \
  --entry "MM=outputs/grounding/youcook2/mm_blip_clip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl=outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl" \
  --entry "TXT=outputs/grounding/youcook2/blip_base_clip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl=outputs/annotation/judegements/youcook2_blip_base_judgments.jsonl" \
  --grid 60 --out outputs/results/sweep_threshold_youcook2_clip.csv

# Table 3
python sweeps/sweep_aggregation.py \
  --entry "MM=outputs/grounding/youcook2/mm_blip_clip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl" \
  --entry "TXT=outputs/grounding/youcook2/blip_base_clip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl" \
  --grid 60 --out outputs/results/sweep_aggregation_youcook2_clip.csv

# Fig. 1 + Table 4
python sweeps/sweep_frames_cached.py --scorer clip --thresh 0.20 \
  --entry "MM=outputs/grounding/youcook2/mm_blip_clip.jsonl=outputs/annotation/vlm/youcook2_mm_blip_full.jsonl" \
  --entry "TXT=outputs/grounding/youcook2/blip_base_clip.jsonl=outputs/annotation/vlm/youcook2_blip_base_full.jsonl" \
  --ks 1,2,4,8,16 --ref-k 16 --out outputs/results/sweep_frames_cached_youcook2_clip.csv

# Table 5 (GPU required)
python sweeps/sweep_judge_prompt.py \
  --judgments outputs/annotation/judegements/youcook2_mm_blip_judgments.jsonl \
  --csv outputs/annotation/youcook2_mm_blip.csv \
  --variants v0_original,v1_reworded,v2_cot,v3_no_ambiguous,v4_order_swap \
  --out outputs/results/sweep_judge_prompt_youcook2_mm.csv \
  --pred-out outputs/results/sweep_judge_prompt_youcook2_mm_preds.jsonl

# Tables 6–8
python sweeps/sweep_per_type.py --datasets youcook2,videoxum --models mm,blip \
  --out outputs/results/sweep_per_type.csv

# All main paper numbers
python master_analysis.py
```

## Large files

Frame grids and model checkpoints are hosted on an anonymized archive for review:

**[Download artifacts](https://zenodo.org/records/21227987?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImU5NjhlNDIzLWI5NTgtNDhlNC05YTE0LTE2NTBlOTE1NjUyNiIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjVhZDU1OGM5MWIxMzQ5ZGQ4YTllMDY1NWU4OGEzOSJ9.d4lOLDfKsF87-OwPi2wrxfrMaG7g0uca91V-A01wFxRCWXJGa6pdiYc1I1sMSry5syhI_AiY17k0KJwG4TW_tg)**

Contents:

- `frames_youcook2.zip` — 457 frame grid PNGs for YouCook2 (567 MB); needed for judge-prompt sweep
- `frames_videoxum.zip` — 4000 frame grid PNGs for VideoXum (4.5 GB); needed for judge-prompt sweep
- `mm/` — multimodal summarizer checkpoint (under separate review)
- `txt/` — text-only baseline checkpoint

Extract frames to `outputs/annotation/frames/{youcook2,videoxum}/` and place checkpoints under `checkpoints/`.

Source videos: YouCook2 · VideoXum

## Audited systems

- **MM** — multimodal generative summarizer (cross-modal attention; subject of separate work under review)
- **TXT** — text-only baseline on the same backbone and training data

## Citation

*Citation details withheld for double-blind review; full citation will be added upon acceptance.*