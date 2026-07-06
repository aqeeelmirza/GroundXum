#!/usr/bin/env python3
"""
GroundSumm — KG extraction script.

Reads a file of generated summaries (one per line) and extracts a per-summary
knowledge graph via Claude Sonnet, using the prompt in prompts/extract.md.

Outputs one JSON object per line (JSONL) to the chosen output file.

Usage:
    python kg_extraction/extract.py \
        --input outputs/summaries/youcook2/mm_blip.txt \
        --dataset youcook2 \
        --output outputs/kg/youcook2/mm_blip.jsonl \
        --limit 10        # smoke-test on first 10 lines; omit for full run

Env:
    ANTHROPIC_API_KEY  — required
    GROUNDSUMM_MODEL   — optional, defaults to claude-sonnet-4-5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from anthropic import Anthropic, APIError, RateLimitError
from tqdm import tqdm
import requests  # for ollama backend


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extract.md"
DEFAULT_MODEL = os.environ.get("GROUNDSUMM_MODEL", "claude-sonnet-4-6")
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("GROUNDSUMM_OLLAMA_MODEL", "qwen2.5:14b-instruct")  # seconds, doubled each retry


def load_prompt() -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_user_message(prompt: str, summary: str, summary_id: str, dataset: str) -> str:
    return (
        f"{prompt}\n\n"
        f"---\n\n"
        f"Now extract the KG for the following input. "
        f"Return ONLY the JSON object, no prose, no markdown fences.\n\n"
        f'summary_id: "{summary_id}"\n'
        f'dataset: "{dataset}"\n'
        f'summary: "{summary}"\n'
    )



def call_ollama(model: str, user_message: str) -> str:
    """Call a local Ollama server."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": user_message}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 2048},
                    "format": "json",
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Ollama error (attempt {attempt}): {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")


def call_model(client: Anthropic, model: str, user_message: str) -> str:
    """Call the model with retries on rate-limit and transient errors."""
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": user_message}],
            )
            # Get text from content blocks of type "text"
            return "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(delay)
            delay *= 2
        except APIError as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  API error (attempt {attempt}): {e}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")


def parse_json(text: str):
    """Try to parse JSON. If the model wrapped it in code fences, strip them."""
    text = text.strip()
    if text.startswith("```"):
        # strip the leading fence
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        # strip the trailing fence
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to summaries .txt (one per line)")
    parser.add_argument("--dataset", required=True, choices=["youcook2", "videoxum"])
    parser.add_argument("--output", required=True, help="path to output .jsonl")
    parser.add_argument("--limit", type=int, default=None, help="process only first N summaries (smoke test)")
    parser.add_argument("--model", default=None, help="model name; defaults depend on backend")
    parser.add_argument("--backend", choices=["anthropic", "ollama"], default="anthropic")
    parser.add_argument("--id-prefix", default=None, help="prefix for summary IDs; defaults to basename of input")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if args.backend == "anthropic" and not api_key:
        print("ERROR: set ANTHROPIC_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    # Resolve model default based on backend
    if args.model is None:
        args.model = DEFAULT_OLLAMA_MODEL if args.backend == "ollama" else DEFAULT_MODEL

    client = Anthropic(api_key=api_key) if args.backend == "anthropic" else None
    prompt = load_prompt()

    # Make output dir
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read summaries
    with open(args.input, "r", encoding="utf-8") as f:
        summaries = [line.rstrip("\n") for line in f]
    if args.limit is not None:
        summaries = summaries[: args.limit]

    id_prefix = args.id_prefix or Path(args.input).stem
    print(f"Loaded {len(summaries)} summaries from {args.input}")
    print(f"Model: {args.model}, dataset: {args.dataset}")
    print(f"Writing to: {args.output}")

    errors = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for idx, summary in enumerate(tqdm(summaries, desc="Extracting")):
            summary_id = f"{id_prefix}_{idx:05d}"
            # Empty line edge case
            if not summary.strip():
                record = {
                    "summary_id": summary_id,
                    "dataset": args.dataset,
                    "summary_text": "",
                    "degenerate": True,
                    "entities": [],
                    "triples": [],
                    "_error": "empty_input",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            user_msg = build_user_message(prompt, summary, summary_id, args.dataset)
            try:
                raw = call_ollama(args.model, user_msg) if args.backend == 'ollama' else call_model(client, args.model, user_msg)
                parsed = parse_json(raw)
                # Add the summary text and ensure ID/dataset present
                parsed["summary_text"] = summary
                parsed.setdefault("summary_id", summary_id)
                parsed.setdefault("dataset", args.dataset)
                out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            except Exception as e:
                errors += 1
                fallback = {
                    "summary_id": summary_id,
                    "dataset": args.dataset,
                    "summary_text": summary,
                    "degenerate": None,
                    "entities": [],
                    "triples": [],
                    "_error": str(e),
                    "_raw": raw if "raw" in locals() else None,
                }
                out_f.write(json.dumps(fallback, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone. Errors: {errors}/{len(summaries)}")


if __name__ == "__main__":
    main()
