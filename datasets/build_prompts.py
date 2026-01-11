#!/usr/bin/env python3
"""Build deterministic prompt pools used by the trace builder.

Outputs two JSONL files into ``slurm/datasets/cache/``:

  * ``sharegpt_pool.jsonl`` — short conversational user prompts for the
    interactive (Experiment A) traces. Source:
    anon8231489123/ShareGPT_Vicuna_unfiltered (research-only).
  * ``cnn_dailymail_pool.jsonl`` — news article + reference summary pairs
    for the offline batch (Experiment B) traces. Source:
    abisee/cnn_dailymail (3.0.0 / test split).

Each line is ``{"prompt": str, "input_tokens": int, "ref_output_tokens": int}``.
Tokenisation uses the Meta-Llama-3 tokenizer so token counts match the
serving runtime exactly. Determinism is guaranteed by the explicit
``--seed`` and a stable sort before sampling, plus an explicit HF
``revision`` pin per dataset (set in ``DATASET_REVISIONS`` below).

The ShareGPT cap is intentionally loose (1500 input tokens) so that the
"about 1% of prompts exceed 900 input tokens" tail described in chapter
8 is preserved. CNN/DailyMail bounds match the chapter's reported P99
of ~2543 input tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

# Pinned upstream HF dataset revisions. Update these intentionally —
# bumping a revision changes the wire-level prompts and therefore the
# trace contents, even if the sampling seed is unchanged.
DATASET_REVISIONS = {
    "anon8231489123/ShareGPT_Vicuna_unfiltered": "main",
    "abisee/cnn_dailymail": "main",
}


def _load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def build_sharegpt(tok, n: int, seed: int, min_in=20, max_in=1500) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
        revision=DATASET_REVISIONS["anon8231489123/ShareGPT_Vicuna_unfiltered"],
    )
    items: list[dict] = []
    for row in ds:
        conv = row.get("conversations") or []
        if not conv:
            continue
        first = conv[0]
        if first.get("from") not in ("human", "user"):
            continue
        text = (first.get("value") or "").strip()
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        L = len(ids)
        if L < min_in or L > max_in:
            continue
        items.append({"prompt": text, "input_tokens": int(L)})
    items.sort(key=lambda r: (r["input_tokens"], r["prompt"]))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def build_cnn(tok, n: int, seed: int, min_in=400, max_in=2600) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split="test",
        revision=DATASET_REVISIONS["abisee/cnn_dailymail"],
    )
    template = (
        "Summarize the following news article in 2-3 sentences.\n\n"
        "Article:\n{article}\n\nSummary:"
    )
    items: list[dict] = []
    for row in ds:
        article = (row.get("article") or "").strip()
        ref = (row.get("highlights") or "").strip()
        if not article or not ref:
            continue
        prompt = template.format(article=article)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        L = len(ids)
        if L < min_in or L > max_in:
            continue
        ref_out = len(tok(ref, add_special_tokens=False)["input_ids"])
        items.append({"prompt": prompt, "input_tokens": int(L),
                      "ref_output_tokens": int(ref_out)})
    items.sort(key=lambda r: (r["input_tokens"], r["prompt"][:64]))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def write_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items):>5} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--n_sharegpt", type=int, default=4000)
    ap.add_argument("--n_cnn", type=int, default=4000)
    ap.add_argument("--tokenizer", default=os.environ.get(
        "PROMPT_TOKENIZER", "meta-llama/Meta-Llama-3-8B-Instruct"))
    args = ap.parse_args()

    tok = _load_tokenizer(args.tokenizer)
    write_jsonl(CACHE / "sharegpt_pool.jsonl",
                build_sharegpt(tok, args.n_sharegpt, args.seed))
    write_jsonl(CACHE / "cnn_dailymail_pool.jsonl",
                build_cnn(tok, args.n_cnn, args.seed))
    # Emit the revision pins next to the pools so any audit trail is local.
    (CACHE / "dataset_revisions.json").write_text(
        json.dumps(DATASET_REVISIONS, indent=2))
    print(f"pinned dataset revisions -> {CACHE / 'dataset_revisions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
