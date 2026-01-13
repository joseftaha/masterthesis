#!/usr/bin/env python3
"""Deterministically generate the seven request-level traces used by
Experiments A and B (chapter 8 of the thesis).

Each trace is a CSV with columns

    req_id, ts, model, input_tokens, output_tokens, prompt_idx, prompt_pool

where ``prompt_idx`` indexes into the corresponding prompt pool JSONL
emitted by ``datasets/build_prompts.py``. Submitting the trace therefore
fully determines the wire-level workload — no run-time prompt sampling.

Trace shapes follow Sec.~8.2 / 8.3 of the thesis:

  interactive_sparse_tail        80/20 8B/70B, mean iat ~1.95 s
  interactive_bursty_tail        80/20 8B/70B, 70B in short bursts
  interactive_popularity_shift   3-phase drift 80/20 -> 50/50 -> 30/70
  interactive_balanced_mix       50/50 with run-len 1..3, mean iat 1.50 s
  offline_batch_c1_primary       100% 8B, single-request waves
  offline_batch_c4_primary       100% 8B, 4-request waves
  offline_batch_c8_primary       100% 8B, 8-request waves
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POOL_DIR = ROOT.parent / "datasets" / "cache"
PRIMARY = "Meta-Llama-3-8B-Instruct"
SECONDARY = "Meta-Llama-3-70B-Instruct"
N_PER_TRACE = 1000


def _load_pool(name: str) -> list[dict]:
    p = POOL_DIR / name
    if not p.exists():
        raise FileNotFoundError(
            f"missing prompt pool {p} - run datasets/build_prompts.py first")
    with p.open() as f:
        return [json.loads(line) for line in f]


def _sample_pool(pool: list[dict], rng: random.Random) -> tuple[int, dict]:
    idx = rng.randrange(len(pool))
    return idx, pool[idx]


def _output_tokens_interactive(rng: random.Random, model: str) -> int:
    # Lognormal centered ~412 (8B) / ~429 (70B), capped at 512 (decoding cap).
    mean = 6.02 if model == PRIMARY else 6.06
    sd = 0.55
    v = int(round(math.exp(rng.gauss(mean, sd))))
    return max(8, min(512, v))


def _output_tokens_batch(rng: random.Random, ref_out: int) -> int:
    # Reference summary length, jittered, capped at 128 (decoding cap).
    v = int(round(rng.gauss(ref_out if ref_out > 0 else 56, 8)))
    return max(16, min(128, v))


def _emit(path: Path, rows: list[dict]) -> None:
    cols = ["req_id", "ts", "model", "input_tokens",
            "output_tokens", "prompt_idx", "prompt_pool"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows):>5} rows -> {path}")


def _interactive_iat(rng: random.Random, mean: float) -> float:
    # Exponential inter-arrival with the requested mean.
    return rng.expovariate(1.0 / mean)


def _build_run_lengths(rng: random.Random, total: int, mean_run: float,
                       p_secondary: float) -> list[str]:
    """Build a model-id sequence of length ``total`` made of runs of the
    same model with geometric run length around ``mean_run`` and
    long-run secondary probability ``p_secondary``."""
    seq: list[str] = []
    cur = PRIMARY if rng.random() > p_secondary else SECONDARY
    while len(seq) < total:
        run = max(1, int(rng.expovariate(1.0 / mean_run)))
        seq.extend([cur] * run)
        cur = SECONDARY if cur == PRIMARY else PRIMARY
        # bias to honour the requested marginal model ratio
        if rng.random() < (p_secondary if cur == SECONDARY else (1 - p_secondary)):
            pass  # keep flip
        else:
            cur = SECONDARY if cur == PRIMARY else PRIMARY
    return seq[:total]


def _enforce_marginal(rng: random.Random, rows: list[dict],
                      target_secondary: int) -> None:
    """Deterministic post-pass: nudge the most recent rows of the wrong
    model to the right model so the global 8B/70B marginal lands
    exactly on the chapter's pinned ratio. The IAT pattern is
    preserved; only the model id (and its model-conditioned output
    token sample) flips."""
    sec = sum(1 for r in rows if r["model"] == SECONDARY)
    deficit = target_secondary - sec
    if deficit == 0:
        return
    if deficit > 0:
        # need more 70B: flip primary -> secondary, taking from the END
        for r in reversed(rows):
            if deficit == 0:
                break
            if r["model"] == PRIMARY:
                r["model"] = SECONDARY
                r["output_tokens"] = _output_tokens_interactive(rng, SECONDARY)
                deficit -= 1
    else:
        # need fewer 70B: flip secondary -> primary, taking from the END
        deficit = -deficit
        for r in reversed(rows):
            if deficit == 0:
                break
            if r["model"] == SECONDARY:
                r["model"] = PRIMARY
                r["output_tokens"] = _output_tokens_interactive(rng, PRIMARY)
                deficit -= 1


def trace_sparse_tail(rng: random.Random, pool: list[dict]) -> list[dict]:
    rows = []
    t = 0.0
    for i in range(N_PER_TRACE):
        is_secondary = rng.random() < 0.20
        model = SECONDARY if is_secondary else PRIMARY
        if i > 0:
            iat = _interactive_iat(rng, mean=1.95)
            t += iat
        idx, p = _sample_pool(pool, rng)
        rows.append({
            "req_id": i, "ts": round(t, 3), "model": model,
            "input_tokens": p["input_tokens"],
            "output_tokens": _output_tokens_interactive(rng, model),
            "prompt_idx": idx, "prompt_pool": "sharegpt_pool.jsonl",
        })
    _enforce_marginal(rng, rows, target_secondary=N_PER_TRACE // 5)
    return rows


def trace_bursty_tail(rng: random.Random, pool: list[dict]) -> list[dict]:
    """8B background runs interleaved with short 70B bursts. Burst sizes
    are sampled from the requested distribution but rebalanced so the
    overall 80/20 marginal holds within a single request."""
    rows: list[dict] = []
    t = 0.0
    i = 0
    target_secondary = N_PER_TRACE // 5  # 200 of 1000 = 20%
    secondary_emitted = 0
    while i < N_PER_TRACE:
        # Long 8B background segment, sized so the running 80/20 ratio
        # stays close to the target.
        remaining = N_PER_TRACE - i
        secondary_budget = target_secondary - secondary_emitted
        primary_budget = remaining - secondary_budget
        bg_max = min(80, max(35, primary_budget // 4 if secondary_budget > 0 else primary_budget))
        bg = min(rng.randint(35, max(35, bg_max)), primary_budget) if primary_budget > 0 else 0
        for _ in range(bg):
            if rows:
                t += rng.expovariate(1.0 / 2.07)
            idx, p = _sample_pool(pool, rng)
            rows.append({"req_id": i, "ts": round(t, 3), "model": PRIMARY,
                         "input_tokens": p["input_tokens"],
                         "output_tokens": _output_tokens_interactive(rng, PRIMARY),
                         "prompt_idx": idx, "prompt_pool": "sharegpt_pool.jsonl"})
            i += 1
            if i >= N_PER_TRACE:
                break
        if i >= N_PER_TRACE:
            break
        # Short 70B burst, capped by remaining secondary budget so the
        # final 80/20 marginal lands within ±1 request.
        secondary_budget = target_secondary - secondary_emitted
        if secondary_budget <= 0:
            continue
        burst = min(rng.randint(5, 14), secondary_budget, N_PER_TRACE - i)
        for _ in range(burst):
            t += rng.expovariate(1.0 / 0.27)
            idx, p = _sample_pool(pool, rng)
            rows.append({"req_id": i, "ts": round(t, 3), "model": SECONDARY,
                         "input_tokens": p["input_tokens"],
                         "output_tokens": _output_tokens_interactive(rng, SECONDARY),
                         "prompt_idx": idx, "prompt_pool": "sharegpt_pool.jsonl"})
            i += 1
            secondary_emitted += 1
    _enforce_marginal(rng, rows, target_secondary=target_secondary)
    return rows


def trace_popularity_shift(rng: random.Random, pool: list[dict]) -> list[dict]:
    """Three-phase drift 80/20 -> 50/50 -> 30/70 (chapter pins boundaries
    at 400, 600). Each phase's marginal is enforced exactly via a
    per-phase post-pass so the chapter's pinned ratios are reproducible
    regardless of the RNG path."""
    plan = [(0, 400, 0.20), (400, 600, 0.50), (600, 1000, 0.70)]
    rows = []
    t = 0.0
    for lo, hi, p_sec in plan:
        seg: list[dict] = []
        for i in range(lo, hi):
            model = SECONDARY if rng.random() < p_sec else PRIMARY
            if rows or seg:
                t += rng.expovariate(1.0 / 1.62)
            idx, p = _sample_pool(pool, rng)
            seg.append({"req_id": i, "ts": round(t, 3), "model": model,
                        "input_tokens": p["input_tokens"],
                        "output_tokens": _output_tokens_interactive(rng, model),
                        "prompt_idx": idx, "prompt_pool": "sharegpt_pool.jsonl"})
        _enforce_marginal(rng, seg, target_secondary=int(round((hi - lo) * p_sec)))
        rows.extend(seg)
    return rows


def trace_balanced_mix(rng: random.Random, pool: list[dict]) -> list[dict]:
    rows = []
    t = 0.0
    cur = PRIMARY
    i = 0
    while i < N_PER_TRACE:
        run = rng.randint(1, 3)
        for _ in range(run):
            if i >= N_PER_TRACE:
                break
            if rows:
                t += rng.expovariate(1.0 / 1.50)
            idx, p = _sample_pool(pool, rng)
            rows.append({"req_id": i, "ts": round(t, 3), "model": cur,
                         "input_tokens": p["input_tokens"],
                         "output_tokens": _output_tokens_interactive(rng, cur),
                         "prompt_idx": idx, "prompt_pool": "sharegpt_pool.jsonl"})
            i += 1
        cur = SECONDARY if cur == PRIMARY else PRIMARY
    return rows


def trace_batch(rng: random.Random, pool: list[dict],
                concurrency: int, mean_gap: float) -> list[dict]:
    rows = []
    t = 0.0
    n_waves = math.ceil(N_PER_TRACE / concurrency)
    rid = 0
    for w in range(n_waves):
        for _ in range(concurrency):
            if rid >= N_PER_TRACE:
                break
            idx, p = _sample_pool(pool, rng)
            rows.append({"req_id": rid, "ts": round(t, 3), "model": PRIMARY,
                         "input_tokens": p["input_tokens"],
                         "output_tokens": _output_tokens_batch(
                             rng, p.get("ref_output_tokens", 56)),
                         "prompt_idx": idx, "prompt_pool": "cnn_dailymail_pool.jsonl"})
            rid += 1
        t += rng.uniform(mean_gap * 0.79, mean_gap * 1.16)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20240501)
    ap.add_argument("--out", type=Path, default=ROOT)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sgpt = _load_pool("sharegpt_pool.jsonl")
    cnn = _load_pool("cnn_dailymail_pool.jsonl")

    builds = {
        "interactive_sparse_tail":      lambda r: trace_sparse_tail(r, sgpt),
        "interactive_bursty_tail":      lambda r: trace_bursty_tail(r, sgpt),
        "interactive_popularity_shift": lambda r: trace_popularity_shift(r, sgpt),
        "interactive_balanced_mix":     lambda r: trace_balanced_mix(r, sgpt),
        "offline_batch_c1_primary":     lambda r: trace_batch(r, cnn, 1, 8.47),
        "offline_batch_c4_primary":     lambda r: trace_batch(r, cnn, 4, 8.25),
        "offline_batch_c8_primary":     lambda r: trace_batch(r, cnn, 8, 8.21),
    }

    stats = {}
    for name, build in builds.items():
        # one independent stream per trace, derived from the master seed
        sub_seed = args.seed + sum(ord(c) for c in name)
        rows = build(random.Random(sub_seed))
        _emit(args.out / f"{name}.csv", rows)
        n = len(rows)
        stats[name] = {
            "rows": n,
            "horizon_s": rows[-1]["ts"] if rows else 0.0,
            "model_ratio": {
                m: sum(1 for r in rows if r["model"] == m) / n
                for m in {r["model"] for r in rows}
            },
        }
    with (args.out / "trace_stats.generated.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nstats -> {args.out / 'trace_stats.generated.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
