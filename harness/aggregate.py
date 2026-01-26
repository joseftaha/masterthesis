"""Aggregate per-policy ``*__details.csv`` and ``*__switches.csv`` files
into a per-trace ``summaries.csv``.

Counts are derived from ``service_mode``:

  warm_count = sum(service_mode in {"hot_warm", "switched_warm"})
  cold_count = sum(service_mode == "cold")

Two derived columns are added on top of the core latency / count
fields:

  throughput_req_s, throughput_tok_s
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

CORE_COLS = [
    "policy", "trace_name", "n_requests",
    "avg_ttft", "p50_ttft", "p95_ttft",
    "avg_e2e", "p50_e2e", "p95_e2e",
    "cold_count", "warm_count", "switch_count", "makespan",
    "throughput_req_s", "throughput_tok_s",
]


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    f = int(math.floor(k))
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _is_finite(x: str) -> bool:
    if x in ("", None):
        return False
    try:
        v = float(x)
    except ValueError:
        return False
    return not math.isnan(v) and not math.isinf(v)


def summarize(details_csv: Path, switches_csv: Path, *, trace: str,
              policy: str) -> dict:
    rows = list(csv.DictReader(details_csv.open()))
    ttfts = [float(r["observed_ttft"]) for r in rows if _is_finite(r.get("observed_ttft", ""))]
    e2es = [float(r["observed_e2e"]) for r in rows if _is_finite(r.get("observed_e2e", ""))]
    arrivals = [float(r["arrival"]) for r in rows if _is_finite(r.get("arrival", ""))]
    finishes = [float(r["finish"]) for r in rows if _is_finite(r.get("finish", ""))]
    out_tokens = sum(int(r.get("output_tokens") or 0) for r in rows)

    cold = sum(1 for r in rows if r.get("service_mode") == "cold")
    warm = sum(1 for r in rows if r.get("service_mode") in ("hot_warm", "switched_warm"))

    n_switches = 0
    if switches_csv.exists():
        sw_rows = list(csv.DictReader(switches_csv.open()))
        # Count only real switches (between two different models). cold loads
        # in the same allocation also produce a switch row but are not
        # "switches" in the policy sense.
        for s in sw_rows:
            fr = (s.get("from_model") or "").strip()
            to = (s.get("to_model") or "").strip()
            if fr and to and fr != to:
                n_switches += 1

    makespan = (max(finishes) - min(arrivals)) if finishes and arrivals else 0.0

    return {
        "policy": policy,
        "trace_name": trace,
        "n_requests": len(rows),
        "avg_ttft": round(statistics.mean(ttfts), 4) if ttfts else float("nan"),
        "p50_ttft": round(_percentile(ttfts, 0.50), 4),
        "p95_ttft": round(_percentile(ttfts, 0.95), 4),
        "avg_e2e": round(statistics.mean(e2es), 4) if e2es else float("nan"),
        "p50_e2e": round(_percentile(e2es, 0.50), 4),
        "p95_e2e": round(_percentile(e2es, 0.95), 4),
        "cold_count": cold,
        "warm_count": warm,
        "switch_count": n_switches,
        "makespan": round(makespan, 4),
        "throughput_req_s": round(len(rows) / makespan, 4) if makespan else float("nan"),
        "throughput_tok_s": round(out_tokens / makespan, 4) if makespan else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="runs/<trace>/ — the per-trace output directory")
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()

    out: list[dict] = []
    for det in sorted(args.runs_dir.glob("*__details.csv")):
        policy = det.name.removesuffix("__details.csv")
        sw = args.runs_dir / f"{policy}__switches.csv"
        out.append(summarize(det, sw, trace=args.trace, policy=policy))

    if not out:
        raise SystemExit(f"no details CSVs in {args.runs_dir}")

    summary = args.runs_dir / "summaries.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CORE_COLS)
        w.writeheader()
        for row in out:
            w.writerow({k: row.get(k, "") for k in CORE_COLS})
    print(f"wrote {summary} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
