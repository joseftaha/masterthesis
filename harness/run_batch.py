"""Experiment B runner — replays one offline batch trace at a fixed
target concurrency against either:

  * the warm hot path (8B server kept resident for the whole trace), or
  * the cold baseline (server stopped between waves so each wave pays
    one cold load).

Concurrency is enforced by a semaphore — clients dispatch as soon as
their trace timestamp arrives *and* a slot is free. The trace itself
already has a wave shape that nominally matches the target concurrency
(see ``traces/build_traces.py``); the semaphore guarantees the actual
in-flight count never exceeds the configured cap, so the server-side
batching behaviour depends only on the runtime.

The per-request CSV uses the same schema as Experiment A so that one
plotting script can draw frontiers and stage decompositions from both
sets of runs:

    columns: policy, trace_name, req_id, model, input_tokens, output_tokens,
             arrival, decision_time, start, first_token, finish,
             observed_ttft, observed_e2e, service_mode,
             hot_model_before, hot_model_after,
             switch_cost, buffer_wait, resource_wait, cold_lane, note

For Experiment B, ``policy`` is set to ``warm_c{C}`` or ``cold_c{C}``
to identify the operating point on the throughput-latency frontier.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from client import submit  # noqa: E402
from vllm_proc import ServeSpec, VLLMServer  # noqa: E402

PRIMARY = "Meta-Llama-3-8B-Instruct"


def _spec(cfg: dict) -> ServeSpec:
    m = cfg["models"]["primary"]
    repo = m.get("quant_repo_override") or m["hf_repo"]
    quant = m["quantization"] if m["quantization"] not in ("bf16", None) else None
    return ServeSpec(
        name=m["name"], model_path=repo,
        tensor_parallel=int(m["tensor_parallel_size"]),
        quantization=quant,
        gpu_memory_utilization=float(m["gpu_memory_utilization"]),
        max_model_len=int(m["max_model_len"]),
        seed=int(cfg["decoding"]["seed"]), extra_args=[],
    )


DETAIL_COLS = [
    "policy", "trace_name", "req_id", "model",
    "input_tokens", "output_tokens",
    "arrival", "decision_time", "start", "first_token", "finish",
    "observed_ttft", "observed_e2e", "service_mode",
    "hot_model_before", "hot_model_after",
    "switch_cost", "buffer_wait", "resource_wait", "cold_lane", "note",
]


async def _serve(session: aiohttp.ClientSession, server: VLLMServer,
                 sem: asyncio.Semaphore, cfg: dict, req: dict,
                 *, t0: float, arrival_rel: float, decision_rel: float,
                 service_mode: str, switch_cost: float,
                 hot_before: str | None, sink: list[dict],
                 policy_label: str, trace_name: str) -> None:
    async with sem:
        out = await submit(
            session, server.url, model=req["model"], prompt=req["_prompt"],
            max_tokens=int(req["output_tokens"]),
            temperature=float(cfg["decoding"]["temperature"]),
            top_p=float(cfg["decoding"]["top_p"]),
            seed=int(cfg["decoding"]["seed"]) + int(req["req_id"]),
        )
        start_rel = round(out["dispatch_ts"] - t0, 4)
        first_rel = (round(out["first_token_ts"] - t0, 4)
                     if out["first_token_ts"] else float("nan"))
        finish_rel = round(out["finish_ts"] - t0, 4)
        sink.append({
            "policy": policy_label,
            "trace_name": trace_name,
            "req_id": int(req["req_id"]),
            "model": req["model"],
            "input_tokens": int(req["input_tokens"]),
            "output_tokens": int(req["output_tokens"]),
            "arrival": arrival_rel,
            "decision_time": decision_rel,
            "start": start_rel,
            "first_token": first_rel,
            "finish": finish_rel,
            "observed_ttft": (round(out["ttft"], 4)
                              if out["ttft"] is not None else float("nan")),
            "observed_e2e": round(out["e2e"], 4),
            "service_mode": service_mode,
            "hot_model_before": hot_before or "",
            "hot_model_after": req["model"],
            "switch_cost": round(switch_cost, 4),
            "buffer_wait": round(decision_rel - arrival_rel, 4),
            "resource_wait": round(start_rel - decision_rel, 4),
            "cold_lane": "",
            "note": ("" if not out["error"] else f"err={out['error']}"),
        })


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--trace-name", default=None)
    ap.add_argument("--pool-dir", type=Path,
                    default=HERE.parent / "datasets" / "cache")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--mode", choices=["warm", "cold"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.log_dir.mkdir(parents=True, exist_ok=True)
    trace_name = args.trace_name or args.trace.stem
    policy_label = f"{args.mode}_c{args.concurrency}"

    with args.trace.open() as f:
        rows = list(csv.DictReader(f))
    pool_cache: dict[str, list[dict]] = {}
    for r in rows:
        r["ts"] = float(r["ts"])
        r["input_tokens"] = int(r["input_tokens"])
        r["output_tokens"] = int(r["output_tokens"])
        r["req_id"] = int(r["req_id"])
        pn = r["prompt_pool"]
        if pn not in pool_cache:
            with (args.pool_dir / pn).open() as pf:
                pool_cache[pn] = [json.loads(line) for line in pf]
        r["_prompt"] = pool_cache[pn][int(r["prompt_idx"])]["prompt"]

    spec = _spec(cfg)
    log_main = args.log_dir / f"vllm_{spec.name}.log"
    server = VLLMServer(spec, str(log_main))

    if args.mode == "warm":
        server.start(); server.wait_ready()

    sink: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    t0 = time.monotonic()
    last_wave_ts = -1.0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        inflight: list[asyncio.Task] = []
        for r in rows:
            wait_for = r["ts"] - (time.monotonic() - t0)
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            arrival_rel = round(time.monotonic() - t0, 4)

            cold_cost = 0.0
            service_mode = "hot_warm" if args.mode == "warm" else "cold"
            hot_before = spec.name if args.mode == "warm" else None

            # Cold mode: each new wave (new trace ts) restarts the server
            # so the wave's first request pays one cold load.
            if args.mode == "cold" and r["ts"] != last_wave_ts:
                if server.proc is not None:
                    server.stop()
                t_load = time.monotonic()
                server.start(); ready_s = server.wait_ready()
                cold_cost = time.monotonic() - t_load
                last_wave_ts = r["ts"]
                hot_before = None

            decision_rel = round(time.monotonic() - t0, 4)

            inflight.append(asyncio.create_task(_serve(
                session, server, sem, cfg, r,
                t0=t0, arrival_rel=arrival_rel, decision_rel=decision_rel,
                service_mode=service_mode, switch_cost=cold_cost,
                hot_before=hot_before, sink=sink,
                policy_label=policy_label, trace_name=trace_name)))
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
    server.stop()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sink.sort(key=lambda d: (d["arrival"], d["req_id"]))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for row in sink:
            w.writerow({k: row.get(k, "") for k in DETAIL_COLS})
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
