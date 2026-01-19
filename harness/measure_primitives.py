"""Measure the six service primitives that anchor the policy thresholds.

For each model:
  * warm TTFT, warm e2e — by hitting an already-running server N times
    after a small warmup.
  * cold TTFT, cold e2e — defined as the *first visible response delay
    including model load*. Measured as ``ready_s + post_load_ttft`` and
    ``ready_s + post_load_e2e`` (i.e. the user starts waiting at the
    moment the request would be issued against a not-yet-loaded model).

Then measure both directed switch costs (8B->70B, 70B->8B): the time
from "stop current server" to "first token of a request issued against
the newly-loaded server", which is the directed switch component the
hybrid break-even rule consumes.

All measurements use the fixed reference prompt of 512 input tokens and
128 output tokens at concurrency 1, as documented in chapter 8 of the
thesis. Output JSON has the exact schema ``controller.py`` consumes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from client import submit  # noqa: E402
from vllm_proc import ServeSpec, VLLMServer  # noqa: E402

PRIMARY = "Meta-Llama-3-8B-Instruct"
SECONDARY = "Meta-Llama-3-70B-Instruct"


def _spec_for(model_name: str, cfg: dict, side: str) -> ServeSpec:
    m = cfg["models"][side]
    repo = m.get("quant_repo_override") or m["hf_repo"]
    quant = m["quantization"] if m["quantization"] not in ("bf16", None) else None
    return ServeSpec(
        name=m["name"],
        model_path=repo,
        tensor_parallel=int(m["tensor_parallel_size"]),
        quantization=quant,
        gpu_memory_utilization=float(m["gpu_memory_utilization"]),
        max_model_len=int(m["max_model_len"]),
        seed=int(cfg["decoding"]["seed"]),
        extra_args=[],
    )


def _make_reference_prompt(target_input_tokens: int) -> str:
    # ~4 chars per token is a stable approximation for English prose; the
    # client sends max_tokens explicitly, so input length only sets prefill.
    base = "The quick brown fox jumps over the lazy dog. " * 4096
    return base[: target_input_tokens * 4]


async def _one_round(url: str, model: str, prompt: str,
                     max_tokens: int) -> dict:
    async with aiohttp.ClientSession() as s:
        return await submit(s, url, model=model, prompt=prompt,
                            max_tokens=max_tokens)


async def measure_for_model(model_name: str, cfg: dict, side: str,
                            log_dir: Path, n_warmup: int, n_measure: int,
                            in_toks: int, out_toks: int) -> dict:
    spec = _spec_for(model_name, cfg, side)
    log = log_dir / f"primitives_{model_name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cold_ttft_total: list[float] = []
    cold_e2e_total: list[float] = []
    cold_ready_times: list[float] = []
    cold_post_load_ttft: list[float] = []
    cold_post_load_e2e: list[float] = []
    warm_ttfts: list[float] = []
    warm_e2es: list[float] = []
    prompt = _make_reference_prompt(in_toks)

    # ---- COLD path: load + first request, repeated --------------------
    # User-visible cold TTFT/E2E = load_time + post_load_first_request.
    for _ in range(n_measure):
        server = VLLMServer(spec, str(log))
        server.start()
        ready_s = server.wait_ready()
        cold_ready_times.append(ready_s)
        first = await _one_round(server.url, model_name, prompt, out_toks)
        if first["ttft"] is not None:
            cold_post_load_ttft.append(first["ttft"])
            cold_post_load_e2e.append(first["e2e"])
            cold_ttft_total.append(ready_s + first["ttft"])
            cold_e2e_total.append(ready_s + first["e2e"])
        server.stop()
        await asyncio.sleep(2.0)

    # ---- WARM path: leave running, n_warmup discarded, n_measure kept --
    server = VLLMServer(spec, str(log))
    server.start()
    server.wait_ready()
    for _ in range(n_warmup):
        await _one_round(server.url, model_name, prompt, out_toks)
    for _ in range(n_measure):
        r = await _one_round(server.url, model_name, prompt, out_toks)
        if r["ttft"] is not None:
            warm_ttfts.append(r["ttft"])
            warm_e2es.append(r["e2e"])
    server.stop()

    def _med(xs: list[float]) -> float:
        return round(statistics.median(xs), 4) if xs else float("nan")

    def _p95(xs: list[float]) -> float:
        if len(xs) >= 20:
            return round(statistics.quantiles(xs, n=20)[-1], 4)
        return _med(xs)

    return {
        "warm_ttft": _med(warm_ttfts),
        "warm_e2e": _med(warm_e2es),
        "cold_ttft": _med(cold_ttft_total),
        "cold_e2e": _med(cold_e2e_total),
        "cold_ready_s_median": _med(cold_ready_times),
        "cold_post_load_ttft_median": _med(cold_post_load_ttft),
        "cold_post_load_e2e_median": _med(cold_post_load_e2e),
        "warm_ttft_p95": _p95(warm_ttfts),
        "warm_e2e_p95": _p95(warm_e2es),
        "samples": {
            "warm_ttft": warm_ttfts,
            "warm_e2e": warm_e2es,
            "cold_ttft_total": cold_ttft_total,
            "cold_e2e_total": cold_e2e_total,
            "cold_ready_s": cold_ready_times,
        },
    }


async def measure_switch(cfg: dict, from_side: str, to_side: str,
                         from_name: str, to_name: str, log_dir: Path,
                         n_measure: int, in_toks: int, out_toks: int) -> dict:
    """Time stop(from) -> start(to) -> first warm response. Returns median
    plus the raw samples so the statistical-scope paragraph can quote
    variance honestly."""
    samples: list[float] = []
    for _ in range(n_measure):
        spec_a = _spec_for(from_name, cfg, from_side)
        srv_a = VLLMServer(spec_a, str(log_dir / f"switch_{from_name}.log"))
        srv_a.start(); srv_a.wait_ready()
        # warm A so the switch starts from a steady state
        await _one_round(srv_a.url, from_name,
                         _make_reference_prompt(in_toks), out_toks)
        t0 = time.monotonic()
        srv_a.stop()
        spec_b = _spec_for(to_name, cfg, to_side)
        srv_b = VLLMServer(spec_b, str(log_dir / f"switch_{to_name}.log"))
        srv_b.start()
        srv_b.wait_ready()
        first = await _one_round(srv_b.url, to_name,
                                 _make_reference_prompt(in_toks), out_toks)
        srv_b.stop()
        if first["ttft"] is None:
            continue
        elapsed = time.monotonic() - t0
        samples.append(elapsed)
        await asyncio.sleep(2.0)
    return {
        "median": round(statistics.median(samples), 3) if samples else float("nan"),
        "samples": samples,
    }


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    pm = cfg["primitives_measurement"]
    in_toks = int(pm["reference_input_tokens"])
    out_toks = int(pm["reference_output_tokens"])
    n_warm = int(pm["warmup_requests"])
    n_meas = int(pm["measured_requests"])

    per_model: dict = {}
    for side_key in ("primary", "secondary"):
        name = cfg["models"][side_key]["name"]
        per_model[name] = await measure_for_model(
            name, cfg, side_key, args.log_dir, n_warm, n_meas, in_toks, out_toks)
        print(f"[{name}] warm_ttft={per_model[name]['warm_ttft']}s "
              f"cold_ttft={per_model[name]['cold_ttft']}s "
              f"(load_s={per_model[name]['cold_ready_s_median']}s)")

    sw_8_70 = await measure_switch(
        cfg, "primary", "secondary", PRIMARY, SECONDARY,
        args.log_dir, n_meas, in_toks, out_toks)
    sw_70_8 = await measure_switch(
        cfg, "secondary", "primary", SECONDARY, PRIMARY,
        args.log_dir, n_meas, in_toks, out_toks)

    out = {
        "config": cfg,
        "per_model": per_model,
        "switch_cost": {
            f"{PRIMARY}|{SECONDARY}": sw_8_70["median"],
            f"{SECONDARY}|{PRIMARY}": sw_70_8["median"],
            f"{PRIMARY}|{PRIMARY}": 0.0,
            f"{SECONDARY}|{SECONDARY}": 0.0,
        },
        "switch_samples": {
            f"{PRIMARY}|{SECONDARY}": sw_8_70["samples"],
            f"{SECONDARY}|{PRIMARY}": sw_70_8["samples"],
        },
        "reference_prompt": {"input_tokens": in_toks, "output_tokens": out_toks,
                             "concurrency": 1},
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
