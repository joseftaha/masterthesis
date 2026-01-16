"""Controller for Experiment A: drives the policy state machine,
manages the single vLLM serving process on the allocated node, replays
a trace at exact inter-arrival timing, and records per-request metrics.

Outputs:

  runs/<trace>/<policy>__details.csv
    columns: policy, trace_name, req_id, model, input_tokens, output_tokens,
             arrival, decision_time, start, first_token, finish,
             observed_ttft, observed_e2e, service_mode,
             hot_model_before, hot_model_after,
             switch_cost, buffer_wait, resource_wait, cold_lane, note

  runs/<trace>/<policy>__switches.csv
    columns: policy, trace_name, at_time, from_model, to_model, cost, reason

All absolute times are seconds since the start of the dispatch loop
(``t0_mono``), i.e. trace-relative.

WAIT handling (hybrid only): a non-blocking pending queue is kept per
non-resident model. New arrivals do NOT stall the loop — they are
enqueued and the policy is re-evaluated. A background watchdog enforces
the bounded-wait (tau_max) safeguard when no fresh arrivals are coming.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from client import submit  # noqa: E402
from policies import (POLICIES, Decision, Primitives, PolicyState,  # noqa: E402
                      WaitItem)
from vllm_proc import ServeSpec, VLLMServer  # noqa: E402

PRIMARY = "Meta-Llama-3-8B-Instruct"
SECONDARY = "Meta-Llama-3-70B-Instruct"


def _load_prompt_pool(pool_dir: Path, name: str) -> list[dict]:
    with (pool_dir / name).open() as f:
        return [json.loads(line) for line in f]


def _spec_for(model_name: str, cfg: dict) -> ServeSpec:
    side = "primary" if model_name == cfg["models"]["primary"]["name"] else "secondary"
    m = cfg["models"][side]
    repo = m.get("quant_repo_override") or m["hf_repo"]
    quant = m["quantization"] if m["quantization"] not in ("bf16", None) else None
    return ServeSpec(
        name=m["name"],
        model_path=os.environ.get(f"MODEL_PATH_{side.upper()}", repo),
        tensor_parallel=int(m["tensor_parallel_size"]),
        quantization=quant,
        gpu_memory_utilization=float(m["gpu_memory_utilization"]),
        max_model_len=int(m["max_model_len"]),
        seed=int(cfg["decoding"]["seed"]),
        extra_args=[],
    )


def _load_primitives(path: Path) -> Primitives:
    with path.open() as f:
        d = json.load(f)
    return Primitives(
        warm_ttft={k: v["warm_ttft"] for k, v in d["per_model"].items()},
        cold_ttft={k: v["cold_ttft"] for k, v in d["per_model"].items()},
        warm_e2e={k: v["warm_e2e"] for k, v in d["per_model"].items()},
        cold_e2e={k: v["cold_e2e"] for k, v in d["per_model"].items()},
        switch_cost={tuple(k.split("|")): float(v)
                     for k, v in d["switch_cost"].items()},
    )


@dataclass
class Pending:
    """An arrival that hit a WAIT decision and is queued for re-evaluation."""
    req: dict
    arrived_at_mono: float
    arrival_rel: float       # seconds since t0
    decision_time_rel: float


@dataclass
class DetailRow:
    policy: str
    trace_name: str
    req_id: int
    model: str
    input_tokens: int
    output_tokens: int
    arrival: float
    decision_time: float
    start: float
    first_token: float
    finish: float
    observed_ttft: float
    observed_e2e: float
    service_mode: str        # hot_warm | switched_warm | cold
    hot_model_before: str | None = None
    hot_model_after: str | None = None
    switch_cost: float = 0.0
    buffer_wait: float = 0.0
    resource_wait: float = 0.0
    cold_lane: int | None = None
    note: str = ""


@dataclass
class SwitchRow:
    policy: str
    trace_name: str
    at_time: float
    from_model: str
    to_model: str
    cost: float
    reason: str


class Controller:
    def __init__(self, *, cfg: dict, primitives: Primitives, policy_name: str,
                 trace_csv: Path, trace_name: str, pool_dir: Path,
                 out_dir: Path, log_dir: Path, hot_init: str | None = None):
        self.cfg = cfg
        self.prim = primitives
        self.policy_name = policy_name
        self.policy = POLICIES[policy_name]
        self.trace_csv = trace_csv
        self.trace_name = trace_name
        self.pool_dir = pool_dir
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.state = PolicyState(resident=None)
        self.server: VLLMServer | None = None
        self._switches: list[SwitchRow] = []
        self._details: list[DetailRow] = []
        self._hot_init = hot_init
        self._t0_mono: float = 0.0
        self._stop_event: asyncio.Event | None = None
        self._tau_max = primitives.cold_e2e
        self._pending_records: dict[str, list[Pending]] = {}

    # --- helpers ---------------------------------------------------------
    def _rel(self, mono: float | None) -> float:
        if mono is None:
            return float("nan")
        return round(mono - self._t0_mono, 4)

    # --- server lifecycle ------------------------------------------------
    def _start_server(self, model_name: str, *, reason: str) -> float:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f"vllm_{model_name}_{int(time.time())}.log"
        spec = _spec_for(model_name, self.cfg)
        self.server = VLLMServer(spec, str(log))
        self.server.start()
        ready_s = self.server.wait_ready()
        prev = self.state.resident
        self.state.resident = model_name
        self._switches.append(SwitchRow(
            policy=self.policy_name, trace_name=self.trace_name,
            at_time=self._rel(time.monotonic()),
            from_model=prev or "",
            to_model=model_name,
            cost=round(ready_s, 4),
            reason=reason,
        ))
        return ready_s

    def _stop_server(self) -> float:
        if self.server is None:
            return 0.0
        stop_s = self.server.stop()
        self.server = None
        self.state.resident = None
        return stop_s

    def _switch_to(self, target: str, *, reason: str) -> float:
        prev = self.state.resident
        stop_s = self._stop_server() if prev is not None else 0.0
        ready_s = self._start_server(target, reason=reason)
        # annotate the just-appended switch row with the full directed cost
        if self._switches:
            self._switches[-1].cost = round(stop_s + ready_s, 4)
            self._switches[-1].from_model = prev or ""
        return stop_s + ready_s

    def _cold_load(self, target: str, *, reason: str,
                   force_reload: bool) -> float:
        prev_resident = self.state.resident
        if force_reload and self.server is not None:
            self._stop_server()
        elif prev_resident is not None and prev_resident != target:
            self._stop_server()
        if self.state.resident == target:
            return 0.0
        return self._start_server(target, reason=reason)

    # --- request dispatch ------------------------------------------------
    async def _serve(self, session: aiohttp.ClientSession, req: dict,
                     arrival_rel: float, decision_rel: float,
                     service_mode: str, *, switch_cost: float,
                     hot_before: str | None, note: str) -> None:
        assert self.server is not None
        prompt = req["_prompt"]
        out = await submit(
            session, self.server.url,
            model=req["model"],
            prompt=prompt,
            max_tokens=int(req["output_tokens"]),
            temperature=float(self.cfg["decoding"]["temperature"]),
            top_p=float(self.cfg["decoding"]["top_p"]),
            seed=int(self.cfg["decoding"]["seed"]) + int(req["req_id"]),
            request_id=f"{self.policy_name}-{req['req_id']}",
        )
        start_rel = self._rel(out["dispatch_ts"])
        first_rel = self._rel(out["first_token_ts"]) if out["first_token_ts"] else float("nan")
        finish_rel = self._rel(out["finish_ts"])
        observed_ttft = (out["ttft"] if out["ttft"] is not None else float("nan"))
        observed_e2e = out["e2e"]
        self._details.append(DetailRow(
            policy=self.policy_name,
            trace_name=self.trace_name,
            req_id=int(req["req_id"]),
            model=req["model"],
            input_tokens=int(req["input_tokens"]),
            output_tokens=int(req["output_tokens"]),
            arrival=arrival_rel,
            decision_time=decision_rel,
            start=start_rel,
            first_token=first_rel,
            finish=finish_rel,
            observed_ttft=round(observed_ttft, 4) if observed_ttft == observed_ttft else float("nan"),
            observed_e2e=round(observed_e2e, 4),
            service_mode=service_mode,
            hot_model_before=hot_before,
            hot_model_after=self.state.resident,
            switch_cost=round(switch_cost, 4),
            buffer_wait=round(decision_rel - arrival_rel, 4),
            resource_wait=round(start_rel - decision_rel, 4),
            cold_lane=None,
            note=note + (f" err={out['error']}" if out["error"] else ""),
        ))

    # --- main loop -------------------------------------------------------
    async def run(self) -> None:
        # Load and hydrate trace.
        with self.trace_csv.open() as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["ts"] = float(r["ts"])
            r["input_tokens"] = int(r["input_tokens"])
            r["output_tokens"] = int(r["output_tokens"])
            r["req_id"] = int(r["req_id"])
        pools: dict[str, list[dict]] = {}
        for r in rows:
            pn = r["prompt_pool"]
            pools.setdefault(pn, _load_prompt_pool(self.pool_dir, pn))
            r["_prompt"] = pools[pn][int(r["prompt_idx"])]["prompt"]

        if self._hot_init:
            self._start_server(self._hot_init, reason="prewarm")

        timeout = aiohttp.ClientTimeout(total=600)
        connector = aiohttp.TCPConnector(limit=64)
        self._t0_mono = time.monotonic()
        self._stop_event = asyncio.Event()

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            inflight: list[asyncio.Task] = []
            watchdog = asyncio.create_task(self._watchdog(session, inflight))

            try:
                for req in rows:
                    # Sleep until the trace timestamp (wall clock, relative to t0).
                    wait_for = req["ts"] - (time.monotonic() - self._t0_mono)
                    if wait_for > 0:
                        await asyncio.sleep(wait_for)
                    arrived_at_mono = time.monotonic()
                    arrival_rel = self._rel(arrived_at_mono)

                    await self._handle_arrival(session, req, arrived_at_mono,
                                               arrival_rel, inflight)
            finally:
                self._stop_event.set()
                await watchdog

            # Drain anything still pending at end of trace.
            await self._drain_remaining(session, inflight)

            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)

        self._stop_server()
        self._flush()

    async def _handle_arrival(self, session: aiohttp.ClientSession, req: dict,
                              arrived_at_mono: float, arrival_rel: float,
                              inflight: list[asyncio.Task]) -> None:
        decision: Decision = self.policy(self.state, req["model"], self.prim,
                                         now=arrived_at_mono)
        decision_rel = self._rel(time.monotonic())
        hot_before = self.state.resident

        if decision.kind == "SERVE_WARM":
            inflight.append(asyncio.create_task(self._serve(
                session, req, arrival_rel, decision_rel,
                service_mode="hot_warm",
                switch_cost=0.0, hot_before=hot_before,
                note=decision.note)))
            return

        if decision.kind == "SWITCH":
            cost = self._switch_to(decision.target, reason=decision.note)
            inflight.append(asyncio.create_task(self._serve(
                session, req, arrival_rel, decision_rel,
                service_mode="switched_warm",
                switch_cost=cost, hot_before=hot_before,
                note=decision.note)))
            # Drain any pending requests now that their model is resident.
            await self._drain_pending_for(session, decision.target,
                                          inflight, mode="switched_warm")
            return

        if decision.kind == "SERVE_COLD":
            if decision.serve_oldest_pending:
                # Hybrid bounded-wait fallback: serve the OLDEST pending,
                # the new incoming request is then re-evaluated.
                pending = self.state.waiting.get(decision.target, [])
                if pending:
                    oldest = pending.pop(0)
                    cost = self._cold_load(decision.target,
                                           reason="bounded-wait fallback",
                                           force_reload=False)
                    inflight.append(asyncio.create_task(self._serve(
                        session, oldest.req,  # type: ignore[attr-defined]
                        oldest.arrival_rel, oldest.decision_time_rel,
                        service_mode="cold", switch_cost=cost,
                        hot_before=hot_before,
                        note="bounded-wait fallback")))
                # Re-evaluate the new arrival.
                await self._handle_arrival(session, req, arrived_at_mono,
                                           arrival_rel, inflight)
                return
            cost = self._cold_load(decision.target,
                                   reason=decision.note,
                                   force_reload=decision.force_reload)
            inflight.append(asyncio.create_task(self._serve(
                session, req, arrival_rel, decision_rel,
                service_mode="cold",
                switch_cost=cost, hot_before=hot_before,
                note=decision.note)))
            return

        if decision.kind == "WAIT":
            self.state.waiting.setdefault(req["model"], []).append(WaitItem(
                req_id=int(req["req_id"]), arrived_at=arrived_at_mono))
            # Stash the full pending record for later dispatch.
            self._pending_records.setdefault(req["model"], []).append(Pending(
                req=req, arrived_at_mono=arrived_at_mono,
                arrival_rel=arrival_rel,
                decision_time_rel=decision_rel))
            return

    async def _drain_pending_for(self, session: aiohttp.ClientSession,
                                 model: str, inflight: list[asyncio.Task],
                                 *, mode: str) -> None:
        recs = self._pending_records.pop(model, [])
        # Keep the policy's own waiting queue consistent.
        self.state.waiting[model] = []
        for pr in recs:
            inflight.append(asyncio.create_task(self._serve(
                session, pr.req, pr.arrival_rel, pr.decision_time_rel,
                service_mode=mode, switch_cost=0.0,
                hot_before=self.state.resident,
                note="drained from pending")))

    async def _drain_remaining(self, session: aiohttp.ClientSession,
                               inflight: list[asyncio.Task]) -> None:
        # End of trace: any remaining pending must be served via cold load.
        for model, recs in list(self._pending_records.items()):
            if not recs:
                continue
            for pr in recs:
                cost = self._cold_load(model, reason="end-of-trace drain",
                                       force_reload=False)
                inflight.append(asyncio.create_task(self._serve(
                    session, pr.req, pr.arrival_rel, pr.decision_time_rel,
                    service_mode="cold", switch_cost=cost,
                    hot_before=self.state.resident,
                    note="end-of-trace drain")))
            self._pending_records[model] = []
            self.state.waiting[model] = []

    async def _watchdog(self, session: aiohttp.ClientSession,
                        inflight: list[asyncio.Task]) -> None:
        """Enforce bounded-wait safeguard between arrivals: every 0.5s,
        check whether the oldest pending request for any model has aged
        past tau_max. If so, serve it via the cold path."""
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                return
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            for model, recs in list(self._pending_records.items()):
                if not recs:
                    continue
                tau = self._tau_max.get(model, float("inf"))
                while recs and (now - recs[0].arrived_at_mono) >= tau:
                    pr = recs.pop(0)
                    # Mirror the WaitItem queue.
                    waiting = self.state.waiting.get(model, [])
                    if waiting:
                        waiting.pop(0)
                    cost = self._cold_load(model,
                                           reason="watchdog bounded-wait",
                                           force_reload=False)
                    decision_rel = self._rel(time.monotonic())
                    inflight.append(asyncio.create_task(self._serve(
                        session, pr.req, pr.arrival_rel, decision_rel,
                        service_mode="cold", switch_cost=cost,
                        hot_before=self.state.resident,
                        note=f"watchdog bounded-wait age="
                             f"{now - pr.arrived_at_mono:.1f}s")))

    # --- output ----------------------------------------------------------
    def _flush(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        det_path = self.out_dir / f"{self.policy_name}__details.csv"
        sw_path = self.out_dir / f"{self.policy_name}__switches.csv"

        det_cols = [
            "policy", "trace_name", "req_id", "model",
            "input_tokens", "output_tokens",
            "arrival", "decision_time", "start", "first_token", "finish",
            "observed_ttft", "observed_e2e", "service_mode",
            "hot_model_before", "hot_model_after",
            "switch_cost", "buffer_wait", "resource_wait", "cold_lane", "note",
        ]
        with det_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=det_cols)
            w.writeheader()
            for d in sorted(self._details, key=lambda x: (x.arrival, x.req_id)):
                row = {k: getattr(d, k) for k in det_cols}
                row["hot_model_before"] = row["hot_model_before"] or ""
                row["hot_model_after"] = row["hot_model_after"] or ""
                row["cold_lane"] = "" if row["cold_lane"] is None else row["cold_lane"]
                w.writerow(row)

        sw_cols = ["policy", "trace_name", "at_time", "from_model",
                   "to_model", "cost", "reason"]
        with sw_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sw_cols)
            w.writeheader()
            for s in self._switches:
                w.writerow({k: getattr(s, k) for k in sw_cols})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--primitives", type=Path, required=True)
    ap.add_argument("--policy", required=True, choices=list(POLICIES.keys()))
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--trace-name", default=None,
                    help="Override the trace name written into the CSVs "
                         "(defaults to the trace file's stem).")
    ap.add_argument("--pool-dir", type=Path,
                    default=HERE.parent / "datasets" / "cache")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    prim = _load_primitives(args.primitives)

    hot_init: str | None = None
    if args.policy == "static_hot_8B":
        hot_init = PRIMARY
    elif args.policy == "static_hot_70B":
        hot_init = SECONDARY
    elif args.policy == "hybrid_break_even":
        hot_init = PRIMARY
    elif args.policy == "reactive_switch_on_miss":
        hot_init = PRIMARY  # deterministic init; see chapter 8 § Experiment A

    ctl = Controller(
        cfg=cfg, primitives=prim, policy_name=args.policy,
        trace_csv=args.trace,
        trace_name=args.trace_name or args.trace.stem,
        pool_dir=args.pool_dir,
        out_dir=args.out_dir, log_dir=args.log_dir,
        hot_init=hot_init,
    )
    asyncio.run(ctl.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
