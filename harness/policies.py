"""The five policies replayed in Experiment A (P0..P4).

Each policy is a small state machine: given the current resident model,
the queue of waiting requests (per non-resident model), and the measured
service primitives, it returns a *Decision* — one of:

  SERVE_WARM (model already resident matches request)
  SERVE_COLD (load this model now; if ``force_reload`` is set the
              controller MUST tear down any current server first, even
              if the target equals the resident — used by P0 to make
              every request pay one cold load)
  SWITCH    (stop current server, start the requested one)
  WAIT      (do nothing yet; ask again at next event)

All thresholds in P4 are derived from the measured primitives, not
hard-coded.

The hybrid policy receives ``state.waiting[req_model]`` as the queue of
non-resident requests **already enqueued** before this call (the
controller pushes the new arrival AFTER the policy returns). The "+1"
in the gain expression therefore accounts for the incoming request.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

DecisionKind = Literal["SERVE_WARM", "SERVE_COLD", "SWITCH", "WAIT"]


@dataclass
class Primitives:
    warm_ttft: dict[str, float]
    cold_ttft: dict[str, float]
    warm_e2e: dict[str, float]
    cold_e2e: dict[str, float]
    switch_cost: dict[tuple[str, str], float]   # (from, to) seconds


@dataclass
class Decision:
    kind: DecisionKind
    target: str | None = None
    note: str = ""
    force_reload: bool = False     # P0: tear down even if target == resident
    serve_oldest_pending: bool = False  # bounded-wait fallback path


@dataclass
class WaitItem:
    req_id: int
    arrived_at: float


@dataclass
class PolicyState:
    resident: str | None
    waiting: dict[str, list[WaitItem]] = field(default_factory=dict)


def policy_all_cold(state: PolicyState, req_model: str, _: Primitives,
                    now: float | None = None) -> Decision:
    return Decision("SERVE_COLD", req_model, "P0 all_cold",
                    force_reload=True)


def _make_static(hot: str):
    def policy(state: PolicyState, req_model: str, _: Primitives,
               now: float | None = None) -> Decision:
        if state.resident is None:
            return Decision("SERVE_COLD", hot, f"static_hot_{hot} init")
        if req_model == state.resident:
            return Decision("SERVE_WARM", req_model)
        return Decision("SERVE_COLD", req_model, "static: serve miss cold")
    return policy


def _make_reactive(default_hot: str):
    """P3 — reactive switch on miss. The default_hot model is loaded at
    init so subsequent decisions are honest 'switch on miss', not
    'whichever model arrived first becomes hot'."""
    def policy(state: PolicyState, req_model: str, _: Primitives,
               now: float | None = None) -> Decision:
        if state.resident is None:
            return Decision("SERVE_COLD", default_hot, "reactive init")
        if req_model == state.resident:
            return Decision("SERVE_WARM", req_model)
        return Decision("SWITCH", req_model, "reactive: switch on miss")
    return policy


def make_hybrid(default_hot: str):
    """P4 — Hybrid Adaptive Break-Even with Bounded Wait.

    Switch when the cumulative avoided cold-cost on the waiting set
    (including the incoming request) matches the directed switch-cycle
    cost; otherwise wait, but never longer than the median cold latency
    of the secondary model (then serve the oldest waiter cold).
    """
    def policy(state: PolicyState, req_model: str, prim: Primitives,
               now: float | None = None) -> Decision:
        if state.resident is None:
            return Decision("SERVE_COLD", default_hot, "hybrid init")
        if req_model == state.resident:
            return Decision("SERVE_WARM", req_model)

        a = state.resident
        b = req_model
        delta_b = max(prim.cold_ttft[b] - prim.warm_ttft[b], 0.0)
        cycle = prim.switch_cost.get((a, b), 0.0) + prim.switch_cost.get((b, a), 0.0)
        threshold = max(1, math.ceil(cycle / max(delta_b, 1e-6)))

        waiting = state.waiting.get(b, [])
        # gain from already-waiting + this incoming request
        gain = (len(waiting) + 1) * delta_b
        if gain >= cycle:
            return Decision("SWITCH", b,
                            f"hybrid: gain {gain:.1f}s >= cycle {cycle:.1f}s "
                            f"(thr={threshold})")

        # bounded-wait safeguard: starve-relieve oldest if too old
        tau_max = prim.cold_e2e[b]
        if waiting and now is not None:
            oldest = waiting[0]
            if (now - oldest.arrived_at) >= tau_max:
                return Decision("SERVE_COLD", b,
                                f"hybrid: bounded-wait fallback "
                                f"(age={now - oldest.arrived_at:.1f}s)",
                                serve_oldest_pending=True)

        return Decision("WAIT", None,
                        f"hybrid: gain {gain:.1f}s < cycle {cycle:.1f}s")
    return policy


POLICIES = {
    "all_cold":               policy_all_cold,
    "static_hot_8B":          _make_static("Meta-Llama-3-8B-Instruct"),
    "static_hot_70B":         _make_static("Meta-Llama-3-70B-Instruct"),
    "reactive_switch_on_miss": _make_reactive("Meta-Llama-3-8B-Instruct"),
    "hybrid_break_even":      make_hybrid("Meta-Llama-3-8B-Instruct"),
}
