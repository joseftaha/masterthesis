#!/usr/bin/env python3
"""Render the experiment figures (overview heatmap, trade-off scatter,
per-trace timelines, TTFT and completion curves) from the per-trace
output directories under ``runs_real/``.

Reads ``summaries.csv``, ``*__details.csv`` and ``*__switches.csv`` from
each ``<runs_root>/<trace>/`` directory and writes PNGs into ``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Theme and palettes
# ---------------------------------------------------------------------------

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.0,
    rc={
        "figure.dpi": 300,
        "savefig.dpi": 900,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

STRATEGY_COLORS: Dict[str, str] = {
    "All-cold": "#8C96A3",
    "Hot: Small": "#2B538B",
    "Hot: Medium": "#2B538B",
    "Hot: Large": "#2B538B",
    "Hot: Primary": "#2B538B",
    "Hot: Secondary": "#2B538B",
    "Switch on miss": "#DF7717",
    "Hybrid adaptive": "#2A9D8F",
}

TRACE_COLORS: Dict[str, str] = {
    "Sparse tail": "#4C78A8",
    "Bursty tail": "#F58518",
    "Balanced mix": "#72B7B2",
    "Popularity shift": "#B279A2",
    "Batch C1": "#A0AEC0",
    "Batch C4": "#718096",
    "Batch C8": "#4A5568",
}

TRACE_MARKERS: Dict[str, str] = {
    "Sparse tail": "o",
    "Bursty tail": "s",
    "Balanced mix": "D",
    "Popularity shift": "^",
    "Batch C1": "o",
    "Batch C4": "s",
    "Batch C8": "D",
}

TRACE_ABBR: Dict[str, str] = {
    "Sparse tail": "ST",
    "Bursty tail": "BT",
    "Balanced mix": "BM",
    "Popularity shift": "PS",
    "Batch C1": "C1",
    "Batch C4": "C4",
    "Batch C8": "C8",
}

MODEL_COLORS: Dict[str, str] = {
    "Small": "#4C78A8",
    "Medium": "#B279A2",
    "Large": "#54A24B",
    "Primary": "#4C78A8",
    "Secondary": "#B279A2",
}

STAGE_COLORS: Dict[str, str] = {
    "Waiting": "#D9D9D9",
    "TTFT": "#F2C14E",
    "Decode": "#5DA5DA",
}

SWITCH_COLOR = "#C44E52"

LOWER_IS_BETTER_CMAP = sns.diverging_palette(145, 15, s=85, l=45, center="light", as_cmap=True)
HIGHER_IS_BETTER_CMAP = sns.diverging_palette(15, 145, s=85, l=45, center="light", as_cmap=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(series: Sequence[float], q: float) -> float:
    arr = np.asarray(list(series), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q * 100.0))


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_trace_dirs(runs_root: Path) -> List[Path]:
    if (runs_root / "summaries.csv").exists():
        return [runs_root]
    trace_dirs = sorted({p.parent for p in runs_root.rglob("summaries.csv")})
    if not trace_dirs:
        raise FileNotFoundError(f"No summaries.csv found under {runs_root}")
    return trace_dirs


def try_load_trace_catalog(runs_root: Path) -> Dict[str, Dict[str, str]]:
    candidates = [
        runs_root / "trace_catalog.json",
        runs_root.parent / "trace_catalog.json",
        runs_root.parent.parent / "trace_catalog.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def is_batch_trace(trace_name: str) -> bool:
    lower = str(trace_name).lower()
    return ("offline" in lower) or ("batch" in lower) or ("wave" in lower)


def human_model_label(name: str) -> str:
    lower = str(name).strip().lower()
    common = {
        "small": "Small",
        "medium": "Medium",
        "large": "Large",
        "primary": "Primary",
        "secondary": "Secondary",
    }
    if lower in common:
        return common[lower]
    label = re.sub(r"[-_]+", " ", str(name)).strip()
    label = re.sub(r"\b(instruct|chat|awq|gptq|int4|int8|fp16|bf16)\b", "", label, flags=re.I)
    label = re.sub(r"\s+", " ", label).strip()
    label = re.sub(r"\b(\d+)b\b", lambda m: f"{m.group(1)}B", label, flags=re.I)
    return label.title() if label else str(name)


def parse_policy(policy: str) -> Tuple[str, Optional[str]]:
    if policy.startswith("static_hot"):
        m = re.search(r"\[(.*?)\]", policy)
        return "static_hot", m.group(1) if m else None
    if policy.startswith("hybrid_break_even"):
        m = re.search(r"\[(.*?)\]", policy)
        return "hybrid_break_even", m.group(1) if m else None
    return policy, None


def policy_label(policy: str, *, collapse_hybrid: bool = False) -> str:
    base, suffix = parse_policy(policy)
    if base == "all_cold":
        return "All-cold"
    if base == "reactive_switch_on_miss":
        return "Switch on miss"
    if base == "static_hot":
        return f"Hot: {human_model_label(suffix or 'Hot')}"
    if base == "hybrid_break_even":
        if collapse_hybrid:
            return "Hybrid adaptive"
        if suffix:
            return f"Hybrid: {human_model_label(suffix)}"
        return "Hybrid adaptive"
    return policy.replace("_", " ").title()


def trace_label(name: str, catalog: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    if catalog and name in catalog and catalog[name].get("display_name"):
        display = str(catalog[name]["display_name"])
        display = display.replace(" (80/20)", "")
        display = display.replace(" (50/50)", "")
        display = display.replace("Offline batch · ", "Batch ")
        return display

    lower = str(name).lower()
    mapping = {
        "interactive_sparse_tail": "Sparse tail",
        "interactive_bursty_tail": "Bursty tail",
        "interactive_balanced_mix": "Balanced mix",
        "interactive_popularity_shift": "Popularity shift",
    }
    if lower in mapping:
        return mapping[lower]
    m = re.match(r"offline_batch_c(\d+)_primary", lower)
    if m:
        return f"Batch C{m.group(1)}"
    return re.sub(r"[_-]+", " ", str(name)).strip().title()


def strategy_sort_key(label: str) -> Tuple[int, str]:
    order = {
        "All-cold": 0,
        "Hot: Small": 1,
        "Hot: Medium": 2,
        "Hot: Large": 3,
        "Hot: Primary": 4,
        "Hot: Secondary": 5,
        "Switch on miss": 6,
        "Hybrid adaptive": 7,
    }
    return (order.get(label, 99), label)


def trace_sort_key(label: str) -> Tuple[int, str]:
    order = {
        "Sparse tail": 0,
        "Bursty tail": 1,
        "Balanced mix": 2,
        "Popularity shift": 3,
        "Batch C1": 10,
        "Batch C4": 11,
        "Batch C8": 12,
    }
    return (order.get(label, 99), label)


def sorted_unique(values: Iterable[str], key_fn) -> List[str]:
    return sorted(set(values), key=key_fn)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def nice_metric_name(metric: str) -> str:
    names = {
        "p95_ttft": "P95 TTFT (s)",
        "avg_e2e": "Avg end-to-end (s)",
        "throughput_rps": "Throughput (req/s)",
        "cold_share": "Cold requests (%)",
    }
    return names.get(metric, metric)


# ---------------------------------------------------------------------------
# Loading and derivation
# ---------------------------------------------------------------------------

def load_outputs(runs_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: List[pd.DataFrame] = []
    detail_frames: List[pd.DataFrame] = []
    switch_frames: List[pd.DataFrame] = []

    for trace_dir in discover_trace_dirs(runs_root):
        summary_path = trace_dir / "summaries.csv"
        if summary_path.exists():
            sdf = pd.read_csv(summary_path)
            sdf["trace_dir"] = str(trace_dir)
            if "trace_name" not in sdf.columns:
                sdf["trace_name"] = trace_dir.name
            summary_frames.append(sdf)
        for path in sorted(trace_dir.glob("*__details.csv")):
            ddf = pd.read_csv(path)
            ddf["trace_dir"] = str(trace_dir)
            if "trace_name" not in ddf.columns:
                ddf["trace_name"] = trace_dir.name
            detail_frames.append(ddf)
        for path in sorted(trace_dir.glob("*__switches.csv")):
            swf = pd.read_csv(path)
            swf["trace_dir"] = str(trace_dir)
            if "trace_name" not in swf.columns:
                swf["trace_name"] = trace_dir.name
            switch_frames.append(swf)

    if not summary_frames:
        raise FileNotFoundError("No result tables found under --runs-root.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    switches_df = pd.concat(switch_frames, ignore_index=True) if switch_frames else pd.DataFrame()

    for df in [summary_df, details_df, switches_df]:
        for col in [
            "arrival", "decision_time", "start", "first_token", "finish",
            "observed_ttft", "observed_e2e", "switch_cost", "buffer_wait", "resource_wait",
            "req_id", "at_time", "cost", "n_requests", "avg_ttft", "p95_ttft", "avg_e2e",
            "p95_e2e", "cold_count", "warm_count", "switch_count", "makespan",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return summary_df, details_df, switches_df


def derive_tables(
    summary_df: pd.DataFrame,
    details_df: pd.DataFrame,
    switches_df: pd.DataFrame,
    *,
    catalog: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = summary_df.copy()
    summary["throughput_rps"] = summary["n_requests"] / summary["makespan"].replace(0, np.nan)
    summary["cold_share"] = 100.0 * summary["cold_count"] / summary["n_requests"].replace(0, np.nan)
    summary["switch_per_100"] = 100.0 * summary["switch_count"] / summary["n_requests"].replace(0, np.nan)
    summary["trace_label"] = summary["trace_name"].map(lambda x: trace_label(x, catalog))
    summary["strategy"] = summary["policy"].map(lambda p: policy_label(p, collapse_hybrid=True))

    if details_df.empty:
        return summary, pd.DataFrame(), pd.DataFrame()

    details = details_df.copy()
    details["decode_time"] = details["observed_e2e"] - details["observed_ttft"]
    details["prefill_time"] = details["first_token"] - details["start"]
    details["queue_total"] = details["start"] - details["arrival"]
    details["trace_label"] = details["trace_name"].map(lambda x: trace_label(x, catalog))
    details["strategy"] = details["policy"].map(lambda p: policy_label(p, collapse_hybrid=True))
    details["model_label"] = details["model"].map(human_model_label)
    details = details.sort_values(["trace_name", "policy", "arrival", "req_id"]).reset_index(drop=True)

    switches = switches_df.copy()
    if not switches.empty:
        switches["trace_label"] = switches["trace_name"].map(lambda x: trace_label(x, catalog))
        switches["strategy"] = switches["policy"].map(lambda p: policy_label(p, collapse_hybrid=True))

    return summary, details, switches


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def best_mask(frame: pd.DataFrame, better: str) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for idx in frame.index:
        row = frame.loc[idx]
        finite = row[np.isfinite(row)]
        if finite.empty:
            continue
        target = finite.max() if better == "higher" else finite.min()
        mask.loc[idx, np.isclose(row, target, equal_nan=False)] = True
    return mask


def plot_overview_heatmap_modern(summary: pd.DataFrame, outdir: Path) -> None:
    sub = summary[~summary["trace_name"].str.startswith("offline")].copy()
    sub["trace_label"] = sub["trace_label"].str.replace(r" Trace", "").str.title()
    strategies = sorted_unique(sub["strategy"], strategy_sort_key)
    traces = sorted_unique(sub["trace_label"], trace_sort_key)
    metrics = [
        ("p95_ttft", "lower", ".2f"),
        ("avg_e2e", "lower", ".2f"),
        ("throughput_rps", "higher", ".2f"),
        ("cold_share", "lower", ".1f"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.4), constrained_layout=True)
    axes = axes.ravel()

    for ax, (metric, better, fmt) in zip(axes, metrics):
        pivot = (
            sub.pivot_table(index="trace_label", columns="strategy", values=metric, aggfunc="first")
            .reindex(index=traces, columns=strategies)
        )
        cmap = LOWER_IS_BETTER_CMAP if better == "lower" else HIGHER_IS_BETTER_CMAP
        annot = pivot.copy()
        for c in annot.columns:
            annot[c] = annot[c].map(lambda x: f"{x:{fmt}}" if pd.notna(x) else "–")

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=annot,
            fmt="",
            linewidths=0.8,
            linecolor="white",
            cbar=False,
            mask=pivot.isna(),
            square=False,
            annot_kws={"fontsize": 8},
        )
        # Outline best value per row.
        # mask = best_mask(pivot, better)
        # for i, row_name in enumerate(pivot.index):
        #     for j, col_name in enumerate(pivot.columns):
        #         if bool(mask.loc[row_name, col_name]):
        #             ax.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="#1F2937", linewidth=1.6))
        ax.set_title(nice_metric_name(metric), pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=28)
        ax.tick_params(axis="y", rotation=0)

    save_figure(fig, outdir, "15_overview_heatmap_modern")


def plot_tradeoff_scatter_modern(summary: pd.DataFrame, outdir: Path) -> None:
    sub = summary[~summary["trace_name"].str.startswith("offline")].copy()
    
    if sub.empty:
        return

    strategy_order = sorted_unique(sub["strategy"], strategy_sort_key)
    trace_order = sorted_unique(sub["trace_label"], trace_sort_key)

    fig, ax = plt.subplots(figsize=(9.5, 6.6), constrained_layout=True)

    sns.scatterplot(
        data=sub,
        x="p95_ttft",
        y="throughput_rps",
        hue="strategy",
        style="trace_label",
        palette={k: STRATEGY_COLORS.get(k, "#4C78A8") for k in strategy_order},
        markers={k: TRACE_MARKERS.get(k, "o") for k in trace_order},
        s=95,
        linewidth=0.8,
        edgecolor="white",
        alpha=0.95,
        ax=ax,
        legend=False,
    )

    # annotate trace abbreviations
    xspan = max(sub["p95_ttft"].max() - sub["p95_ttft"].min(), 1e-9)
    yspan = max(sub["throughput_rps"].max() - sub["throughput_rps"].min(), 1e-9)
    for _, row in sub.iterrows():
        ax.text(
            row["p95_ttft"] + 0.012 * xspan,
            row["throughput_rps"] + 0.010 * yspan,
            TRACE_ABBR.get(row["trace_label"], row["trace_label"][:2].upper()),
            fontsize=8,
            color="#374151",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.72),
        )

    ax.set_xlabel("P95 TTFT (s) ↓")
    ax.set_ylabel("Throughput (req/s) ↑")

    strategy_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=STRATEGY_COLORS.get(s, "#4C78A8"), markeredgecolor="white", label=s)
        for s in strategy_order
    ]
    trace_handles = [
        Line2D([0], [0], marker=TRACE_MARKERS.get(t, "o"), linestyle="", markersize=7,
               markerfacecolor="white", markeredgecolor="#374151", label=f"{TRACE_ABBR.get(t, t[:2].upper())} = {t}")
        for t in trace_order
    ]
    leg1 = ax.legend(handles=strategy_handles, title="Strategy", loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=trace_handles, title="Trace", loc="upper right", bbox_to_anchor=(0.75, 1.0))

    save_figure(fig, outdir, "16_tradeoff_scatter_modern")


def _timeline_strategy_order(details: pd.DataFrame) -> List[str]:
    return sorted_unique(details["strategy"], strategy_sort_key)


def plot_timelines_modern(details: pd.DataFrame, switches: pd.DataFrame, outdir: Path) -> None:
    if details.empty:
        return
    interactive_traces = [t for t in sorted_unique(details["trace_label"], trace_sort_key) if not t.startswith("Batch")]
    for trace_lab in interactive_traces:
        sub = details[details["trace_label"] == trace_lab].copy()
        if sub.empty:
            continue
        strategies = _timeline_strategy_order(sub)
        fig, axes = plt.subplots(
            len(strategies), 1,
            figsize=(15.4, max(7.2, 2.2 * len(strategies))),
            sharex=True,
            constrained_layout=True,
        )
        if len(strategies) == 1:
            axes = [axes]

        for ax, strategy in zip(axes, strategies):
            pol = sub[sub["strategy"] == strategy].sort_values(["arrival", "req_id"]).reset_index(drop=True)
            max_req = len(pol)
            for ridx, row in pol.iterrows():
                y = ridx + 0.15
                height = 0.68
                edge = MODEL_COLORS.get(row["model_label"], "#444444")
                hatch = "///" if str(row.get("service_mode", "")).lower() == "cold" else None
                segments = [
                    (float(row["arrival"]), max(0.0, float(row["start"] - row["arrival"])), STAGE_COLORS["Waiting"]),
                    (float(row["start"]), max(0.0, float(row["first_token"] - row["start"])), STAGE_COLORS["TTFT"]),
                    (float(row["first_token"]), max(0.0, float(row["finish"] - row["first_token"])), STAGE_COLORS["Decode"]),
                ]
                for x0, width, color in segments:
                    if width <= 0:
                        continue
                    ax.broken_barh(
                        [(x0, width)],
                        (y, height),
                        facecolors=color,
                        edgecolors=edge,
                        linewidth=0.75,
                        hatch=hatch,
                    )
            if not switches.empty:
                sw = switches[(switches["trace_label"] == trace_lab) & (switches["strategy"] == strategy)]
                for _, row in sw.iterrows():
                    ax.axvline(float(row["at_time"]), color=SWITCH_COLOR, linestyle=(0, (3, 3)), linewidth=1.2, alpha=0.9)

            ax.set_title(strategy, color=STRATEGY_COLORS.get(strategy, "#374151"), loc="left", pad=6)
            ax.set_ylabel("Request")
            ax.set_ylim(0, max_req + 0.4)
            ax.grid(axis="x", alpha=0.18)
            ax.grid(axis="y", visible=False)
            # show only sparse y ticks for readability
            if max_req <= 12:
                ticks = np.arange(1, max_req + 1)
            else:
                ticks = np.arange(5, max_req + 1, 5)
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(int(t)) for t in ticks])

        axes[-1].set_xlabel("Time (s)")

        stage_handles = [
            Patch(facecolor=STAGE_COLORS["Waiting"], edgecolor="none", label="Waiting"),
            Patch(facecolor=STAGE_COLORS["TTFT"], edgecolor="none", label="TTFT stage"),
            Patch(facecolor=STAGE_COLORS["Decode"], edgecolor="none", label="Decode stage"),
        ]
        model_handles = [
            Line2D([0], [0], color=MODEL_COLORS[m], linewidth=2.5, label=f"{m} request")
            for m in [m for m in ["Small", "Medium", "Large", "Primary", "Secondary"] if m in set(sub["model_label"])]
        ]
        cold_handle = Patch(facecolor="white", edgecolor="#666666", hatch="///", label="Cold path")
        switch_handle = Line2D([0], [0], color=SWITCH_COLOR, linestyle=(0, (3, 3)), linewidth=1.2, label="Switch")
        fig.legend(
            handles=stage_handles + model_handles + [cold_handle, switch_handle],
            loc="lower center",
            ncol=min(7, len(stage_handles) + len(model_handles) + 2),
            bbox_to_anchor=(0.5, -0.005),
        )

        stem = f"17_timeline_modern__{trace_lab.lower().replace(' ', '_')}"
        save_figure(fig, outdir, stem)


def plot_completion_curves_by_trace(details: pd.DataFrame, outdir: Path) -> None:
    sub = details[~details["trace_label"].str.startswith("Batch")].copy()
    if sub.empty:
        return
    traces = sorted_unique(sub["trace_label"], trace_sort_key)
    strategies = sorted_unique(sub["strategy"], strategy_sort_key)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), sharey=False, constrained_layout=True)
    axes = axes.ravel()
    for ax, trace_lab in zip(axes, traces):
        tdf = sub[sub["trace_label"] == trace_lab].copy()
        for strategy in strategies:
            pol = tdf[tdf["strategy"] == strategy].sort_values(["finish", "req_id"])
            if pol.empty:
                continue
            x = pol["finish"].to_numpy(dtype=float)
            y = np.arange(1, len(pol) + 1)
            ax.step(x, y, where="post", linewidth=2.0, color=STRATEGY_COLORS.get(strategy, "#4C78A8"), label=strategy)
        ax.set_title(trace_lab, color=TRACE_COLORS.get(trace_lab, "#374151"), pad=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Completed requests")
    handles = [Line2D([0], [0], color=STRATEGY_COLORS.get(s, "#4C78A8"), lw=2.2, label=s) for s in strategies]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), bbox_to_anchor=(0.5, -0.005))
    save_figure(fig, outdir, "21_completion_curves_by_trace")


def plot_ttft_curves_by_trace(details: pd.DataFrame, outdir: Path) -> None:
    sub = details[~details["trace_label"].str.startswith("Batch")].copy()
    if sub.empty:
        return
    traces = sorted_unique(sub["trace_label"], trace_sort_key)
    strategies = sorted_unique(sub["strategy"], strategy_sort_key)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), sharey=False, constrained_layout=True)
    axes = axes.ravel()
    for ax, trace_lab in zip(axes, traces):
        tdf = sub[sub["trace_label"] == trace_lab].copy()
        for strategy in strategies:
            pol = tdf[tdf["strategy"] == strategy].sort_values(["arrival", "req_id"]).reset_index(drop=True)
            if pol.empty:
                continue
            x = np.arange(1, len(pol) + 1)
            raw = pol["observed_ttft"].astype(float)
            smooth = raw.rolling(window=5, min_periods=1).median()
            ax.scatter(x, raw, s=12, color=STRATEGY_COLORS.get(strategy, "#4C78A8"), alpha=0.15)
            ax.plot(x, smooth, linewidth=2.0, color=STRATEGY_COLORS.get(strategy, "#4C78A8"), label=strategy)
        ax.set_title(trace_lab, color=TRACE_COLORS.get(trace_lab, "#374151"), pad=8)
        ax.set_xlabel("Request order")
        ax.set_ylabel("Rolling median TTFT (s)")
    handles = [Line2D([0], [0], color=STRATEGY_COLORS.get(s, "#4C78A8"), lw=2.2, label=s) for s in strategies]
    raw_handle = Line2D([0], [0], marker="o", linestyle="", color="#9AA0A6", alpha=0.45, markersize=4, label="Raw request TTFT")
    fig.legend(handles=handles + [raw_handle], loc="lower center", ncol=min(6, len(handles) + 1), bbox_to_anchor=(0.5, -0.005))
    save_figure(fig, outdir, "22_ttft_curves_by_trace")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Render experiment figures from per-trace result directories")
    parser.add_argument("--runs-root", required=True, help="Directory containing one subdirectory per trace")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.out)
    ensure_outdir(outdir)

    catalog = try_load_trace_catalog(runs_root)
    summary_df, details_df, switches_df = load_outputs(runs_root)
    summary, details, switches = derive_tables(summary_df, details_df, switches_df, catalog=catalog)

    plot_overview_heatmap_modern(summary, outdir)
    plot_tradeoff_scatter_modern(summary, outdir)


if __name__ == "__main__":
    main()
