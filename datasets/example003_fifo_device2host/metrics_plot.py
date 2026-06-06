#!/usr/bin/env python3
"""
Plotting module for GPU communication benchmark metrics.

Parses METRICS_JSON lines from benchmark output and generates:
  1. Metrics vs num_pushes (one curve per block_size)
  2. Metrics vs block_size  (one curve per num_pushes)
"""

import json
import os
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_all_metrics(text: str) -> List[Dict[str, Any]]:
    """Extract every METRICS_JSON line from benchmark output."""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("METRICS_JSON:"):
            json_str = line[len("METRICS_JSON:"):].strip()
            try:
                results.append(json.loads(json_str))
            except json.JSONDecodeError:
                continue
    return results


_METRIC_LABELS = {
    "items_per_sec": "Throughput (items/s)",
    "throughput_MBps": "Throughput (MB/s)",
    "lat_avg_ns": "Average Latency (ns)",
    "throughput_gbps": "Throughput (GB/s)",
    "latency_ms": "Latency (ms)",
    "data_size_mb": "Block Size (MB)",
}


def plot_metrics(
    metrics: List[Dict[str, Any]],
    output_dir: str = ".",
    metric_keys: Optional[List[str]] = None,
    prefix: str = "bench",
) -> List[str]:
    """Generate benchmark plots and return paths of saved figures.

    Produces two groups of plots:
      - Fixed block_size, sweep num_pushes   (one figure per metric)
      - Fixed num_pushes, sweep block_size   (one figure per metric)

    Parameters
    ----------
    metrics : list of dicts
        Each dict is one METRICS_JSON record.
    output_dir : str
        Directory to save PNG files.
    metric_keys : list of str, optional
        Which metrics to plot. Defaults to items_per_sec, throughput_MBps,
        lat_avg_ns.
    prefix : str
        Filename prefix for saved PNGs.

    Returns
    -------
    list of str
        Paths to saved PNG files.
    """
    if plt is None:
        print("[plotting] matplotlib is not installed – skipping plots.")
        return []

    if not metrics:
        print("[plotting] No metrics to plot.")
        return []

    if metric_keys is None:
        metric_keys = ["items_per_sec", "throughput_MBps", "lat_avg_ns"]

    os.makedirs(output_dir, exist_ok=True)

    # Organise data ---------------------------------------------------------
    block_sizes = sorted({m["block_size"] for m in metrics})
    push_counts = sorted({m["num_pushes"] for m in metrics})

    # Build lookup: (block_size, num_pushes) -> metric dict
    lookup: Dict[tuple, Dict] = {}
    for m in metrics:
        lookup[(m["block_size"], m["num_pushes"])] = m

    saved: List[str] = []

    # --- Group 1: fixed block_size, x-axis = num_pushes ---------------------
    for mk in metric_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        for bs in block_sizes:
            xs, ys = [], []
            for np_ in push_counts:
                rec = lookup.get((bs, np_))
                if rec and mk in rec:
                    xs.append(np_)
                    ys.append(rec[mk])
            if xs:
                ax.plot(xs, ys, marker="o", label=f"block_size={bs}")
        ax.set_xlabel("num_pushes")
        ax.set_ylabel(_METRIC_LABELS.get(mk, mk))
        ax.set_title(f"{_METRIC_LABELS.get(mk, mk)}  vs  num_pushes")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_{mk}_vs_num_pushes.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"[plotting] saved {path}")

    # --- Group 2: fixed num_pushes, x-axis = block_size ---------------------
    for mk in metric_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        for np_ in push_counts:
            xs, ys = [], []
            for bs in block_sizes:
                rec = lookup.get((bs, np_))
                if rec and mk in rec:
                    xs.append(bs)
                    ys.append(rec[mk])
            if xs:
                ax.plot(xs, ys, marker="s", label=f"num_pushes={np_}")
        ax.set_xlabel("block_size")
        ax.set_ylabel(_METRIC_LABELS.get(mk, mk))
        ax.set_title(f"{_METRIC_LABELS.get(mk, mk)}  vs  block_size")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{prefix}_{mk}_vs_block_size.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"[plotting] saved {path}")

    return saved


# ── Colour palette shared across implementations ─────────────────────────
_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_metrics_compare(
    named_metrics: Dict[str, List[Dict[str, Any]]],
    output_dir: str = ".",
    metric_keys: Optional[List[str]] = None,
    prefix: str = "compare",
    fixed_block_size: int = 256,
) -> List[str]:
    """Compare multiple implementations on the same figures.

    Fixes block_size and plots each metric vs num_pushes, with one curve per
    implementation.

    Parameters
    ----------
    named_metrics : dict[str, list[dict]]
        Mapping from implementation label (e.g. "ref", "generated") to its
        list of METRICS_JSON records.
    output_dir : str
        Directory to save PNG files.
    metric_keys : list[str], optional
        Metrics to plot.  Defaults to items_per_sec, throughput_MBps,
        lat_avg_ns.
    prefix : str
        Filename prefix for saved PNGs.
    fixed_block_size : int
        The block_size to fix (default: 256).

    Returns
    -------
    list of str
        Paths to saved PNG files.
    """
    if plt is None:
        print("[plotting] matplotlib is not installed – skipping plots.")
        return []

    if not named_metrics:
        print("[plotting] No metrics to plot.")
        return []

    if metric_keys is None:
        metric_keys = ["items_per_sec", "throughput_MBps", "lat_avg_ns"]

    os.makedirs(output_dir, exist_ok=True)

    impl_names = list(named_metrics.keys())
    line_styles = ["-", "--", "-.", ":"]

    # Build lookup per implementation, filtered to fixed_block_size
    all_push_counts: set = set()
    lookups: Dict[str, Dict[int, Dict]] = {}  # name -> {num_pushes -> record}
    for name, mlist in named_metrics.items():
        lk: Dict[int, Dict] = {}
        for m in mlist:
            if m["block_size"] == fixed_block_size:
                all_push_counts.add(m["num_pushes"])
                lk[m["num_pushes"]] = m
        lookups[name] = lk

    push_counts = sorted(all_push_counts)

    if not push_counts:
        print(f"[plotting] No data for block_size={fixed_block_size} – skipping.")
        return []

    saved: List[str] = []

    for mk in metric_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for ii, name in enumerate(impl_names):
            ls = line_styles[ii % len(line_styles)]
            color = _COLORS[ii % len(_COLORS)]
            lk = lookups[name]
            xs, ys = [], []
            for np_ in push_counts:
                rec = lk.get(np_)
                if rec and mk in rec:
                    xs.append(np_)
                    ys.append(rec[mk])
            if xs:
                ax.plot(xs, ys, marker="o", color=color, linestyle=ls,
                        label=name)
        ax.set_xlabel("num_pushes")
        ax.set_ylabel(_METRIC_LABELS.get(mk, mk))
        ax.set_title(f"{_METRIC_LABELS.get(mk, mk)}  vs  num_pushes  "
                      f"(block_size={fixed_block_size})")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize="small")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        path = os.path.join(output_dir,
                            f"{prefix}_{mk}_vs_num_pushes_bs{fixed_block_size}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"[plotting] saved {path}")

    return saved
