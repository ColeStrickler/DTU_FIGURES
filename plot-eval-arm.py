#!/usr/bin/env python3
"""
Plot slowdowns for ARM platform under Single-Bank Attack.
Reads CSVs similar to the AMD script but produces a single-panel figure.
"""
import argparse
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch rendering
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
import numpy as np

# Matplotlib < 3.9 expects the deprecated np.Inf alias; restore it when missing.
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


def build_color_map(spec_workloads: list[str], matmul_workloads: list[str], feedsim_workloads: list[str], mediawiki_workloads: list[str]) -> dict[str, tuple[str, str, str, str]]:
    """Build color map with 4 color shades (1, 2, 4, 8 attackers) for each workload."""
    # Colors for 1, 2, 4, 8 attackers (lightest to darkest)
    spec_colors = ("#C5DDEC", "#8AB8D6", "#4F8BBF", "#1F4E79")
    matmul_colors = ("#F2C4CD", "#E59AAA", "#C96481", "#7A2E3E")
    feedsim_colors = ("#C9E5DC", "#9DCFBA", "#5FA88A", "#266E5E")
    mediawiki_colors = ("#FFD6A8", "#FFB366", "#E68A33", "#CC6600")

    color_map: dict[str, tuple[str, str, str, str]] = {}

    for w in spec_workloads:
        color_map[w] = spec_colors
    for w in matmul_workloads:
        color_map[w] = matmul_colors
    for w in feedsim_workloads:
        color_map[w] = feedsim_colors
    for w in mediawiki_workloads:
        color_map[w] = mediawiki_colors

    return color_map


def get_workload_label(workload: str) -> str:
    """Convert workload name to display label."""
    if workload.startswith("feedsim-"):
        qps = workload.replace("feedsim-", "").replace("qps", "").upper()
        return f"Feedsim@{qps}QPS"
    # Normalize MediaWiki label with capital M
    if workload.lower().startswith("mediawiki-"):
        return "Mediawiki"
    return workload


def plot_panel(ax, title: str, workloads: list[str], data: dict[int, dict[str, float]], color_map: dict[str, tuple[str, str, str, str]]):
    """Plot grouped bars for baseline (0), 1, 2, 4, and 8 attackers."""
    edge = "#2a2a2a"
    bar_width = 0.15
    # Determine attacker counts dynamically from available data
    attacker_counts = sorted(data.keys())
    xs = np.arange(len(workloads))
    
    for i, num_attackers in enumerate(attacker_counts):
        # Offset: -0.30, -0.15, 0, 0.15, 0.30 for 5 bars
        offset = (i - 2) * bar_width
        
        for j, workload in enumerate(workloads):
            h = data.get(num_attackers, {}).get(workload, 1.0)
            x_pos = xs[j] + offset
            
            # Use gray for baseline (0 attackers), color shades for others
            if num_attackers == 0:
                color = "#808080"  # Gray for baseline
            else:
                # Map attacker counts to color indices: 1->0, 2->1, 4->2, 8->3; default to darkest for others
                color_idx = {1: 0, 2: 1, 4: 2, 8: 3}.get(num_attackers, 3)
                color = color_map.get(workload, ("#cccccc", "#aaaaaa", "#888888", "#666666"))[color_idx]
            
            ax.bar(x_pos, h, width=bar_width, color=color, edgecolor=edge, linewidth=0.6, zorder=3)

    matmul_start = None
    feedsim_start = None
    mediawiki_start = None

    for j, workload in enumerate(workloads):
        if matmul_start is None and workload.startswith(("matmul-", "MatMult")):
            matmul_start = j
        if feedsim_start is None and workload.startswith("feedsim-"):
            feedsim_start = j
        if mediawiki_start is None and workload.startswith("mediawiki-"):
            mediawiki_start = j

    line_color = "#c0c0c0"
    line_width = 1.2
    if matmul_start is not None:
        ax.axvline(matmul_start - 0.5, color=line_color, linewidth=line_width, alpha=0.8, zorder=2, linestyle="-")
    if feedsim_start is not None:
        ax.axvline(feedsim_start - 0.5, color=line_color, linewidth=line_width, alpha=0.8, zorder=2, linestyle="-")
    if mediawiki_start is not None:
        ax.axvline(mediawiki_start - 0.5, color=line_color, linewidth=line_width, alpha=0.8, zorder=2, linestyle="-")

    ax.axhline(1.0, color="#FF0000", linewidth=0.8, alpha=0.8, zorder=9, linestyle="--")

    ax.set_xticks(xs)
    workload_labels = [get_workload_label(w) for w in workloads]
    ax.set_xticklabels(workload_labels, rotation=35, ha="right", fontsize=8)

    ax.set_ylabel("Slowdown", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1500)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)


def load_arm_benchmark_data(data_dir: str = ".") -> tuple[list[str], dict[int, dict[str, float]]]:
    """Load ARM benchmark data from available CSV files (spec, matmul, feedsim, mediawiki).

    Mirrors the AMD loader structure. ARM supports attacker counts: 0, 1, 2, 4, 8.
    """
    arm_ddr = {}
    all_benchmarks: list[str] = []

    data_path = Path(data_dir)

    # Map of CSV file to workload column name for direct Slowdown reads
    csv_configs = {
        "arm_spec.csv": "spec_bench",
    }

    for csv_file, workload_col in csv_configs.items():
        csv_path = data_path / csv_file
        if not csv_path.exists():
            continue

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get(workload_col) or not row.get("Slowdown"):
                    continue

                try:
                    num_attackers = int(float(row.get("num_attackers", "0")))
                    bench = row[workload_col].strip()
                    slowdown = float(row.get("Slowdown", ""))
                except (ValueError, KeyError):
                    continue

                if bench not in all_benchmarks:
                    all_benchmarks.append(bench)

                arm_ddr.setdefault(num_attackers, {})[bench] = slowdown

    # Matmul: use Slowdown directly from arm_matmul.csv with Matrix column
    matmul_path = data_path / "arm_matmul.csv"
    if matmul_path.exists():
        with open(matmul_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                matrix = (row.get("Matrix") or "").strip()
                slowdown_str = (row.get("Slowdown") or "").strip()
                
                if not matrix or not slowdown_str:
                    continue
                
                try:
                    num_attackers = int(float(row.get("num_attackers", "0")))
                    slowdown = float(slowdown_str)
                except (ValueError, KeyError):
                    continue
                
                # Use Matrix column directly as workload name
                workload = matrix
                
                arm_ddr.setdefault(num_attackers, {})[workload] = slowdown
                if workload not in all_benchmarks:
                    all_benchmarks.append(workload)

    # Feedsim: compute slowdown from p95_latency_ms grouped by requested_qps
    feedsim_path = data_path / "arm_feedsim.csv"
    if feedsim_path.exists():
        rows: list[dict[str, str]] = []
        with open(feedsim_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        baselines: dict[str, float] = {}
        for r in rows:
            rq = r.get("requested_qps", "").strip()
            try:
                na = int(float(r.get("num_attackers", "0")))
                p95 = float(r.get("p95_latency_ms", ""))
            except ValueError:
                continue
            if not rq:
                continue
            if na == 0:
                baselines[rq] = p95

        for r in rows:
            rq = r.get("requested_qps", "").strip()
            try:
                na = int(float(r.get("num_attackers", "0")))
                p95 = float(r.get("p95_latency_ms", ""))
            except ValueError:
                continue
            if not rq:
                continue
            base = baselines.get(rq)
            if base and base > 0:
                w = f"feedsim-{rq}qps"
                slowdown = p95 / base
                arm_ddr.setdefault(na, {})[w] = slowdown
                if w not in all_benchmarks:
                    all_benchmarks.append(w)

    # MediaWiki: compute slowdown from wrk_rps relative to baseline (num_attackers == 0)
    mediawiki_path = data_path / "arm_mediawiki.csv"
    if mediawiki_path.exists():
        rows: list[dict[str, str]] = []
        with open(mediawiki_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        # Compute baseline as average wrk_rps with 0 attackers
        baseline_vals: list[float] = []
        for r in rows:
            try:
                na = int(float(r.get("num_attackers", "0")))
                rps = float((r.get("wrk_rps") or "").strip())
            except ValueError:
                continue
            if na == 0:
                baseline_vals.append(rps)

        baseline_rps = float(np.mean(baseline_vals)) if baseline_vals else None

        if baseline_rps and baseline_rps > 0:
            for r in rows:
                try:
                    na = int(float(r.get("num_attackers", "0")))
                    rps = float((r.get("wrk_rps") or "").strip())
                except ValueError:
                    continue
                if rps <= 0:
                    continue

                slowdown = baseline_rps / rps
                workload = "mediawiki-RPS"
                arm_ddr.setdefault(na, {})[workload] = slowdown
                if workload not in all_benchmarks:
                    all_benchmarks.append(workload)

    # Sort workloads: spec | matmul | feedsim | Mediawiki
    def sort_key(x: str) -> tuple:
        if x.startswith(("600.", "602.", "605.", "620.", "623.")):
            return (0, x)
        elif x.startswith(("matmul-", "MatMult")):
            return (1, x)
        elif x.startswith("feedsim-"):
            return (2, x)
        elif x.startswith("Mediawiki-"):
            return (3, x)
        else:
            return (4, x)

    all_benchmarks.sort(key=sort_key)

    return all_benchmarks, arm_ddr


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot slowdowns for ARM platform (single figure).")
    parser.add_argument("--out", help="Output image path (PNG/PDF). If omitted, writes caption and exits.")
    parser.add_argument("--dir", default=".", help="Directory containing ARM CSVs (arm_spec.csv, arm_matmul.csv, arm_feedsim.csv, arm_Mediawiki.csv).")
    args = parser.parse_args()

    arm_benchmarks, arm_ddr = load_arm_benchmark_data(args.dir)

    spec_workloads = [w for w in arm_benchmarks if w.startswith(("600.", "602.", "605.", "620.", "623."))]
    matmul_workloads = [w for w in arm_benchmarks if w.startswith(("matmul-", "MatMult"))]
    feedsim_workloads = [w for w in arm_benchmarks if w.startswith("feedsim-")]
    mediawiki_workloads = [w for w in arm_benchmarks if w.startswith("mediawiki-")]
    color_map = build_color_map(spec_workloads, matmul_workloads, feedsim_workloads, mediawiki_workloads)

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "font.family": "serif",
        "hatch.color": "#ffffff",
        "hatch.linewidth": 0.6,
    })

    # Scale width based on number of workloads
    n_workloads = len(arm_benchmarks)
    fig_width = max(8.0, 2.0 + n_workloads * 0.4)
    fig, ax = plt.subplots(1, 1, figsize=(4, 2))

    plot_panel(ax, "ARM (Single Figure)", arm_benchmarks, arm_ddr, color_map)

    fig.tight_layout(pad=0.6, rect=[0.0, 0.0, 1.0, 0.94])

    caption = (
        "Slowdowns under single-bank attacks on ARM platform. "
        "Single figure shows baseline and attackers across SPEC, matmul, feedsim, and mediawiki."
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=450, bbox_inches="tight")
        print(f"Wrote plot to {out_path}")
        print(caption)
    else:
        print(caption)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
