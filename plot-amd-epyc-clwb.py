#!/usr/bin/env python3
"""
Plot slowdown for same-cluster vs diff-cluster, split by instruction.
Data is loaded from a TSV/CSV file with mean/min/max victim bandwidths.
"""

import argparse
import csv
from pathlib import Path
import numpy as np

# --- NumPy 2.x compatibility shim for older Matplotlib ---
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    last_cluster = ""
    last_instr = ""
    with csv_path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            delimiter = dialect.delimiter
            reader = csv.DictReader(f, delimiter=delimiter)
        except Exception:
            reader = csv.DictReader(f)  # fallback to comma

        for raw in reader:
            raw = {
                (k or "").strip(): (v or "").strip() for k, v in raw.items()
            }
            # Carry forward missing cluster/instruction values (robust to sparse tables)
            cluster = raw.get("cluster", "")
            instruction = raw.get("instruction", "")
            if cluster:
                last_cluster = cluster
            if instruction:
                last_instr = instruction
            raw["cluster"] = last_cluster
            raw["instruction"] = last_instr
            rows.append(raw)
    return rows


def _slowdown_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    # Map: cluster -> instruction -> num_attackers -> stats
    data: dict[str, dict[str, dict[int, dict[str, float]]]] = {}

    # Collect baseline (solo) MEDIAN per cluster/instruction for slowdown computation.
    baseline_median: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            cluster = r["cluster"]  # do not lowercase
            instr = r["instruction"]
            num_attackers = int(float(r["num_attackers"]))
        except Exception:
            continue
        if num_attackers == 0:
            try:
                baseline_median[(cluster, instr)] = float(r["victim_bw_median_mb_s"])
            except Exception:
                pass

    for r in rows:
        try:
            cluster = r["cluster"]  # do not lowercase
            instr = r["instruction"]
            num_attackers = int(float(r["num_attackers"]))
            med_bw = float(r["victim_bw_median_mb_s"])
            min_bw = float(r["victim_bw_min_mb_s"])
            max_bw = float(r["victim_bw_max_mb_s"])
        except Exception:
            continue

        solo_median = baseline_median.get((cluster, instr))
        if not solo_median:
            continue

        slow_med = solo_median / med_bw if med_bw else np.nan
        # Error bars derived from min/max of victim throughput
        slow_low = solo_median / max_bw if max_bw else np.nan
        slow_high = solo_median / min_bw if min_bw else np.nan

        data.setdefault(cluster, {}).setdefault(instr, {})[num_attackers] = {
            "median": slow_med,
            "low": slow_low,
            "high": slow_high,
        }
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot slowdown (same vs diff cluster) by instruction.")
    parser.add_argument("--csv", default="amd-epyc.csv", help="Input TSV/CSV (tab-delimited) file.")
    parser.add_argument("--out", help="Output path for plot (PNG/PDF).")
    args = parser.parse_args()

    # -----------------------
    # Data (from file)
    # -----------------------
    rows = _read_rows(Path(args.csv))
    data = _slowdown_stats(rows)

    attack_labels = ["Solo", "1 attk", "2 attk", "3 attk"]

    # -----------------------
    # Styling (match your other script)
    # -----------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 150,
    })

    # -----------------------
    # X layout: 3 groups (Same CCX, Diff CCX, Diff CCD) for each instruction
    # (Same CCX: clwb, nt-store) | (Diff CCX: clwb, nt-store) | (Diff CCD: clwb, nt-store)
    # -----------------------
    cluster_types = ["Same CCX", "Diff CCX", "Diff CCD"]
    instr_types = ["clwb", "nt-store"]
    
    # Now groups are organized by cluster first, then instruction
    groups = [(c, i) for c in cluster_types for i in instr_types]
    
    x = np.arange(len(groups))
    n_bars = 4
    width = 0.19
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * width

    def get_stats(cluster: str, instr: str, idx: int) -> dict[str, float]:
        # Map display cluster names to CSV cluster keys
        cluster_map = {
            "Same CCX": "sameCCX",
            "Diff CCX": "diffCCX",
            "Diff CCD": "diffCCD",
        }
        csv_cluster = cluster_map.get(cluster, cluster)
        try:
            return data[csv_cluster][instr][idx]
        except KeyError:
            return {"median": np.nan, "low": np.nan, "high": np.nan}

    # -----------------------
    # Colors: gray (solo) + pink shades for attackers
    # Hatch: solid for clwb, hatched for nt-store
    # -----------------------
    attacker_colors = [
        "#9A9A9A",  # Solo (gray)
        "#DAA1AC",  # 1 attacker (pink)
        "#cd808c",  # 2 attackers (medium pink)
        "#bc5566",  # 3 attackers (dark pink)
    ]

    # Hatch per instruction
    hatch_map = {
        "clwb": None,       # Solid
        "nt-store": "//",   # Hatched
    }

    # -----------------------
    # Plot
    # -----------------------
    fig, ax = plt.subplots(figsize=(7, 2.5))

    for i in range(n_bars):
        stats = [get_stats(cluster, instr, i) for cluster, instr in groups]
        heights = [s["median"] for s in stats]
        lows = [s["low"] for s in stats]
        highs = [s["high"] for s in stats]

        yerr = np.array([
            np.maximum(0.0, np.array(heights) - np.array(lows)),
            np.maximum(0.0, np.array(highs) - np.array(heights)),
        ])

        # Apply hatch based on instruction (not cluster)
        hatches = [hatch_map[instr] for _, instr in groups]

        bars = ax.bar(
            x + offsets[i],
            heights,
            width=width,
            label=attack_labels[i],
            color=attacker_colors[i],
            edgecolor="#3a3a3a",
            linewidth=0.6,
            yerr=yerr,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "capsize": 2},
            zorder=2,
        )

        # Apply hatch per bar (instruction-based)
        for b, hatch in zip(bars, hatches):
            if hatch:
                b.set_hatch(hatch)

    # -----------------------
    # Axes formatting
    # -----------------------
    ax.set_yscale("log")
    _base_formatter = LogFormatterMathtext()
    def _log_tick_formatter(val, pos):
        if np.isclose(val, 1.0):
            return "1"
        return _base_formatter(val, pos)
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_formatter))
    ax.set_ylim(0, 5000)
    ax.set_ylabel("Slowdown")

    # Set x-ticks in the center of each cluster group
    xtick_positions = [0.5, 2.5, 4.5]  # centers for Same CCX, Diff CCX, Diff CCD
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(["Same CCX", "Diff CCX", "Diff CCD"])

    # Dividers between cluster groups
    ax.axvline(1.5, color="gray", linewidth=1.2, zorder=3)
    ax.axvline(3.5, color="gray", linewidth=1.2, zorder=3)
    
    ax.axhline(1, color="red", linestyle="--", linewidth=0.8, zorder=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y", zorder=0)
    ax.set_axisbelow(True)

    # -----------------------
    # Legends
    # -----------------------
    # 1) Attacker-count legend (colors) - unchanged
    # leg1 = ax.legend(
    #     frameon=False,
    #     ncol=4,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.35),
    # )

    # 2) Instruction legend (hatches) - now for clwb vs nt-store
    clwb_patch = mpatches.Patch(
        facecolor="white", edgecolor="#3a3a3a", hatch=None, label="write+clwb"
    )
    nts_patch = mpatches.Patch(
        facecolor="white", edgecolor="#3a3a3a", hatch="//", label="nt-store"
    )

    leg2 = ax.legend(
        handles=[clwb_patch, nts_patch],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        fontsize=11,
        handlelength=1.5,
        handleheight=0.8,
        borderpad=0.2,
        labelspacing=0.3,
    )

    # ax.add_artist(leg1)  # keep both legends

    # -----------------------
    # Manual spacing
    # -----------------------
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.28)

    # -----------------------
    # Output / show
    # -----------------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=400, bbox_inches="tight")
        print(f"Wrote plot to {out_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())