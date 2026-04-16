import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/avg_memory_traffic_boom.csv")

df.columns = df.columns.str.strip()
df["benchmark"] = df["benchmark"].astype(str).str.strip()
df["type"] = df["type"].str.strip()

df = df.dropna(how="all")

metrics = ["L1Refill", "L1WB", "LLCRefill", "LLCWB"]

for m in metrics:
    df[m] = pd.to_numeric(df[m], errors="coerce")

df = df.dropna(subset=["benchmark", "type"])

# collapse duplicates
df = df.groupby(["benchmark", "type"], as_index=False).mean()

benchmarks = sorted(df["benchmark"].unique())
x = np.arange(len(benchmarks))
width = 0.35

# -----------------------------
# Paper style settings
# -----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "legend.fontsize": 8,

    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "hatch.linewidth": 0.8,

    # Make global text bold
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

colors = ["#DAA1AC","#bc5566" ]   # CPU, DTU
hatches = ['///', 'xxx']          # CPU, DTU

# -----------------------------
# Figure (LaTeX-friendly)
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), sharex=True)
axes = axes.flatten()





for i, metric in enumerate(metrics):
    ax = axes[i]

    cpu = df[df["type"] == "cpu"].set_index("benchmark").reindex(benchmarks)[metric]
    dtu = df[df["type"] == "dtu"].set_index("benchmark").reindex(benchmarks)[metric]
        # DTU bars
    dtu_bars = ax.bar(
        x - width/2,
        dtu,
        width,
        label="DTU",
        color=colors[1],
        hatch=hatches[1],
        edgecolor="black",
        linewidth=0.6
    )


    # CPU bars
    ax.bar(
        x + width/2,
        cpu,
        width,
        label="CPU",
        color=colors[0],
        hatch=hatches[0],
        edgecolor="black",
        linewidth=0.6
    )



    for bar in dtu_bars:
        height = bar.get_height()
        if np.isnan(height):
            continue

        ax.text(
            bar.get_x() + 0.8 * bar.get_width() / 2,
            height + 0.02 * height,   # small vertical offset
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    if i == 0 or i == 2:
        ax.set_ylabel(f"Normalized Traffic", fontsize=11)
    ax.set_title(metric)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

# -----------------------------
# Shared x-axis
# -----------------------------
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.set_ylim(0, 1.15)

legend_elements = [
    Patch(facecolor="#bc5566" , edgecolor="black", hatch="xxx", label="DTU"),
    Patch(facecolor="#DAA1AC", edgecolor="black", hatch="///", label="CPU"),
]

labels = ["w/ DTU", "CPU Only"]

fig.legend(
    handles=legend_elements,
    labels = labels,
    loc="upper center",
    bbox_to_anchor=(0.535, 1.02),
    ncol=2,
)

plt.tight_layout(pad=1.0)

# Export for LaTeX
plt.savefig("figures/avg_cache_stats.pdf", bbox_inches="tight")
plt.savefig("figures/avg_cache_stats.png", dpi=300, bbox_inches="tight")

plt.show()