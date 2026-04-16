import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/darknet_data.csv").set_index("type")

df["DRAM"] = df["RegularDRAMAccess"] + df["DTUDramAccess"]
df = df.drop(columns=["RegularDRAMAccess", "DTUDramAccess", "instret", "gemm_cycles"])

# -----------------------------
# Normalize to CPU
# -----------------------------
cpu = df.loc["cpu"]
df_norm = df.div(cpu)

metrics = df_norm.columns

# -----------------------------
# Paper style settings (IDENTICAL)
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

    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold"
})

colors = ["#DAA1AC", "#bc5566"]   # CPU, DTU
hatches = ['///', 'xxx']

# -----------------------------
# Layout: one subplot per metric
# -----------------------------
n_metrics = len(metrics)
ncols = 2
nrows = math.ceil(n_metrics / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(6.8, 2.6 * nrows), sharey=True)
axes = np.array(axes).flatten()

width = 0.04
x = np.array([0])  # single bar position per subplot

# -----------------------------
# Plot
# -----------------------------
for i, metric in enumerate(metrics):
    ax = axes[i]

    cpu_val = df_norm.loc["cpu", metric]
    dtu_val = df_norm.loc["dtu-6", metric]

    # DTU bars (LEFT)
    dtu_bars = ax.bar(
        x - width/2,
        [dtu_val],
        width,
        label="DTU",
        color=colors[1],
        hatch=hatches[1],
        edgecolor="black",
        linewidth=0.6
    )

    # CPU bars (RIGHT)
    cpu_bars = ax.bar(
        x + width/2,
        [cpu_val],
        width,
        label="CPU",
        color=colors[0],
        hatch=hatches[0],
        edgecolor="black",
        linewidth=0.6
    )
    ax.set_xlim(-0.075, 0.075)

    # -----------------------------
    # Value labels (same style)
    # -----------------------------
    for bar in dtu_bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.02 * h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


    ax.axhline(1.0, linestyle="--", linewidth=0.8, alpha=0.8, color='black')
    # -----------------------------
    # Formatting (matched)
    # -----------------------------
    ax.set_title(metric)
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.set_ylim(0, 1.4)

    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    if i == 0 or i == 2:
        ax.set_ylabel("Normalized Value")

# turn off unused axes
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

# -----------------------------
# Shared legend (IDENTICAL STYLE)
# -----------------------------
legend_elements = [
    Patch(facecolor=colors[1], edgecolor="black", hatch="xxx", label="DTU"),
    Patch(facecolor=colors[0], edgecolor="black", hatch="///", label="CPU"),
]

fig.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.535, .98),
    ncol=2,
)

plt.tight_layout(pad=3.0)

plt.savefig("figures/darknet_metrics.pdf", bbox_inches="tight")
plt.savefig("figures/darknet_metrics.png", dpi=300, bbox_inches="tight")

plt.show()