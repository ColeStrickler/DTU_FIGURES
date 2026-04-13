import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

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
df = pd.read_csv("data/avg_memory_traffic_boom.csv")
pivot = df.pivot(index="benchmark", columns="type", values="memtraffic")

fig, ax = plt.subplots(figsize=(7.2, 4.3))


colors = ["#7FAFD4", "#345670" ]
#colors = ['#4d4d4d', '#bfbfbf']
hatches = ['///', 'xxx']

pivot.plot(
    kind="bar",
    ax=ax,
    color=colors,
    edgecolor='black',
    width=0.8
)

# Apply hatching
for i, container in enumerate(ax.containers):
    for bar in container:
        bar.set_hatch(hatches[i % len(hatches)])

# --- Reference line at 1.0 ---
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1)

# Labels and styling
ax.set_ylabel("Avg. Total DRAM Traffic (Normalized)", fontsize=9)
ax.set_title("DTU vs CPU DRAM Traffic by Benchmark", fontsize=9)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel("")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
ax.set_axisbelow(True)
ax.set_ylim(0,1.6)

handles = ax.containers  # bar containers only
labels = ["CPU", "DTU"]
ax.legend(handles, labels)
# Optional: remove legend entirely

dtu_container = ax.containers[1]

for bar in dtu_container:
    height = bar.get_height()

    ax.text(
        bar.get_x() + 1.18 * bar.get_width() / 2,
        height + 0.02 * height,   # small vertical offset
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )


plt.tight_layout()

plt.savefig("figures/avg_memtraffic.png", bbox_inches="tight")
plt.savefig("figures/avg_memtraffic.pdf", bbox_inches="tight")
plt.show()