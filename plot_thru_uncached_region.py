import pandas as pd
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("data/uncached_region_benchmark.csv")

# Unique benchmarks
benchmarks = df["benchmark"].unique()
data_sizes = sorted(df["data_size"].unique())

fig, axes = plt.subplots(1, len(benchmarks), figsize=(12, 5), sharey=True)

bar_width = 0.35
colors = {"cached": "black", "uncached": "gray"}

for ax, bench in zip(axes, benchmarks):
    subset = df[df["benchmark"] == bench]

    for i, dtype in enumerate(["cached", "uncached"]):
        times = subset[subset["type"] == dtype].set_index("data_size").loc[data_sizes]["time"]
        ax.bar(
            [x + i * bar_width for x in range(len(data_sizes))],
            times,
            width=bar_width,
            label=dtype,
            color=colors[dtype],
        )

    ax.set_xticks([x + bar_width/2 for x in range(len(data_sizes))])
    ax.set_xticklabels(data_sizes)
    ax.set_title(f"{bench.capitalize()} Benchmark")
    ax.set_xlabel("Data size")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

axes[0].set_ylabel("Time (cycles)")
axes[0].legend(title="Type")

plt.tight_layout()
plt.show()
