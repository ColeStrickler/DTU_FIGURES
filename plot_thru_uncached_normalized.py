import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/uncached_region_benchmark.csv")

# Normalize times by cached time for each (benchmark, data_size)
df["norm_time"] = df.groupby(["benchmark", "data_size"])["time"].transform(lambda x: x / x[df["type"] == "cached"].values[0])

# Colors: uncached=black, cached=gray
colors = {"cached": "gray", "uncached": "black"}

benchmarks = df["benchmark"].unique()
types = df["type"].unique()
data_sizes = sorted(df["data_size"].unique())

fig, axes = plt.subplots(1, len(benchmarks), figsize=(12, 5), sharey=True)

for ax, bench in zip(axes, benchmarks):
    subset = df[df["benchmark"] == bench]
    bar_width = 0.35
    x = range(len(data_sizes))

    for i, dtype in enumerate(types):
        data = subset[subset["type"] == dtype].sort_values("data_size")
        offset = (-0.5 + i) * bar_width
        ax.bar(
            [xi + offset for xi in x],
            data["norm_time"],
            width=bar_width,
            label=dtype,
            color=colors[dtype]
        )

    ax.set_xticks(range(len(data_sizes)))
    ax.set_xticklabels(data_sizes)
    ax.set_title(f"{bench.capitalize()} (normalized)")
    ax.set_xlabel("Data Size")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

axes[0].set_ylabel("Normalized exec. time")
axes[0].legend(title="Type")
plt.tight_layout()
plt.show()