import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/data_boom1.0.csv", skipinitialspace=True)

# Clean columns
df["cycle"] = pd.to_numeric(df["cycle"])
df["benchmark"] = df["benchmark"].str.strip()
df["type"] = df["type"].str.strip()

# Pivot
pivot = df.pivot(index="benchmark", columns="type", values="cycle")

pivot["speedup"] = pivot["cpu"] / pivot["dtu"]
pivot["cpu"] = 1.0   # cleaner than dividing by itself

plot_df = pivot[["cpu", "speedup"]]
# Add labels


ax = plot_df.plot(
    kind="bar",
    figsize=(16,7),
    color={
        "cpu": "darkgray",
        "speedup": "xkcd:ruby"
    },
    edgecolor="black"
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)


plt.axhline(1.0, color="black", linewidth=2)
plt.title("DTU Speedup BOOM Core @ 1.0GHz")
plt.ylabel("Speedup (CPU normalized)")
plt.xlabel("Benchmark")

plt.tight_layout()
plt.savefig("figures/dtu_boom.png", bbox_inches="tight")
plt.show()
