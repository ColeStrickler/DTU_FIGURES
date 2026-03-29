import pandas as pd
import matplotlib.pyplot as plt

color_cpu       = "#869098"
color_transform = "#869098"  # pale yellow
color_dtu       = "xkcd:ocean"

# Load CSV
df = pd.read_csv("data/rocket_prefetch1.0.csv", skipinitialspace=True)

# Clean columns
df["cycle"] = pd.to_numeric(df["cycle"])
df["benchmark"] = df["benchmark"].str.strip()
df["type"] = df["type"].str.strip()

# Pivot
pivot = df.pivot(index="benchmark", columns="type", values="cycle")
transform = df.pivot(index="benchmark", columns="type", values="transform_cost")

pivot["transform_cpu"] = transform["cpu"]
pivot["transform_dtu"] = transform["dtu"]

pivot["transform_norm"] = pivot["transform_cpu"] / pivot["cpu"]
pivot["speedup_dtu"] = pivot["dtu"] / pivot["cpu"] 
pivot["base_cpu"] = 1.0 - pivot["transform_norm"]

plot_df = pivot[["base_cpu", "transform_norm", "speedup_dtu"]]

# Plot
fig, ax = plt.subplots(figsize=(16,9))

# CPU (stacked)
pivot[["base_cpu", "transform_norm"]].plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=[color_cpu, color_transform],
    edgecolor="black",
    width=0.2,
    position=1
)

# Add hatch to transform
for i, container in enumerate(ax.containers):
    if i == 1:  # transform_norm is second in the stacked bars
        for bar in container:
            bar.set_hatch('//')

# DTU speedup (not stacked)
pivot["speedup_dtu"].plot(
    kind="bar",
    ax=ax,
    color=color_dtu,
    edgecolor="black",
    width=0.2,
    position=0
)

ax.set_xlim(-0.5, len(pivot) - 0.5)
plt.axhline(1.0, color="black", linewidth=2)

# --- Make all text bold and larger ---
title_font = {'fontsize': 20, 'fontweight': 'bold'}
label_font = {'fontsize': 16, 'fontweight': 'bold'}
tick_font = {'labelsize': 12, 'fontweight': 'bold'}

ax.set_title("DTU Speedup Rocket Core", **title_font)
ax.set_xlabel("Benchmark", **label_font)
ax.set_ylabel("Normalized Exec. Time", **label_font)
ax.tick_params(axis='x', labelsize=tick_font['labelsize'], labelrotation=45)
ax.tick_params(axis='y', labelsize=tick_font['labelsize'])
# Set x-ticks at the center of the group
ax.set_xticks(range(len(pivot)))
ax.set_xticklabels(pivot.index, rotation=90,ha="right", fontweight='bold', fontsize=14)
# Optionally, make legend bold
leg = ax.legend(fontsize=16)
for text in leg.get_texts():
    text.set_fontweight('bold')

plt.tight_layout()
plt.savefig("figures/dtu_rocket.png", bbox_inches="tight")
plt.savefig("figures/dtu_rocket.pdf", bbox_inches="tight")
plt.show()