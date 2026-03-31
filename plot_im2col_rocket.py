import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter
color_cpu       =         "#7FAFD4"
color_transform =         "#5588AA"
color_dtu       =         "#345670" 

# Greens 
# color_cpu       =         "#8FC8A9"
# color_transform =         "#5A9E78"
# color_dtu       =         "#356B45" 

edge = "#2a2a2a"

bar_width = 0.1
x_axis_width_scale = 0.25
fig_height_scale = 3
fig_width_scale = 4



# -----------------------------
# Load + clean
# -----------------------------
df = pd.read_csv("data/im2col_rocket1.0.csv", skipinitialspace=True)

df["cycle"] = pd.to_numeric(df["cycle"])
df["benchmark"] = df["benchmark"].str.strip()
df["type"] = df["type"].str.strip()

# -----------------------------
# Extract image size
# -----------------------------
sizes = df["benchmark"].str.extract(r'im2col_(\d+)_')[0] + "x" + df["benchmark"].str.extract(r'im2col_(\d+)_(\d+)')[1]
batch_size = df["benchmark"].str.extract(r'im2col_(\d+)_(\d+)_(\d+)_(\d+)')[3]

print(batch_size)
df["img_size"] = sizes

df["batch_size"] = batch_size.astype(int)
#print(df["batch_size"])
# ---------------------

print("here\n")
unique_sizes = sorted(df["img_size"].unique())
#print(unique_sizes)


def label_total_bar(ax):
    # assume CPU stacked bars are the first two containers
    cpu_containers = ax.containers[:2]  # base + transform
    n_bars = len(cpu_containers[0])
    
    for i in range(n_bars):
        total_height = sum(container[i].get_height() for container in cpu_containers)
        x = cpu_containers[0][i].get_x() + cpu_containers[0][i].get_width()/2
        ax.text(
            x,
            total_height * 1.02,
            f"{total_height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8
        )


def plot_ax(ax, pivot_mean, index, xlabel,ylabel):
    # Label x-axis with benchmark names
    ax.set_xticks(np.arange(len(pivot_mean))*x_axis_width_scale)
    ax.set_xticklabels(index, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

    ax.set_ylim(0.0, 4)         # set lower and upper limits

    # Define ticks you want explicitly
    ax.set_yticks([0.5, 1, 2, 4])
    ax.yaxis.set_major_formatter(ScalarFormatter())  # show normal numbers instead of scientific

    # Title for this subplot (optional)
    ax.set_title(f"Image size {size}", fontsize=12, fontweight="bold")

    # Legend
    #ax.legend(["DTU","CPU Base", "CPU Transform"], fontsize=6)



def hatch_ax(ax):
    # Optional: add hatch to transform portion to make it visually distinct
    for i, container in enumerate(ax.containers):
        if i == 1:  # second stack (transform)
            for bar in container:
                bar.set_hatch('//')




# Step 2: create a subplot for each image size, sharing the y-axis
fig, axes = plt.subplots(
    1, len(unique_sizes),      # one row, multiple columns
    figsize=(fig_width_scale * len(unique_sizes), fig_height_scale),
    sharey=True                # this is the shared y-axis
)

# Step 3: if only one size, axes is not a list, so wrap it
if len(unique_sizes) == 1:
    axes = [axes]

#print(axes)  # just to check we have the correct axes objects
i = 0
for ax, size in zip(axes, unique_sizes):
    # Select only rows for this image size
    sub_df = df[df["img_size"] == size]


    # Aggregate per benchmark + type
    grouped = sub_df.groupby(["benchmark", "type"]).agg( # combine rows for the same benchmark and type (CPU or DTU).
    cycle_mean=("cycle", "mean"), # calculate mean and standard deviation for cycle and transform_cost.
    cycle_std=("cycle", "std"),
    transform_mean=("transform_cost", "mean"),
    transform_std=("transform_cost", "std")
    ).reset_index() # make back into normal data frame



   # print(grouped)


        # pivot so the type column is separated into dtu+cpu
    pivot_mean = grouped.pivot(index="benchmark", columns="type", values="cycle_mean")
    pivot_std  = grouped.pivot(index="benchmark", columns="type", values="cycle_std")
    transform_mean = grouped.pivot(index="benchmark", columns="type", values="transform_mean")

    #print(pivot_mean)
    #print(pivot_std)
    #print(transform_mean)

    # Fraction of CPU spent on transform vs base execution
    pivot_mean["transform_n"] = transform_mean["cpu"] / pivot_mean["cpu"]  # transform fraction
    pivot_mean["base_cpu"] = (1.0 - pivot_mean["transform_n"]) * (pivot_mean["cpu"] / pivot_mean["dtu"])               # remaining execution
    pivot_mean["transform_norm"] = pivot_mean["transform_n"] * (pivot_mean["cpu"] / pivot_mean["dtu"])
    pivot_mean["total"] = pivot_mean["base_cpu"] + pivot_mean["transform_norm"]
    print(pivot_mean)
    print( (pivot_mean["cpu"] / pivot_mean["dtu"])  )
    pivot_mean["dtu"] = 1.0

    # Keep the benchmark order sorted by batch_size
    benchmark_order = sub_df.groupby("benchmark")["batch_size"].mean().sort_values().index

    # Reindex pivot_mean so rows are in this order
    pivot_mean = pivot_mean.reindex(benchmark_order)
    pivot_std  = pivot_std.reindex(benchmark_order)
    transform_mean = transform_mean.reindex(benchmark_order)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
    ax.set_yscale("log", base=2)
    x = np.arange(len(pivot_mean))*x_axis_width_scale  # numeric positions for each benchmark


    # CPU stacked bars
    ax.bar(
        x + bar_width/2, 
        pivot_mean["base_cpu"], 
        width=bar_width, 
        color=color_cpu,
        edgecolor=edge,
        label="CPU Base"
    )
    ax.bar(
        x + bar_width/2, 
        pivot_mean["transform_norm"], 
        width=bar_width, 
        bottom=pivot_mean["base_cpu"], 
        color=color_transform,
        edgecolor=edge,
        label="CPU Transform",
        hatch='//'
    )

    # DTU bar (always 1)
    ax.bar(
        x - bar_width/2,
        pivot_mean["dtu"],
        width=bar_width,
        color=color_dtu,
        edgecolor=edge,
        label="DTU"
    )
    batch_labels = sub_df.groupby("benchmark")["batch_size"].mean().loc[benchmark_order].astype(int)


    plot_ax(ax, pivot_mean, batch_labels, "Kernel Size", "Normalized Exec. Time")
    label_total_bar(ax)
    



plt.tight_layout(pad=2.0)
#fig.subplots_adjust(right=0.85)  # leave space for legend on right
fig.legend(
    ["DTU", "CPU Base", "CPU Transform"],  # labels
    loc="upper center",                   # position above all subplots
    ncol=3,                               # spread horizontally
    fontsize=10,
    frameon=False                         # optional: no box around legend
)
fig.text(
    0.02,      # x position (slightly left of the figure)
    0.55,       # y position (centered vertically)
    "Normalized Exec. Time",
    va='center', ha='center',
    rotation='vertical',
    fontsize=12,
    fontweight='bold'
)
#fig.set_yticks([1, 2, 4, 8, 16, 32,64,128])
#fig.get_yaxis().set_major_formatter(plt.ScalarFormatter())
#fig.set_ylabel("Normalized Exec. Time", fontsize=12, fontweight="bold")


plt.savefig("figures/im2col_rocket.png", bbox_inches="tight")
plt.savefig("figures/im2col_rocket.pdf", bbox_inches="tight")
plt.show()