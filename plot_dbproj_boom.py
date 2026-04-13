import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter










color_cpu       =         "#7FAFD4"
color_inplace =         "#5588AA"
color_dtu       =         "#345670" 



# Greens 
# color_cpu       =         "#8FC8A9"
# color_transform =         "#5A9E78"
# color_dtu       =         "#356B45" 

edge = "#2a2a2a"

bar_width = 0.25
x_axis_width_scale = 0.25
fig_height_scale = 3
fig_width_scale = 4



# -----------------------------
# Load + clean
# -----------------------------
df = pd.read_csv("data/dbproj_boom1.0.csv", skipinitialspace=True)

df["cycle"] = pd.to_numeric(df["cycle"])
df["benchmark"] = df["benchmark"].str.strip()
df["type"] = df["type"].str.strip()


# -----------------------------
# Extract image size
# -----------------------------
sizes = df["benchmark"].str.extract(r'_(\d+)$')[0]
#print(sizes)
df["row_size"] = sizes


mode = df["benchmark"].str.extract(r'_(\d+)_(\d+)$')[0]
print(mode)






df["Columns"] = mode.astype(int)
#print(df["mode"])
# ---------------------


unique_sizes = sorted(df["row_size"].unique())
unique_sizes.reverse()
print(unique_sizes)


def label_total_bar(ax):
    # assume CPU stacked bars are the first two containers
    cpu_containers = ax.containers[:2]  # base + transform
    n_bars = len(cpu_containers[0])
    
    for i in range(n_bars):
        total_height = cpu_containers[0][i].get_height()
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
    ax.set_xticks(np.arange(len(pivot_mean)))
    ax.set_xticklabels(index, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

    ax.set_ylim(0.0, 4.5)         # set lower and upper limits

    # Define ticks you want explicitly
    ax.set_yticks([ 1, 2, 3, 4])
    ax.yaxis.set_major_formatter(ScalarFormatter())  # show normal numbers instead of scientific

    # Title for this subplot (optional)
    ax.set_title(f"RowSize {int(size)*4} bytes", fontsize=12, fontweight="bold")

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
    sub_df = df[df["row_size"] == size]


    #sub_df['transform_cost'] = pd.to_numeric(sub_df['transform_cost'], errors='coerce')
    #sub_df['cycle'] = pd.to_numeric(sub_df['cycle'], errors='coerce')
    #sub_df.to_csv("debug_sub_df.csv", index=False)
    #print(f"here {unique_sizes}\n")
    # Aggregate per benchmark + type
    grouped = sub_df.groupby(["benchmark", "type"]).agg( # combine rows for the same benchmark and type (CPU or DTU).
    cycle_mean=("cycle", "mean"), # calculate mean and standard deviation for cycle and transform_cost.
    cycle_std=("cycle", "std"),
    transform_mean=("transform_cost", "mean"),
    transform_std=("transform_cost", "std")
    ).reset_index() # make back into normal data frame
    
    #@grouped.to_csv("debug_sub_df.csv", index=False)
    
   # print(grouped)

    #pivot_mean = grouped.pivot(index="benchmark", columns="type", values="dtubw_mean")
    #pivot_std  = grouped.pivot(index="benchmark", columns="type", values="bw_mean")


        # pivot so the type column is separated into dtu+cpu
    pivot_mean = grouped.pivot(index="benchmark", columns="type", values="cycle_mean")
    pivot_std  = grouped.pivot(index="benchmark", columns="type", values="cycle_std")
    transform_mean = grouped.pivot(index="benchmark", columns="type", values="transform_mean")

    pivot_mean["base_cpu"] = pivot_mean["cpu"] / pivot_mean["dtu"]
    pivot_mean["base_col"] = pivot_mean["col"] / pivot_mean["dtu"] 

    pivot_mean["base_dtu"] = 1.0


    print(pivot_mean)
    #print(pivot_std)
    #print(transform_mean)

    # Keep the benchmark order sorted by mode
    benchmark_order = sub_df.groupby("benchmark")["Columns"].mean().sort_values().index

    # Reindex pivot_mean so rows are in this order
    pivot_mean = pivot_mean.reindex(benchmark_order)
    pivot_std  = pivot_std.reindex(benchmark_order)

   #ax.set_yscale("log", base=2)
    x = np.arange(len(pivot_mean))# *x_axis_width_scale  # numeric positions for each benchmark


        # CPU stacked bars
    ax.bar(
        x + bar_width, 
        pivot_mean["base_cpu"], 
        width=bar_width, 
        color=color_cpu,
        edgecolor=edge,
        label="CPU Base"
    )

    ax.bar(
        x,
        pivot_mean["base_col"],
        width=bar_width,
        color=color_inplace,
        edgecolor=edge,
        label="Column"
    )
    

    # DTU bar (always 1)
    ax.bar(
        x - bar_width,
        pivot_mean["base_dtu"],
        width=bar_width,
        color=color_dtu,
        edgecolor=edge,
        label="DTU"
    )
    
    batch_labels = sub_df.groupby("benchmark")["Columns"].mean().loc[benchmark_order].astype(int)

    print(batch_labels)

    plot_ax(ax, pivot_mean, batch_labels, "Columns projected", "Normalized Exec. Time")
    label_total_bar(ax)
    
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")


plt.tight_layout(pad=2.0)
#fig.subplots_adjust(right=0.85)  # leave space for legend on right


handles = ax.containers  # bar containers only
labels = ["Row", "Col", "DTU"]
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
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


plt.savefig("figures/dbproj_boom.png", bbox_inches="tight")
plt.savefig("figures/dbproj_boom.pdf", bbox_inches="tight")
plt.show()