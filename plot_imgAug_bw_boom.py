import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter
color_cpu       = "#E07ABF"   # Main magenta (bright but soft)
color_transform = "#C15A9F"   # Mid-tone magenta
color_dtu       = "#9C3F7D"   # Darker magenta
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
df = pd.read_csv("data/img_augmentation2_boom1.0.csv", skipinitialspace=True)
#print(df)
df["cycle"] = pd.to_numeric(df["cycle"])
df["benchmark"] = df["benchmark"].str.strip()
df["type"] = df["type"].str.strip()

# -----------------------------
# Extract image size
# -----------------------------
# -----------------------------
sizes = df["benchmark"].str.extract(r'img_augmentation_(\d+)_')[0]
batch_size = df["benchmark"].str.extract(r'img_augmentation_(\d+)_(\d+)')[1]
#print(batch_size)
df["img_size"] = sizes + "x" + sizes
df["img_size_num"] = sizes.astype(int)
df["batch_size"] = batch_size.astype(int)
#print(df["batch_size"])
# ---------------------
print(df)

unique_sizes = sorted(df["img_size_num"].unique())
#print(unique_sizes)


def label_total_bar(ax):
    # assume CPU stacked bars are the first two containers
    cpu_containers = ax.containers[:2]  # base + transform
    n_bars = len(cpu_containers[0])
    
    for i in range(n_bars):
        total_height = cpu_containers[1][i].get_height() 
        x = cpu_containers[1][i].get_x() + cpu_containers[1][i].get_width()/2
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
    ax.set_xticklabels(index, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")

    ax.set_ylim(0.0, 4)         # set lower and upper limits

    # Define ticks you want explicitly
    ax.set_yticks([1, 2, 3, 4,5,6])
    ax.yaxis.set_major_formatter(ScalarFormatter())  # show normal numbers instead of scientific

    # Title for this subplot (optional)
    ax.set_title(f"ImgAug Total Accesses {size}x{size}", fontsize=12, fontweight="bold")

    # Legend
    #ax.legend(["DTU","CPU Base", "CPU Transform"], fontsize=6)



def hatch_ax(ax):
    # Optional: add hatch to transform portion to make it visually distinct
    for i, container in enumerate(ax.containers):
        if i == 1:  # second stack (transform)
            for bar in container:
                bar.set_hatch('//')


total_for_savings = 0

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
    sub_df = df[df["img_size_num"] == size]


    # Aggregate per benchmark + type
    grouped = sub_df.groupby(["benchmark", "type"]).agg( # combine rows for the same benchmark and type (CPU or DTU).
    bw_mean=("RegularDRAMAccess", "mean"), # calculate mean and standard deviation for cycle and transform_cost.
    dtubw_mean=("DTUDramAccess", "mean"),
    cycle_std=("cycle", "std"),
    transform_mean=("transform_cost", "mean"),
    transform_std=("transform_cost", "std")
    ).reset_index() # make back into normal data frame



    # print(grouped)
    # pivot so the type column is separated into dtu+cpu
    pivot_mean = grouped.pivot(index="benchmark", columns="type", values="dtubw_mean")
    pivot_std  = grouped.pivot(index="benchmark", columns="type", values="bw_mean")
   # transform_mean = grouped.pivot(index="benchmark", columns="type", values="transform_mean")

    #print(pivot_mean)
    #print(pivot_std)
    #print(transform_mean)

    # Fraction of CPU spent on transform vs base execution
    

    pivot_mean["base_cpu"] = 1.0 

    pivot_mean["base_dtu"] = (pivot_mean["dtu"] + pivot_std["dtu"]) / pivot_std["cpu"]  # transform fraction
    print(pivot_std)
    print("\n\n")
    print(pivot_mean)

    total_for_savings += pivot_mean["base_dtu"].mean()
    # Keep the benchmark order sorted by batch_size
    benchmark_order = sub_df.groupby("benchmark")["batch_size"].mean().sort_values().index

    # Reindex pivot_mean so rows are in this order
    pivot_mean = pivot_mean.reindex(benchmark_order)
    pivot_std  = pivot_std.reindex(benchmark_order)
    # = transform_mean.reindex(benchmark_order)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
    #ax.set_yscale("log", base=2)
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
    # DTU bar (always 1)
    ax.bar(
        x - bar_width/2,
        pivot_mean["base_dtu"],
        width=bar_width,
        color=color_dtu,
        edgecolor=edge,
        label="DTU"
    )
    batch_labels = sub_df.groupby("benchmark")["batch_size"].mean().loc[benchmark_order].astype(int)


    plot_ax(ax, pivot_mean, batch_labels, "Batch Size", "Normalized # DRAM Accesses")
    label_total_bar(ax)
    



plt.tight_layout(pad=3.0)
#fig.subplots_adjust(right=0.85)  # leave space for legend on right
fig.legend(
    ["CPU", "CPU", "DTU"],  # labels
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

total_for_savings /= len(unique_sizes)
total_df = pd.read_csv("data/avg_memory_traffic_boom.csv")

dtu_row = {
    "benchmark" : "img_augmentation",
    "type": "dtu",
    "memtraffic": total_for_savings
}

cpu_row = {
    "benchmark" : "img_augmentation",
    "type": "cpu",
    "memtraffic": 1.0
}


# Append
total_df = pd.concat([total_df, pd.DataFrame([dtu_row, cpu_row])], ignore_index=True)

# Save back
total_df.to_csv("data/avg_memory_traffic_boom.csv", index=False)



plt.savefig("figures/imAug_bw_boom.png", bbox_inches="tight")
plt.savefig("figures/imAug_bw_boom.pdf", bbox_inches="tight")
plt.show()