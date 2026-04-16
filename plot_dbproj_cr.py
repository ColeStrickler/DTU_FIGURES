import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib.patches import Patch
color_cpu       =         "#DAA1AC"
color_transform =         "#cd808c"
color_dtu       =         "#bc5566" 
color_savings       =    "#7FAFD4"     
color_baseline =         "#345670" 
  
color_cpu       = "#DAA1AC"   # Main magenta (bright but soft)
color_inplace = "#cd808c"   # Mid-tone magenta
color_dtu       = "#bc5566"   # Darker magenta




# Greens 
# color_cpu       =         "#8FC8A9"
# color_transform =         "#5A9E78"
# color_dtu       =         "#356B45" 

edge = "#2a2a2a"

bar_width = 0.07
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



def label_total_bar(ax):
    # assume CPU stacked bars are the first two containers
    cpu_container = ax.containers[0]  # base + transform
    n_bars = len(cpu_container)
    
    for bar in cpu_container:
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2

        ax.text(
            x,
            height * 1.02,
            f"{height:.2f}",
            ha='center',
            va='bottom',
            fontsize=8
        )




def label_total_bar2(ax):
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



def plot_ax(ax, pivot_mean, index, xlabel,ylabel, title=f"Image size {'720x1080'}"):
    # Label x-axis with benchmark names
    ax.set_xticks(np.arange(len(pivot_mean))*x_axis_width_scale)
    ax.set_xticklabels(index, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")


    # Define ticks you want explicitly
    #ax.set_yticks([1, 2, 3, 4])
    ax.yaxis.set_major_formatter(ScalarFormatter())  # show normal numbers instead of scientific

    # Title for this subplot (optional)
    #ax.set_title(title, fontsize=12, fontweight="bold")

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
    1, 2,      # one row, multiple columns
    figsize=(fig_width_scale * 2, fig_height_scale),
    sharey=False                # this is the shared y-axis
)


#print(axes)  # just to check we have the correct axes objects
i = 0
ax = axes[0]

ax.set_ylim(0.0, 6)
print(df['row_size'])
# Select only rows for this image size
size = 256
sub_df = df[df["row_size"] == '128']
# Aggregate per benchmark + type
grouped = sub_df.groupby(["benchmark", "type"]).agg( # combine rows for the same benchmark and type (CPU or DTU).
cycle_mean=("cycle", "mean"), # calculate mean and standard deviation for cycle and transform_cost.
cycle_std=("cycle", "std"),
transform_mean=("transform_cost", "mean"),
transform_std=("transform_cost", "std")
).reset_index() # make back into normal data frame


pivot_mean = grouped.pivot(index="benchmark", columns="type", values="cycle_mean")
pivot_std  = grouped.pivot(index="benchmark", columns="type", values="cycle_std")
transform_mean = grouped.pivot(index="benchmark", columns="type", values="transform_mean")
# Fraction of CPU spent on transform vs base execution


pivot_mean["base_cpu"] = pivot_mean["cpu"] / pivot_mean["dtu"]
pivot_mean["base_col"] = pivot_mean["col"] / pivot_mean["dtu"] 
pivot_mean["base_dtu"] = 1.0


print( (pivot_mean["cpu"] / pivot_mean["dtu"])  )
pivot_mean["dtu"] = 1.0
# Keep the benchmark order sorted by batch_size
benchmark_order = sub_df.groupby("benchmark")["Columns"].mean().sort_values().index
# Reindex pivot_mean so rows are in this order
pivot_mean = pivot_mean.reindex(benchmark_order)
pivot_std  = pivot_std.reindex(benchmark_order)
transform_mean = transform_mean.reindex(benchmark_order)
ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
#ax.set_yscale("log", base=2)
x = np.arange(len(pivot_mean))*x_axis_width_scale  # numeric positions for each benchmark
# CPU stacked bars



        # CPU stacked bars
ax.bar(
    x + bar_width, 
    pivot_mean["base_cpu"], 
    width=bar_width, 
    color=color_cpu,
    edgecolor=edge,
    label="Row"
)
ax.bar(
    x,
    pivot_mean["base_col"],
    width=bar_width,
    color=color_inplace,
    edgecolor=edge,
    label="Col"
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
plot_ax(ax, pivot_mean, batch_labels, "Columns Projected", "Normalized Exec. Time", "DTU Vs. CPU Execution Time")
label_total_bar(ax)



plt.tight_layout(pad=3.0)
#fig.subplots_adjust(right=0.85)  # leave space for legend on right
from matplotlib.patches import Patch

handles = [
    Patch(facecolor=color_dtu, edgecolor="black"),                     # DTU
    Patch(facecolor=color_inplace, edgecolor="black"),                     # CPU Base
    Patch(facecolor=color_cpu, edgecolor="black") # CPU Transform
]

labels = ["DTU", "Col", "Row"]

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.38),
    ncol=3,
    fontsize=8,
    columnspacing=0.8,
)


ax.text(
    -0.15, 0.5,
    "Normalized Exec. Time",
    va='center',
    ha='center',
    rotation='vertical',
    transform=ax.transAxes,
    fontsize=12,
    fontweight='bold'
)
#fig.set_yticks([1, 2, 4, 8, 16, 32,64,128])
#fig.get_yaxis().set_major_formatter(plt.ScalarFormatter())
#fig.set_ylabel("Normalized Exec. Time", fontsize=12, fontweight="bold")


###############
# PLOT 2      #
###############

ax = axes[1]


def label_total_bar2(ax):
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



data = []

def img_aug2():
    return 8;

for batch_size in [1,2,3,4,5,6,7,8]:
    data.append({
        "benchmark": batch_size,
        "memory usage": 1.0,
        "type": "savings"
    })

    data.append({
        "benchmark": batch_size,
        "memory usage": 1,
        "type": "base"
    })



df = pd.DataFrame(data).reset_index()
# Pivot so each type is a column
pivot = df.pivot(index="benchmark", columns="type", values="memory usage").fillna(0)

# Plot side-by-side bars
x = np.arange(len(pivot_mean)) * x_axis_width_scale
bar_width = 0.1

# Base bars
ax.bar(x + bar_width/2, pivot["base"], width=bar_width, color=color_baseline, edgecolor=edge, label="Base")


# Savings bars
ax.bar(x - bar_width/2, pivot["savings"], width=bar_width, color=color_savings, edgecolor=edge, label="Savings")




ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
label_total_bar2(ax)
#hatch_ax(ax)
#hatch_ax2(ax)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=10, fontweight="bold")

ax.set_ylabel("Normalized WSS Size", fontsize=12, fontweight="bold")
#ax.set_title("CPU vs. DTU WSS Size", fontsize=12, fontweight="bold")

ax.set_ylim(0.0,2.5)
ax.set_yticks([0,1,2])
# Set y-axis to logarithmic scale
#ax.set_yscale('log', base=2)

# Optional: customize the ticks (base 10)

handles = [
    Patch(facecolor=color_savings, edgecolor="black"),                     # DTU
    Patch(facecolor=color_baseline, edgecolor="black"),                     # CPU Base
]

labels =     ["w/ DTU", "CPU only"]

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.38),
    ncol=2,
    fontsize=8,
    columnspacing=0.8,
)



ax.set_xlabel("Columns Projected", fontsize=12, fontweight="bold")
# Define ticks you want explicitly
#ax.set_yticks([1, 2, 3, 4])
# Title for this subplot (optional)
#ax.set_title("DTU vs. CPU WSS", fontsize=12, fontweight="bold")
# Legend
#ax.legend(["DTU","CPU Base", "CPU Transform"], fontsize=6)



plt.savefig("figures/dbproj_boom_cr.png", bbox_inches="tight")
plt.savefig("figures/dbproj_boom_cr.pdf", bbox_inches="tight")
plt.show()