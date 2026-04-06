import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter

color_baseline = "#8FC8A9"  # brighter, more turquoise than #8FC8A9
color_savings   = "#5A9E78"  # deeper, complementary turquoise
edge = "#2a2a2a"


    

def im2col_matsize(cin, height, width, ksize):
    return cin * (height - ksize + 1) * (width - ksize + 1) * ksize * ksize 


def base_imgsize(cin,height,width):
        return cin*height*width


def img_aug():
    return 1/8;

def unfold():
    return 1/2;







def im2col_matsize(cin, height, width, ksize):
    return cin * (height - ksize + 1) * (width - ksize + 1) * ksize * ksize 


def base_imgsize(cin,height,width):
        return cin*height*width

def im2col_usage(cin,height,width,ksize):
    return base_imgsize(3,256,256) / im2col_matsize(3,256,256,ksize)
def im2col_usage2(cin,height,width,ksize):
    return  im2col_matsize(3,256,256,ksize) / base_imgsize(3,256,256)


def img_aug():
    return 1/8;

def unfold():
    return 1/2;











def hatch_ax(ax):
    # Optional: add hatch to transform portion to make it visually distinct
    for i, container in enumerate(ax.containers):
        if i == 0:  # second stack (transform)
            for bar in container:
                bar.set_hatch('//')



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




data = []

for ksize in [2,3,4,5]:
    ratio = base_imgsize(3,256,256) / im2col_matsize(3,256,256,ksize)
    data.append({
        "benchmark": f"Im2Col_{ksize}x{ksize}",
        "memory usage": ratio,
        "type": "savings"
    })
    data.append({
        "benchmark": f"Im2Col_{ksize}x{ksize}",
        "memory usage": 1.0,
        "type": "base"
    })

data.append({
    "benchmark": "ImageAug",
    "memory usage": img_aug(),
    "type": "savings"
})

data.append({
    "benchmark": "TensorUnfold",
    "memory usage": unfold(),
    "type": "savings"
})

data.append({
    "benchmark": "ImageAug",
    "memory usage": 1.0,
    "type": "base"
})

data.append({
    "benchmark": "TensorUnfold",
    "memory usage": 1.0,
    "type": "base"
})

df = pd.DataFrame(data).reset_index()
# Pivot so each type is a column
pivot = df.pivot(index="benchmark", columns="type", values="memory usage").fillna(0)

# Plot side-by-side bars
x = np.arange(len(pivot))       # positions for benchmarks
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8,5))


# Savings bars
ax.bar(x + bar_width/2, pivot["savings"], width=bar_width, color=color_savings, edgecolor=edge, label="Savings")

# Base bars
ax.bar(x - bar_width/2, pivot["base"], width=bar_width, color=color_baseline, edgecolor=edge, label="Base")




ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
label_total_bar(ax)
hatch_ax(ax)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_ylabel("Normalized Memory Usage", fontsize=12, fontweight="bold")
ax.set_title("Basline vs. DTU Memory Usage Comparison", fontsize=12, fontweight="bold")
fig.legend(
    [ "Baseline Memory Usage", "DTU Memory Usage" ],  # labels
    loc="upper center",                   # position above all subplots
    ncol=4,                               # spread horizontally
    fontsize=10,
    frameon=False                         # optional: no box around legend
)

ax.set_ylim(0,30)

plt.tight_layout(pad=3.0)


plt.savefig("figures/memory_savings.png", bbox_inches="tight")
plt.savefig("figures/memory_savings.pdf", bbox_inches="tight")
plt.show()