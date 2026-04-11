import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import LogLocator, ScalarFormatter

color_baseline = "#8FC8A9"   # brighter, more turquoise than #8FC8A9
color_savings   = "#4A7F6A"  # deeper, complementary turquoise
edge = "#2a2a2a"

def im2col_matsize(cin, height, width, ksize):
    return cin * (height - ksize + 1) * (width - ksize + 1) * ksize * ksize 


def base_imgsize(cin,height,width):
        return cin*height*width


def img_aug():
    return 1/8;

def unfold():
    return 1/2;




def vol2col_size(cin,height,width,depth,ksize):
    return cin*(height-ksize+1)*(width-ksize+1)*(depth-ksize+1)*ksize*ksize*ksize


def im2col_matsize(cin, height, width, ksize):
    return cin * (height - ksize + 1) * (width - ksize + 1) * ksize * ksize 


def base_imgsize(cin,height,width):
        return cin*height*width

def im2col_usage(cin,height,width,ksize):
    return base_imgsize(3,256,256) / im2col_matsize(3,256,256,ksize)
def im2col_usage2(cin,height,width,ksize):
    return  im2col_matsize(3,256,256,ksize) / base_imgsize(3,256,256)


def vol2col_usage(cin,height,width,depth,ksize):
        return vol2col_size(cin,height,width,depth,ksize) / (cin*height*width*depth)


def img_aug2():
    return 8;

def unfold2():
    return 2;











def hatch_ax(ax):
    # Optional: add hatch to transform portion to make it visually distinct
    for i, container in enumerate(ax.containers):
        if i == 0:  # second stack (transform)
            for bar in container:
                bar.set_hatch('//')


def hatch_ax2(ax):
    # Optional: add hatch to transform portion to make it visually distinct
    for i, container in enumerate(ax.containers):
        if i == 1:  # second stack (transform)
            for bar in container:
                bar.set_hatch('\\\\')





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




data = []

data.append({
    "benchmark": "TensorUnfold",
    "memory usage": unfold2(),
    "type": "base"
})

data.append({
    "benchmark": "TensorUnfold",
    "memory usage": 1.0,
    "type": "savings"
})

data.append({
    "benchmark": "ImageAug",
    "memory usage": 1.0,
    "type": "savings"
})

data.append({
    "benchmark": "ImageAug",
    "memory usage": img_aug2(),
    "type": "base"
})



for ksize in [2,3,4,5]:
    ratio = im2col_usage2(3,512,512,ksize)
    data.append({
        "benchmark": f"Im2Col_{ksize}x{ksize}",
        "memory usage": 1.0,
        "type": "savings"
    })
    data.append({
        "benchmark": f"Im2Col_{ksize}x{ksize}",
        "memory usage": ratio,
        "type": "base"
    })


for ksize in [2,3,4]:
    ratio = vol2col_usage(3,128,128,16,ksize)
    data.append({
        "benchmark": f"Vol2Col_k{ksize}",
        "memory usage": 1.0,
        "type": "savings"
    })
    data.append({
        "benchmark": f"Vol2Col_k{ksize}",
        "memory usage": ratio,
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
hatch_ax2(ax)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=10, fontweight="bold")

ax.set_ylabel("Normalized Memory Usage", fontsize=12, fontweight="bold")
ax.set_title("Baseline vs. DTU Memory Usage Comparison", fontsize=12, fontweight="bold")
fig.legend(
    [ "", "DTU Memory Usage", "Baseline Memory Usage" ],  # labels
    loc="upper center",                   # position above all subplots
    ncol=4,                               # spread horizontally
    fontsize=10,
    frameon=False                         # optional: no box around legend
)

ax.set_ylim(0.1,64)
# Set y-axis to logarithmic scale
ax.set_yscale('log', base=2)

# Optional: customize the ticks (base 10)
ax.yaxis.set_major_locator(LogLocator(base=2, subs=None, numticks=6))
ax.yaxis.set_major_formatter(ScalarFormatter())  # show normal numbers instead of scientific notation
ax.set_yticks([0.5, 1, 2, 4, 8, 16, 32, 64])
for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(10)
plt.tight_layout(pad=3.0)


plt.savefig("figures/memory_savings.png", bbox_inches="tight")
plt.savefig("figures/memory_savings.pdf", bbox_inches="tight")
plt.show()