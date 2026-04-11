import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load CSV
df = pd.read_csv("data/darknet_data.csv")

# set indexv
df = df.set_index("type")
df = df.drop(columns=["RegularDRAMAccess"])
df = df.drop(columns=["DTUDramAccess"])


# normalize to CPU
cpu = df.loc["cpu"]
df_norm = df.div(cpu)

metrics = df_norm.columns
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(14,6))

bars_cpu = ax.bar(x - width, df_norm.loc["cpu"], width, label="CPU")
bars_dtu7 = ax.bar(x, df_norm.loc["dtu-7"], width, label="DTU-7")
bars_dtu6 = ax.bar(x + width, df_norm.loc["dtu-6"], width, label="DTU-6")

ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.set_ylabel("Normalized (CPU = 1)")
ax.axhline(1.0, linestyle="--")
ax.legend()

# label bars
def label_bars(bars):
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width()/2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

label_bars(bars_cpu)
label_bars(bars_dtu7)
label_bars(bars_dtu6)

plt.tight_layout()

plt.savefig("figures/darknet_test.pdf", bbox_inches="tight")
plt.savefig("figures/darknet_test.png", bbox_inches="tight")
plt.show()