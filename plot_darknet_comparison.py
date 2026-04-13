import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data/darknet_compare.csv")

df = pd.DataFrame(data)

df["im2col_cycles"] = pd.to_numeric(df["im2col_cycles"])
df["gemm_cycles"]   = pd.to_numeric(df["gemm_cycles"])
df["total_cycles"]  = pd.to_numeric(df["total_cycles"])

# normalize to percentages
df["im2col_pct"] = df["im2col_cycles"] / df["total_cycles"]
df["gemm_pct"]   = df["gemm_cycles"] / df["total_cycles"]
df["other_pct"]  = 1 - df["im2col_pct"] - df["gemm_pct"]

x = np.arange(len(df))

plt.figure(figsize=(6,4))

plt.bar(x, df["im2col_pct"], label="Im2Col")
plt.bar(x, df["gemm_pct"], bottom=df["im2col_pct"], label="GEMM")
plt.bar(x, df["other_pct"], bottom=df["im2col_pct"] + df["gemm_pct"], label="Other")

plt.xticks(x, df["platform"], rotation=20)
plt.ylabel("Fraction of Total Cycles")
plt.ylim(0, 1.0)
plt.legend()

plt.tight_layout()
plt.show()