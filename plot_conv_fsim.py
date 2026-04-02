import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("data/conv_fsim.csv")

# Normalize times to the 'naive' benchmark
naive_time = df.loc[df['benchmark'] == 'naive', 'time'].values[0]
df['normalized'] = df['time'] / naive_time

# Sort so naive appears first if you want
df = df.sort_values('benchmark')

# Plot in black and white
plt.figure(figsize=(6, 4))
plt.bar(df['benchmark'], df['normalized'], color='black', edgecolor='black')

# Labels and title
plt.ylabel("Normalized Time (relative to naive)")
plt.xlabel("Benchmark")
plt.title("Normalized Benchmark Performance Rocket Core")

# Annotate each bar
for idx, val in enumerate(df['normalized']):
    plt.text(idx, val + 0.02, f"{val:.2f}×", ha='center', va='bottom')
plt.ylim(0, max(df['normalized']) * 1.2)  # add top padding
plt.tight_layout()

# Save to file
plt.savefig("figures/conv_fsim.png", dpi=300, bbox_inches='tight')

plt.show()
