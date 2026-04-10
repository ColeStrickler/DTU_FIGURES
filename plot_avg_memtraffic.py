import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/avg_memory_traffic_boom.csv")

pivot = df.pivot(index="benchmark", columns="type", values="memtraffic")

pivot.plot(kind="bar")

plt.ylabel("Average Total Memory Traffic (normalized)")
plt.title("DTU vs CPU Memory Traffic by Benchmark")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()