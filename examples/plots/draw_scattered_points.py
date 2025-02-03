import matplotlib.pyplot as plt

labels = [
    "MS_GraphRAG",
    "RoG",
    "Fast-graphrag",
    "Graph-CoT",
    "Top-k SP",
    "Top-k CSP",
    "PolyG",
]
pos = [(17, 27), (10, 13), (30, 12), (50, 10), (5, 30), (17, 18), (10, 72)]
x_time = [15.82, 11.99, 36.30, 59.14, 10.13, 15.84, 13.62]
y_quality = [27.81, 19.27, 7.40, 5.52, 26.98, 19.27, 78.65]

# Create a scatter plot with labels
FONT_SIZE = 24
plt.figure(figsize=(10, 6))
plt.scatter(x_time[:-1], y_quality[:-1], color="dodgerblue", s=200, marker="o")
plt.scatter(x_time[-1], y_quality[-1], color="red", s=350, marker="*")
for i, txt in enumerate(labels):
    plt.annotate(txt, pos[i], fontsize=FONT_SIZE)
plt.xlim(5, 62)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.xlabel("Average End-to-end Latency (s)", fontsize=FONT_SIZE)
plt.ylabel("Average Win Rate (%)", fontsize=FONT_SIZE)
plt.grid(axis="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/scatter_quality_time.pdf", bbox_inches="tight")
