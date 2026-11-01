# make_cross_session_cm.py
import numpy as np
import matplotlib.pyplot as plt

OUT_PATH = r"D:\MTP-2\Report\images\cross_session_confusion_matrix.png"

labels = ["unripe", "ripe", "overripe"]
counts = np.array([
    [58, 2, 0],
    [0, 57, 3],
    [0, 2, 46],
])
row_totals = counts.sum(axis=1, keepdims=True)
normalized = counts / row_totals

fig, ax = plt.subplots(figsize=(6, 4.8))
im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Normalized)\nCross-Session Test Set", fontsize=12)
for i in range(3):
    for j in range(3):
        color = "white" if normalized[i, j] > 0.5 else "black"
        ax.text(j, i, f"{normalized[i,j]:.2f}\n({counts[i,j]})", ha="center", va="center", color=color, fontsize=11)
fig.colorbar(im, ax=ax, label="Fraction of true class")
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200)
print(f"Saved to {OUT_PATH}")