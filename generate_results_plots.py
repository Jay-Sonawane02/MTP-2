from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt"
TEST_ROOT = Path(r"D:\MTP-2\testing")
OUT_DIR = Path(r"D:\MTP-2\report_figures")
CONF = 0.25
CLASS_ORDER = ["unripe", "ripe", "overripe"]   # fixed, consistent axis order

OUT_DIR.mkdir(exist_ok=True)

model = YOLO(MODEL_PATH)
names = model.names
print(f"Loaded model: {MODEL_PATH}")
print(f"Classes: {names}\n")

# ─── Run inference, collect true/pred pairs ─────────────────────────
y_true, y_pred, y_conf = [], [], []

for cls_dir in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
    true = cls_dir.name
    images = sorted(cls_dir.glob("test*.jpg"), key=lambda p: int(p.stem.replace("test", "")))
    for img in images:
        boxes = model.predict(str(img), conf=CONF, verbose=False)[0].boxes
        if len(boxes) == 0:
            y_true.append(true); y_pred.append("none"); y_conf.append(0.0)
            continue
        top = boxes[boxes.conf.argmax()]
        y_true.append(true)
        y_pred.append(names[int(top.cls[0])])
        y_conf.append(float(top.conf[0]))

labels = CLASS_ORDER + (["none"] if "none" in y_pred else [])
n = len(labels)

# ─── Build confusion matrix (rows = true, cols = predicted) ────────
cm = np.zeros((n, n), dtype=int)
idx = {l: i for i, l in enumerate(labels)}
for t, p in zip(y_true, y_pred):
    cm[idx[t], idx[p]] += 1

# ─── Plot 1: confusion matrix heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix — Cross-Session Test Set")
for i in range(n):
    for j in range(n):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
fig.colorbar(im, ax=ax, label="Frame count")
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

# ─── Plot 2: per-class accuracy bar chart ───────────────────────────
per_class_acc = {}
for c in CLASS_ORDER:
    total = sum(1 for t in y_true if t == c)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
    per_class_acc[c] = correct / total if total else 0.0
overall_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

fig, ax = plt.subplots(figsize=(6, 4))
bars = list(per_class_acc.keys()) + ["Overall"]
vals = list(per_class_acc.values()) + [overall_acc]
colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]
ax.bar(bars, [v * 100 for v in vals], color=colors)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
ax.set_title("Per-Class Accuracy — Cross-Session Test Set")
for i, v in enumerate(vals):
    ax.text(i, v * 100 + 1.5, f"{v*100:.1f}%", ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "accuracy_per_class.png", dpi=200)
plt.close(fig)

# ─── Precision / Recall / F1 per class ──────────────────────────────
print("PER-CLASS METRICS")
print(f"{'Class':10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
metrics_rows = []
for c in CLASS_ORDER:
    ci = idx[c]
    tp = cm[ci, ci]
    fp = cm[:, ci].sum() - tp
    fn = cm[ci, :].sum() - tp
    support = cm[ci, :].sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics_rows.append((c, precision, recall, f1, support))
    print(f"{c:10s} {precision:10.3f} {recall:10.3f} {f1:10.3f} {support:10d}")

print(f"\nOverall accuracy: {overall_acc:.1%}  ({sum(1 for t,p in zip(y_true,y_pred) if t==p)}/{len(y_true)})")

# ─── Plot 3: metrics table as an image ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 2.2))
ax.axis("off")
table_data = [[c, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}", str(s)] for c, p, r, f1, s in metrics_rows]
table = ax.table(cellText=table_data,
                  colLabels=["Class", "Precision", "Recall", "F1-score", "Support"],
                  loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics_table.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved figures to: {OUT_DIR}")
print("  confusion_matrix.png")
print("  accuracy_per_class.png")
print("  metrics_table.png")
