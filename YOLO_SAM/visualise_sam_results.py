import cv2
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────
DATASET_ROOT = r"D:\MTP-2\YOLO_SAM\fullframe_dataset"
OUTPUT_ROOT = r"D:\MTP-2\YOLO_SAM\validation_preview"
CLASS_NAMES = ["unripe", "ripe", "overripe"]
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 255)}  # BGR: unripe=green, ripe=red, overripe=magenta

img_dir = Path(DATASET_ROOT) / "images"
lbl_dir = Path(DATASET_ROOT) / "labels"
out_dir = Path(OUTPUT_ROOT)
out_dir.mkdir(parents=True, exist_ok=True)

images = sorted(img_dir.glob("*.jpg"))
print(f"Found {len(images)} images to validate")

drawn = 0
missing_labels = 0

for img_path in images:
    label_path = lbl_dir / f"{img_path.stem}.txt"

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  Could not read {img_path.name}")
        continue

    h, w = frame.shape[:2]

    if not label_path.exists():
        missing_labels += 1
        print(f"  No label file for {img_path.name}")
        cv2.imwrite(str(out_dir / img_path.name), frame)
        continue

    with open(label_path) as f:
        lines = f.read().strip().split("\n")

    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls_idx = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        color = CLASS_COLORS.get(cls_idx, (255, 255, 255))
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, cls_name, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(str(out_dir / img_path.name), frame)
    drawn += 1

print(f"\n=== Done ===")
print(f"Drawn: {drawn}")
print(f"Missing labels: {missing_labels}")
print(f"Saved to: {out_dir}")