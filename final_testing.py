from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt"
TEST_ROOT = Path(r"D:\MTP-2\testing")
CONF = 0.25

model = YOLO(MODEL_PATH)
names = model.names
print(f"Loaded model: {MODEL_PATH}")
print(f"Classes: {names}\n")

labels = [names[i] for i in sorted(names)] + ["none"]
frame_cm = defaultdict(lambda: defaultdict(int))   # true -> predicted -> count
per_frame_log = []

for cls_dir in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
    true = cls_dir.name
    images = sorted(cls_dir.glob("test*.jpg"), key=lambda p: int(p.stem.replace("test", "")))

    for img in images:
        boxes = model.predict(str(img), conf=CONF, verbose=False)[0].boxes
        if len(boxes) == 0:
            frame_cm[true]["none"] += 1
            per_frame_log.append((img.name, true, "none", 0.0))
            continue
        top = boxes[boxes.conf.argmax()]
        pred = names[int(top.cls[0])]
        conf = float(top.conf[0])
        frame_cm[true][pred] += 1
        per_frame_log.append((img.name, true, pred, conf))

print("PER-FRAME CONFUSION  (rows = true class, columns = predicted; each image is an independent data point)")
print(f"{'':10s}" + "".join(f"{l:>10s}" for l in labels) + "   accuracy")
total_correct, total_frames = 0, 0
for true in sorted(frame_cm):
    row = frame_cm[true]
    total = sum(row.values())
    correct = row.get(true, 0)
    total_correct += correct
    total_frames += total
    print(f"{true:10s}" + "".join(f"{row.get(l, 0):>10d}" for l in labels)
          + f"   {correct / total:.1%}")
print(f"\nOverall frame-level accuracy: {total_correct}/{total_frames} ({total_correct/total_frames:.1%})")

print("\nPER-IMAGE LOG")
for name, true, pred, conf in per_frame_log:
    mark = "OK" if pred == true else "WRONG"
    print(f"  {name:12s} true={true:9s} pred={pred:9s} conf={conf:.2f}  {mark}")
