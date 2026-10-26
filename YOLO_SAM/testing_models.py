from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

MODELS = {
    "continued v2": r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_continued_v2\weights\best.pt",
    "coco v2":      r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt",
}
TEST_ROOT = Path(r"D:\MTP-2\chamber_capture_TEST")
CONF = 0.25

def evaluate(weights):
    model = YOLO(weights)
    names = model.names
    labels = [names[i] for i in sorted(names)] + ["none"]
    frame_cm = defaultdict(lambda: defaultdict(int))        # true -> predicted -> count
    fruit_scores = defaultdict(lambda: defaultdict(float))  # (true, tomato id) -> class -> summed conf

    for cls_dir in sorted(p for p in TEST_ROOT.iterdir() if p.is_dir()):
        true = cls_dir.name
        for img in sorted(cls_dir.glob("*.jpg")):
            tid = img.stem.split("_")[1]
            fruit_scores[(true, tid)]
            boxes = model.predict(str(img), conf=CONF, verbose=False)[0].boxes
            if len(boxes) == 0:
                frame_cm[true]["none"] += 1
                continue
            top = boxes[boxes.conf.argmax()]
            frame_cm[true][names[int(top.cls[0])]] += 1
            for b in boxes:
                fruit_scores[(true, tid)][names[int(b.cls[0])]] += float(b.conf[0])
    return labels, frame_cm, fruit_scores

summary = {}
for name, weights in MODELS.items():
    labels, frame_cm, fruit_scores = evaluate(weights)

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print("PER-FRAME  (rows = true class, columns = predicted)")
    print(f"{'':10s}" + "".join(f"{l:>10s}" for l in labels) + "   accuracy")
    for true in sorted(frame_cm):
        row = frame_cm[true]
        total = sum(row.values())
        print(f"{true:10s}" + "".join(f"{row.get(l, 0):>10d}" for l in labels)
              + f"   {row.get(true, 0) / total:.1%}")
        summary[(name, true, "frame")] = f"{row.get(true, 0)}/{total}"

    print("\nPER-TOMATO  (scores summed over its frames)")
    tally = defaultdict(lambda: [0, 0])
    for (true, tid), sc in sorted(fruit_scores.items()):
        pred = max(sc, key=sc.get) if sc else "none"
        ok = pred == true
        tally[true][0] += ok
        tally[true][1] += 1
        print(f"  {true} #{tid} -> {pred:9s} {'OK' if ok else 'WRONG'}  "
              f"{ {k: round(v, 2) for k, v in sc.items()} }")
    for true, (ok, n) in tally.items():
        summary[(name, true, "tomato")] = f"{ok}/{n}"

print(f"\n{'=' * 70}\nSIDE BY SIDE\n{'=' * 70}")
classes = sorted({k[1] for k in summary})
print(f"{'':28s}" + "".join(f"{c + ' frames':>16s}{c + ' tomatoes':>18s}" for c in classes))
for name in MODELS:
    print(f"{name:28s}" + "".join(
        f"{summary.get((name, c, 'frame'), '-'):>16s}{summary.get((name, c, 'tomato'), '-'):>18s}"
        for c in classes))