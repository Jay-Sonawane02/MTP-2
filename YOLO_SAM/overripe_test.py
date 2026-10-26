from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

MODELS = {
    "continued v2": r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_continued_v2\weights\best.pt",
    "coco v2":      r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt",
}
DS = Path(r"D:\MTP-2\YOLO_SAM\dataset_3class")
CONF = 0.25

for name, weights in MODELS.items():
    model = YOLO(weights)
    names = model.names
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    for split, tag in (("val", "HELD-OUT (never trained on)"),
                       ("train", "TRAINING (already seen - sanity check only)")):
        per_frame = defaultdict(int)
        per_tomato = defaultdict(lambda: defaultdict(float))
        n = 0
        for img in sorted((DS / split / "images").glob("overripe_*.jpg")):
            tid = img.stem.split("_")[2]
            per_tomato[tid]
            n += 1
            boxes = model.predict(str(img), conf=CONF, verbose=False)[0].boxes
            if len(boxes) == 0:
                per_frame["none"] += 1
                continue
            top = boxes[boxes.conf.argmax()]
            per_frame[names[int(top.cls[0])]] += 1
            for b in boxes:
                per_tomato[tid][names[int(b.cls[0])]] += float(b.conf[0])
        print(f"\n{tag}: {n} frames")
        print(f"  per frame: {dict(per_frame)}  -> overripe {per_frame['overripe']}/{n}")
        for tid, sc in sorted(per_tomato.items()):
            pred = max(sc, key=sc.get) if sc else "none"
            print(f"  overripe #{tid} -> {pred:9s} {'OK' if pred == 'overripe' else 'WRONG'}  "
                  f"{ {k: round(v, 2) for k, v in sc.items()} }")