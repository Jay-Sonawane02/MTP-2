from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = r"D:\MTP-2\runs\detect\train2\weights\best.pt"   # YOLO11n
# MODEL_PATH = r"D:\MTP-1\YOLO\runs\detect\train\weights\best.pt"
CROPS_ROOT = r"D:\MTP-2\phase_a_crops"

model = YOLO(MODEL_PATH)

for class_name in ["ripe", "unripe"]:
    crop_dir = Path(CROPS_ROOT) / class_name
    images = sorted(crop_dir.glob("*.jpg"))
    print(f"\n=== TRUE CLASS: {class_name} ({len(images)} crops) ===")

    correct = 0
    no_detection = 0
    wrong = {}

    for img_path in images:
        results = model.predict(str(img_path), conf=0.25, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            no_detection += 1
            print(f"  {img_path.name}: NO DETECTION")
            continue

        # take the highest-confidence box
        best_box = boxes[boxes.conf.argmax()]
        pred_class = results[0].names[int(best_box.cls[0])]
        conf = float(best_box.conf[0])

        if pred_class == class_name:
            correct += 1
        else:
            wrong[pred_class] = wrong.get(pred_class, 0) + 1
            print(f"  {img_path.name}: predicted '{pred_class}' (conf {conf:.2f}) — WRONG")

    total = len(images)
    print(f"\n  Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"  No detection: {no_detection}/{total}")
    print(f"  Wrong predictions breakdown: {wrong}")