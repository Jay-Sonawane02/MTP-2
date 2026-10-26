import os
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# ─── CONFIG ──────────────────────────────────────────────────────────
SAM2_CHECKPOINT = r"D:\MTP-1\YOLO\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
YOLO_MODEL_PATH = r"D:\MTP-2\runs\detect\train2\weights\best.pt"

CROPS_ROOT = r"D:\MTP-2\phase_a_crops"     # your existing real-background crops
OUTPUT_ROOT = r"D:\MTP-2\phase_a_blackbg"  # new black-bg versions for this test

DIRECT_BOX_IN_CROP = None  # will auto-detect since crop is already ~just the tomato

# ─── LOAD MODELS ─────────────────────────────────────────────────────
sam_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
predictor = SAM2ImagePredictor(sam_model)
yolo_model = YOLO(YOLO_MODEL_PATH)

def segment_to_black_bg(image):
    h, w = image.shape[:2]
    center_point = np.array([[w // 2, h // 2]])  # center of the crop = inside the tomato

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=np.array([1]),
        multimask_output=True
    )
    best_mask = masks[np.argmax(scores)].astype(bool)

    black_bg = np.zeros_like(image)
    black_bg[best_mask] = image[best_mask]
    return black_bg

def test_class(class_name):
    crop_dir = Path(CROPS_ROOT) / class_name
    out_dir = Path(OUTPUT_ROOT) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(crop_dir.glob("*.jpg"))
    print(f"\n=== TRUE CLASS: {class_name} ({len(images)} crops) ===")

    correct = 0
    no_detection = 0
    wrong = {}

    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))
        black_bg_image = segment_to_black_bg(image)

        # save for visual spot-checking
        Image.fromarray(black_bg_image).save(out_dir / img_path.name)

        results = yolo_model.predict(black_bg_image, conf=0.25, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            no_detection += 1
            print(f"  {img_path.name}: NO DETECTION")
            continue

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

if __name__ == "__main__":
    test_class("ripe")
    test_class("unripe")