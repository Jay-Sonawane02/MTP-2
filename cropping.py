import os
from pathlib import Path
import numpy as np
from PIL import Image

SAM2_CHECKPOINT = r"D:\MTP-1\YOLO\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

SOURCE_ROOT = r"D:\MTP-2\chamber_capture"
OUTPUT_ROOT = r"D:\MTP-2\phase_a_crops"
CROP_PADDING_PERCENT = 10

# ─── SAME BOXES AS THE RIPE SESSION — reused since setup didn't change ──
DIRECT_BOX = [753, 716, 1123, 968]
MIRROR_BOX = [788, 375, 1022, 545]

MIN_AREA_RATIO = 0.01
MAX_AREA_RATIO = 0.20

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

def load_predictor(checkpoint, config, device="cuda"):
    model = build_sam2(config, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)

def mask_to_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def is_plausible(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(h * w)
    return MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO

def crop_with_padding(image, bbox, padding_pct):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad = int(padding_pct / 100.0 * max(bw, bh))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]

def segment_from_box(predictor, image, box):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(box=np.array(box), multimask_output=True)
    best_idx = np.argmax(scores)
    return masks[best_idx].astype(bool)

def process_phase_a(class_name):
    src_dir = Path(SOURCE_ROOT) / class_name
    out_dir = Path(OUTPUT_ROOT) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = load_predictor(SAM2_CHECKPOINT, SAM2_CONFIG)
    images = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    print(f"Found {len(images)} frames in {src_dir}")

    total_saved = 0
    total_rejected = 0

    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))

        for label, box in [("direct", DIRECT_BOX), ("mirror", MIRROR_BOX)]:
            mask = segment_from_box(predictor, image, box)
            bbox = mask_to_bbox(mask)

            if bbox is None or not is_plausible(bbox, image.shape):
                print(f"  REJECTED ({label}) in {img_path.name} — bbox={bbox}")
                total_rejected += 1
                continue

            crop = crop_with_padding(image, bbox, CROP_PADDING_PERCENT)
            out_name = f"{img_path.stem}_{label}.jpg"
            Image.fromarray(crop).save(out_dir / out_name)
            total_saved += 1

    print(f"\n=== {class_name} done ===")
    print(f"Saved: {total_saved}")
    print(f"Rejected: {total_rejected}")

if __name__ == "__main__":
    process_phase_a("unripe")