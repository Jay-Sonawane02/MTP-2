from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

SAM2_CHECKPOINT = r"D:\MTP-1\YOLO\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SRC_DIR = Path(r"D:\MTP-2\chamber_capture_TEST\ripe")
OUTPUT_ROOT = Path(r"D:\MTP-2\YOLO_SAM\fullframe_dataset")
CLASS_IDX = 1                        # ripe
TRAIN_IDS = {"000", "001", "002"}    # move into training; 003-005 stay as the ripe test set
ID_OFFSET = 100                      # 000 -> 100, 001 -> 101, ...

sam_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
generator = SAM2AutomaticMaskGenerator(
    sam_model, points_per_side=32, pred_iou_thresh=0.7,
    stability_score_thresh=0.92, min_mask_region_area=200,
)

def mask_to_yolo_box(mask, img_w, img_h):
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return ((x1+x2)/2/img_w, (y1+y2)/2/img_h, (x2-x1)/img_w, (y2-y1)/img_h)

def is_tomato_colored(mask, image_rgb, min_fraction=0.5):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[:, :, 0][mask], hsv[:, :, 1][mask]
    if len(hue) == 0:
        return False
    ok = ((hue <= 35) | (hue >= 170)) & (sat > 60)
    return ok.sum() / len(hue) >= min_fraction

def score_mask(mask_info, image_rgb, min_area=0.005, max_area=0.5, border=4):
    mask = mask_info['segmentation'].astype(bool)
    h, w = image_rgb.shape[:2]
    if not (min_area <= mask.sum() / (h*w) <= max_area):
        return None
    if mask[:border].any() or mask[-border:].any() or mask[:, :border].any() or mask[:, -border:].any():
        return None
    if not is_tomato_colored(mask, image_rgb):
        return None
    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    fill = mask.sum() / max(1, (x2-x1+1)*(y2-y1+1))
    center_dist = abs((x1+x2)/2/w - 0.5) + abs((y1+y2)/2/h - 0.5)
    return 2*mask.sum()/(h*w) + fill - center_dist, mask

def top2_masks(masks, image_rgb):
    scored = [r for r in (score_mask(m, image_rgb) for m in masks) if r]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:2]

(OUTPUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / "labels").mkdir(parents=True, exist_ok=True)

saved = skipped = 0
for img_path in sorted(SRC_DIR.glob("*.jpg")):
    _, tid, pos, ts = img_path.stem.split("_")
    if tid not in TRAIN_IDS:
        continue
    image = np.array(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]
    top2 = top2_masks(generator.generate(image), image)
    if len(top2) < 2:   # a frame labelled with only one of its two tomatoes teaches the model the other is background
        print(f"  {img_path.name}: only {len(top2)} mask(s) passed - skipped")
        skipped += 1
        continue

    name = f"ripe_ripe_{int(tid) + ID_OFFSET:03d}_{pos}_{ts}"
    lines = []
    for _, mask in top2:
        xc, yc, bw, bh = mask_to_yolo_box(mask, w, h)
        lines.append(f"{CLASS_IDX} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    Image.fromarray(image).save(OUTPUT_ROOT / "images" / f"{name}.jpg")
    (OUTPUT_ROOT / "labels" / f"{name}.txt").write_text("\n".join(lines))
    saved += 1

print(f"Saved {saved}, skipped {skipped}")