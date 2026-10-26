import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

SAM2_CHECKPOINT = r"D:\MTP-1\YOLO\sam2\checkpoints\sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SOURCE_ROOT = r"D:\MTP-2\chamber_capture"
OUTPUT_ROOT = r"D:\MTP-2\YOLO_SAM\fullframe_dataset"
CLASS_NAMES = ["unripe", "ripe", "overripe"]

sam_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda")
generator = SAM2AutomaticMaskGenerator(
    sam_model, points_per_side=32, pred_iou_thresh=0.7,
    stability_score_thresh=0.92, min_mask_region_area=200,
)

def mask_to_yolo_box(mask, img_w, img_h):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return ((x1+x2)/2/img_w, (y1+y2)/2/img_h, (x2-x1)/img_w, (y2-y1)/img_h)

def is_tomato_colored(mask, image_rgb, min_fraction=0.5):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0][mask]
    sat = hsv[:, :, 1][mask]
    if len(hue) == 0:
        return False
    tomato_hue = ((hue <= 35) | (hue >= 170))
    high_sat = sat > 60
    valid_fraction = (tomato_hue & high_sat).sum() / len(hue)
    return valid_fraction >= min_fraction

def score_mask(mask_info, image_rgb, min_area=0.005, max_area=0.5, border_margin=4):
    mask = mask_info['segmentation'].astype(bool)
    h, w = image_rgb.shape[:2]
    area_ratio = mask.sum() / (h*w)
    if not (min_area <= area_ratio <= max_area):
        return None
    if mask[:border_margin,:].any() or mask[-border_margin:,:].any() or \
       mask[:,:border_margin].any() or mask[:,-border_margin:].any():
        return None
    if not is_tomato_colored(mask, image_rgb):
        return None
    ys, xs = np.where(mask)
    x1,y1,x2,y2 = xs.min(),ys.min(),xs.max(),ys.max()
    fill_ratio = mask.sum() / max(1,(x2-x1+1)*(y2-y1+1))
    cx, cy = (x1+x2)/2/w, (y1+y2)/2/h
    center_dist = abs(cx-0.5) + abs(cy-0.5)
    score = 2*area_ratio + fill_ratio - center_dist
    return score, mask

def top2_masks(masks, image_rgb):
    scored = [r for r in (score_mask(m, image_rgb) for m in masks) if r]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:2]

def process_class(class_name, class_idx):
    src_dir = Path(SOURCE_ROOT) / class_name
    img_out = Path(OUTPUT_ROOT) / "images"; img_out.mkdir(parents=True, exist_ok=True)
    lbl_out = Path(OUTPUT_ROOT) / "labels"; lbl_out.mkdir(parents=True, exist_ok=True)
    images = sorted(src_dir.glob("*.jpg"))
    print(f"\n=== {class_name} (class_idx={class_idx}) — {len(images)} frames ===")
    saved, one_mask, zero_mask = 0, 0, 0

    for img_path in images:
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        masks = generator.generate(image)
        top2 = top2_masks(masks, image)

        if len(top2) == 0:
            zero_mask += 1
            print(f"  {img_path.name}: 0 masks passed filters")
            continue
        if len(top2) == 1:
            one_mask += 1
            print(f"  {img_path.name}: only 1 mask passed filters")

        lines = []
        for score, mask in top2:
            box = mask_to_yolo_box(mask, w, h)
            if box:
                xc, yc, bw, bh = box
                lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        out_name = f"{class_name}_{img_path.stem}"
        Image.fromarray(image).save(img_out / f"{out_name}.jpg")
        with open(lbl_out / f"{out_name}.txt", "w") as f:
            f.write("\n".join(lines))
        saved += 1

    print(f"\n  Saved: {saved} | Only 1 mask: {one_mask} | 0 masks: {zero_mask}")

if __name__ == "__main__":
    # process_class("unripe", 0)
    # process_class("ripe", 1)
    process_class("overripe", 2)