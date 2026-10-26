import random
import shutil
from pathlib import Path

DATASET_ROOT = Path(r"D:\MTP-2\YOLO_SAM\fullframe_dataset")
SPLIT_ROOT = Path(r"D:\MTP-2\YOLO_SAM\dataset")

VAL_FRACTION = 0.15   # ~15% held out for validation
SEED = 42

img_dir = DATASET_ROOT / "images"
lbl_dir = DATASET_ROOT / "labels"

images = sorted(img_dir.glob("*.jpg"))
random.seed(SEED)
random.shuffle(images)

n_val = max(1, int(len(images) * VAL_FRACTION))
val_images = images[:n_val]
train_images = images[n_val:]

for split_name, split_images in [("train", train_images), ("val", val_images)]:
    (SPLIT_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (SPLIT_ROOT / split_name / "labels").mkdir(parents=True, exist_ok=True)

    for img_path in split_images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        shutil.copy(img_path, SPLIT_ROOT / split_name / "images" / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, SPLIT_ROOT / split_name / "labels" / lbl_path.name)

print(f"Train: {len(train_images)} images")
print(f"Val:   {len(val_images)} images")