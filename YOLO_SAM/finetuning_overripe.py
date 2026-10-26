import random, re, shutil
from collections import defaultdict
from pathlib import Path

SRC = Path(r"D:\MTP-2\YOLO_SAM\fullframe_dataset")
DST = Path(r"D:\MTP-2\YOLO_SAM\dataset_3class")
VAL_FRACTION = 0.15
SEED = 42

pattern = re.compile(r"^(unripe|ripe|overripe)_\1_(\d+)_p\d+_\d+$")

groups = defaultdict(lambda: defaultdict(list))   # class -> tomato id -> [stems]
for img in sorted((SRC / "images").glob("*.jpg")):
    m = pattern.match(img.stem)
    if m:
        groups[m.group(1)][m.group(2)].append(img.stem)

if DST.exists():
    shutil.rmtree(DST)
for split in ("train", "val"):
    (DST / split / "images").mkdir(parents=True)
    (DST / split / "labels").mkdir(parents=True)

random.seed(SEED)
for cls, tomatoes in groups.items():
    ids = sorted(tomatoes)
    random.shuffle(ids)
    n_val = min(max(1, round(len(ids) * VAL_FRACTION)), len(ids) - 1)
    val_ids = set(ids[:n_val])

    n_tr = n_va = 0
    for tid, stems in tomatoes.items():
        split = "val" if tid in val_ids else "train"
        for stem in stems:
            shutil.copy(SRC / "images" / f"{stem}.jpg", DST / split / "images")
            shutil.copy(SRC / "labels" / f"{stem}.txt", DST / split / "labels")
        if split == "val":
            n_va += len(stems)
        else:
            n_tr += len(stems)
    print(f"{cls:9s}: {len(ids)} tomatoes -> {len(ids)-n_val} train ({n_tr} frames), {n_val} val ({n_va} frames)")