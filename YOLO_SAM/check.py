import os
for split in ["train", "val"]:
    img_dir = f"D:/MTP-2/YOLO_SAM/dataset/{split}/images"
    lbl_dir = f"D:/MTP-2/YOLO_SAM/dataset/{split}/labels"
    n_img = len(os.listdir(img_dir))
    n_lbl = len(os.listdir(lbl_dir))
    print(f"{split}: {n_img} images, {n_lbl} labels")