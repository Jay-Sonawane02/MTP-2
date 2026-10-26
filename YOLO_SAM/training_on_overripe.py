from ultralytics import YOLO

def main():
    model = YOLO(r"D:\MTP-2\YOLO_SAM\runs\fullframe_coco_base\weights\best.pt")
    model.train(
        data=r"D:\MTP-2\YOLO_SAM\data_3class.yaml",
        epochs=80, imgsz=640, batch=16, device=0, workers=0,
        optimizer="AdamW", lr0=0.0005,
        project=r"D:\MTP-2\YOLO_SAM\runs",
        name="fullframe_3class_continued",
    )

if __name__ == "__main__":
    main()