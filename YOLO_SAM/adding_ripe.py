from ultralytics import YOLO

RUN = "coco"   # set to "coco" in the second terminal

def main():
    if RUN == "continued":
        model = YOLO(r"D:\MTP-2\YOLO_SAM\runs\fullframe_coco_base\weights\best.pt")
        extra = dict(optimizer="AdamW", lr0=0.0005)
        name = "fullframe_3class_continued_v2"
    else:
        model = YOLO("yolo11n.pt")
        extra = {}
        name = "fullframe_3class_coco_v2"

    model.train(
        data=r"D:\MTP-2\YOLO_SAM\data_3class.yaml",
        epochs=80, imgsz=640, batch=16, device=0, workers=0,
        project=r"D:\MTP-2\YOLO_SAM\runs", name=name, **extra,
    )

if __name__ == "__main__":
    main()