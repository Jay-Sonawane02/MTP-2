from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    model.train(
        data=r"D:\MTP-2\YOLO_SAM\data_3class.yaml",
        epochs=80, imgsz=640, batch=16, device=0, workers=0,
        project=r"D:\MTP-2\YOLO_SAM\runs",
        name="fullframe_3class_coco",
    )

if __name__ == "__main__":
    main()