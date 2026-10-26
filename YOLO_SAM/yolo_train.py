from ultralytics import YOLO

def main():
    # Option A: start from COCO base (clean slate, no black-bg bias)
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=r"D:\MTP-2\YOLO_SAM\data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=r"D:\MTP-2\YOLO_SAM\runs",
        name="fullframe_coco_base"
    )

if __name__ == "__main__":
    main()