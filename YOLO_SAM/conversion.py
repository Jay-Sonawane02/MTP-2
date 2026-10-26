from ultralytics import YOLO
YOLO(r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt").export(format="ncnn", imgsz=640)