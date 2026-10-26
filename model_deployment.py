from ultralytics import YOLO

model = YOLO(r"D:\MTP-2\runs\detect\train2\weights\best.pt")
model.export(format="onnx", imgsz=640, opset=12)   # well under the opset ≤17 limit