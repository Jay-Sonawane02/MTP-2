import cv2
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = r"D:\MTP-2\YOLO_SAM\runs\fullframe_3class_coco_v2\weights\best.pt"
IMAGE_PATH = r"D:\MTP-2\chamber_capture\overripe\overripe_000_p0_001256.jpg"   
CONF_THRESHOLD = 0.25

# ─── LOAD MODEL ──────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")
print(f"Classes: {model.names}")

# ─── RUN INFERENCE ────────────────────────────────────────────────────
results = model.predict(IMAGE_PATH, conf=CONF_THRESHOLD, verbose=False)

frame = cv2.imread(IMAGE_PATH)
boxes = results[0].boxes

if len(boxes) == 0:
    print("\nNo detections in this image.")
else:
    print(f"\nFound {len(boxes)} detection(s):\n")
    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = results[0].names[cls]

        x1, y1, x2, y2 = xyxy
        print(f"  Class: {cls_name:10s} | Conf: {conf:.2f} | Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# ─── SAVE + DISPLAY ANNOTATED IMAGE ───────────────────────────────────
annotated = results[0].plot()
output_path = r"D:\MTP-2\test_result.jpg"
cv2.imwrite(output_path, annotated)
print(f"\nAnnotated image saved to: {output_path}")

cv2.imshow("YOLO Prediction", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()