import cv2
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = r"D:\MTP-2\runs\detect\train2\weights\best.pt"
# MODEL_PATH = r"D:\MTP-1\YOLO\runs\detect\train\weights\best.pt"
CAMERA_INDEX = 1          # same index used in your capture script
CONF_THRESHOLD = 0.5

# ─── LOAD MODEL ──────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ─── OPEN CAMERA ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Camera failed to open — try changing CAMERA_INDEX to 0, 1, or 2")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Running live YOLO inference. Press Q to quit.")
print("-" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_h, frame_w = frame.shape[:2]

    # Run inference on this frame
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # Print raw detection details to terminal
    boxes = results[0].boxes
    if len(boxes) == 0:
        print("No detections this frame.")
    else:
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = results[0].names[cls]

            x1, y1, x2, y2 = xyxy
            box_w = x2 - x1
            box_h = y2 - y1
            area_ratio = (box_w * box_h) / (frame_w * frame_h)

            print(f"Class: {cls_name:10s} | Conf: {conf:.2f} | "
                  f"Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] | "
                  f"Size: {box_w:.0f}x{box_h:.0f} | Area ratio: {area_ratio:.4f}")

    print("-" * 70)

    # Draw boxes/labels on the frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Live Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()