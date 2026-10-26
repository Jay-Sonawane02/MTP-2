import cv2
import numpy as np
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = r"D:\MTP-2\runs\detect\train2\weights\best.pt"   # old MTP-1 YOLO11n
CAMERA_INDEX = 1
CONF_THRESHOLD = 0.25

SATURATION_THRESHOLD = 50
MIN_CONTOUR_AREA = 2000

model = YOLO(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def detect_tomato_regions(frame):
    """Naive OpenCV detection — no cap on number of boxes.
    Demonstrates the drawback: picks up every saturated region, not just tomatoes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    _, mask = cv2.threshold(s, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes

def classify_crop(crop):
    if crop.size == 0:
        return None, 0.0
    results = model.predict(crop, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, 0.0
    best_box = boxes[boxes.conf.argmax()]
    return results[0].names[int(best_box.cls[0])], float(best_box.conf[0])

print("\n[MULTIPLE BOXES DEMO] Press Q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    regions = detect_tomato_regions(frame)
    print(f"Boxes found this frame: {len(regions)}")

    for (x, y, w, h) in regions:
        crop = frame[y:y+h, x:x+w]
        cls_name, conf = classify_crop(crop)
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
        label = f"{cls_name} {conf:.2f}" if cls_name else "?"
        cv2.putText(display, label, (x, max(y-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(display, f"Boxes: {len(regions)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("OpenCV Detection - UNCONSTRAINED (drawback demo)", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()